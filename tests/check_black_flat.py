"""Run Black asset, rollout/reset, and standard PPO smoke checks.

From the repository: uv run python tests/check_black_flat.py
Artifacts are written to --output (a temporary directory by default).
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from nwwolf_mjlab.robots.black.black_constants import (
    BLACK_ACTION_SCALE, BLACK_FOOT_NAMES, BLACK_JOINT_NAMES,
    INIT_STATE, get_black_robot_cfg,
)
from nwwolf_mjlab.tasks.velocity.black.env_cfgs import black_flat_env_cfg
from nwwolf_mjlab.tasks.velocity.black.rl_cfg import black_ppo_runner_cfg


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), "Non-finite tensor"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix='black-flat-check-'))
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    robot = Entity(get_black_robot_cfg())
    model = robot.spec.compile()
    assert (model.nq, model.nv, model.nu, model.nkey) == (19, 18, 12, 1)
    assert robot.joint_names == BLACK_JOINT_NAMES
    expected = np.array([INIT_STATE.joint_pos[name] for name in BLACK_JOINT_NAMES])
    np.testing.assert_allclose(model.key_qpos[0, :3], [0, 0, .45])
    np.testing.assert_allclose(model.key_qpos[0, 7:], expected)
    np.testing.assert_array_equal(model.actuator_trnid[:, 0], np.arange(1, 13))
    np.testing.assert_allclose(model.actuator_forcerange, np.tile([-20, 20], (12, 1)))
    for leg in BLACK_FOOT_NAMES:
        geom = model.geom(f'{leg}_foot_collision')
        assert geom.contype != 0 and geom.conaffinity != 0
        assert geom.group == 3
        model.site(leg)
    for name in ('imu_ang_vel', 'imu_lin_vel', 'imu_lin_acc', 'root_angmom'):
        model.sensor(name)

    cfg = black_flat_env_cfg()
    cfg.seed = 42
    cfg.scene.num_envs = 4
    # Short episodes exercise timeout and automatic resets during both rollouts and PPO.
    cfg.episode_length_s = .4
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = None
    report = {'device': args.device, 'num_envs': 4, 'control_dt': env.step_dt}
    try:
        wrapped = RslRlVecEnvWrapper(env)
        data = env.scene['robot'].data
        expected_t = torch.tensor(expected, device=args.device, dtype=torch.float32)
        torch.testing.assert_close(data.default_joint_pos[0], expected_t)
        torch.testing.assert_close(data.joint_pos[0], expected_t)
        # Default reset adds 1--5 cm above the nominal initial height.
        assert torch.all((data.root_link_pos_w[:,2] >= .459) & (data.root_link_pos_w[:,2] <= .501))
        for actuator in env.scene['robot'].actuators:
            assert torch.all(actuator.stiffness == 40.)
            assert torch.all(actuator.damping == (1.0 if actuator.cfg.target_names_expr == ('RL_thigh_joint',) else 1.2))
        action = env.action_manager.get_term('joint_pos')
        assert action.target_names == list(BLACK_JOINT_NAMES)
        assert action.scale == BLACK_ACTION_SCALE == .25
        probe = torch.zeros(4, 12, device=args.device)
        probe[:, 0] = 1.
        env.action_manager.process_action(probe)
        env.action_manager.apply_action()
        target = data.joint_pos_target
        torch.testing.assert_close(target[:,0], data.default_joint_pos[:,0]+.25-data.encoder_bias[:,0])
        torch.testing.assert_close(target[:,1:], data.default_joint_pos[:,1:]-data.encoder_bias[:,1:])
        wrapped.reset()
        initial = wrapped.get_observations()
        report['observations'] = {k:list(v.shape) for k,v in initial.items()}
        contact = env.scene['feet_ground_contact']
        assert tuple(contact.primary_names) == tuple(f'{leg}_foot_collision' for leg in BLACK_FOOT_NAMES)
        for mode in ('zero', 'random'):
            wrapped.reset()
            resets, contacts = 0, torch.zeros(4, dtype=torch.bool, device=args.device)
            for _ in range(100):
                actions = torch.zeros(4,12,device=args.device)
                if mode == 'random':
                    actions.uniform_(-1.,1.)
                obs, rewards, dones, extras = wrapped.step(actions)
                finite(obs)
                finite(rewards)
                finite(data.joint_pos)
                finite(data.joint_vel)
                finite(contact.data.force)
                assert torch.all(data.actuator_force.abs() <= 20.0001)
                contacts |= contact.data.found.bool().reshape(4,4).any(0)
                resets += int(dones.sum())
            assert resets > 0, 'No automatic reset observed'
            assert contacts.all(), f'Missing foot contacts: {contacts.tolist()}'
            before = data.joint_pos[1:].clone()
            env.reset(env_ids=torch.tensor([0],device=args.device))
            torch.testing.assert_close(data.joint_pos[0], expected_t)
            torch.testing.assert_close(data.joint_pos[1:], before)
            report[mode] = {'steps':100, 'resets':resets, 'contact_seen_each_foot':contacts.tolist()}

        rcfg = black_ppo_runner_cfg()
        rcfg.logger = 'tensorboard'
        rcfg.upload_model = False
        rcfg.num_steps_per_env = 24
        rcfg.algorithm.num_learning_epochs = 2
        rcfg.algorithm.num_mini_batches = 2
        runner = VelocityOnPolicyRunner(wrapped, asdict(rcfg), str(output), args.device)
        before = [p.detach().clone() for p in runner.alg.actor.parameters()]
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
        after = list(runner.alg.actor.parameters())
        for parameter in after:
            finite(parameter)
        assert any(not torch.equal(a,b) for a,b in zip(before,after)), 'Actor did not update'
        checkpoints = sorted(output.glob('model_*.pt'))
        assert checkpoints, 'No checkpoint written'
        saved = torch.load(checkpoints[-1], weights_only=False, map_location='cpu')
        finite(saved)
        # Resume/play starts with a fresh runner. Reusing a trained instance
        # hits upstream inference-tensor normalizer buffers in this dependency version.
        reloaded = VelocityOnPolicyRunner(wrapped, asdict(rcfg), device=args.device)
        reloaded.load(str(checkpoints[-1]), map_location=args.device)
        policy = reloaded.get_inference_policy(device=args.device)
        with torch.inference_mode():
            finite(policy(wrapped.get_observations()))
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        accumulator = EventAccumulator(str(output)).Reload()
        tags = [t for t in accumulator.Tags()['scalars'] if 'loss' in t.lower()]
        assert tags, 'No loss metrics logged'
        for tag in tags:
            assert all(np.isfinite(e.value) for e in accumulator.Scalars(tag)), tag
        report['ppo'] = {'iterations':2, 'steps_per_env':24, 'checkpoint_reload':True, 'loss_tags':tags}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        print(f'PASS: artifacts in {output}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
