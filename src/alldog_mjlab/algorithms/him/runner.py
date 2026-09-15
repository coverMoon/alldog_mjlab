"""mjlab runner interface for the isolated HIM algorithm."""
from pathlib import Path
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from .actor_critic import HIMActorCritic
from .ppo import HIMPPO


class HIMRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("HIMRunner currently supports a single training process; use --gpu-ids 0.")
        self.env, self.cfg, self.device = env, train_cfg, device
        self.log_dir = Path(log_dir) if log_dir else None
        model = HIMActorCritic(270, 238, 45, 12, **train_cfg["policy"]).to(device)
        self.alg = HIMPPO(model, device=device, **train_cfg["him_algorithm"])
        self.alg.init_storage(env.num_envs, train_cfg["num_steps_per_env"], [270], [238], [12])
        self.current_learning_iteration = 0
        self.writer = None
        self.obs = env.get_observations()

    def add_git_repo_to_log(self, path):
        # Source snapshots and all effective task/agent configs are saved by mjlab CLI.
        pass

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=self.env.max_episode_length)
        self.alg.train_mode()
        obs = self.env.get_observations()
        end = self.current_learning_iteration + num_learning_iterations
        try:
            for iteration in range(self.current_learning_iteration, end):
                with torch.inference_mode():
                    for _ in range(self.cfg["num_steps_per_env"]):
                        actions = self.alg.act(obs["actor"], obs["critic"])
                        obs, rewards, dones, extras = self.env.step(actions)
                        next_critic = obs["critic"].clone()
                        ids = extras.get("him_terminal_ids")
                        if ids is not None:
                            next_critic[ids] = extras["him_terminal_critic"]
                        self.alg.process_env_step(rewards, dones, extras, next_critic)
                        if self.writer:
                            for key, value in extras.get("log", {}).items():
                                self.writer.add_scalar(key, value, iteration)
                    self.alg.compute_returns(obs["critic"])
                losses = self.alg.update()
                if not all(torch.isfinite(torch.tensor(x)) for x in losses):
                    raise RuntimeError(f"Non-finite HIM losses: {losses}")
                self.current_learning_iteration = iteration + 1
                if self.writer:
                    for key, value in zip(("value", "surrogate", "estimation", "swap", "symmetry"), losses):
                        self.writer.add_scalar("Loss/"+key, value, iteration)
                print(f"HIM iteration {iteration+1}/{end}: losses={losses}", flush=True)
                if self.log_dir and self.current_learning_iteration % self.cfg["save_interval"] == 0:
                    self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")
        finally:
            if self.log_dir:
                self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")
            if self.writer:
                self.writer.close()

    def save(self, path, infos=None):
        command = self.env.unwrapped.command_manager.get_term("twist")
        torch.save({"model_state_dict":self.alg.actor_critic.state_dict(), "optimizer_state_dict":self.alg.optimizer.state_dict(), "estimator_optimizer_state_dict":self.alg.actor_critic.estimator.optimizer.state_dict(), "iter":self.current_learning_iteration, "learning_rate":self.alg.learning_rate, "command_range_x":command.cfg.ranges.lin_vel_x, "infos":infos}, path)

    def load(self, path, load_optimizer=True, load_cfg=None, strict=True, map_location=None, **kwargs):
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if load_cfg is not None:
            load_optimizer = False
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(checkpoint["estimator_optimizer_state_dict"])
            self.alg.learning_rate = checkpoint.get("learning_rate", self.alg.optimizer.param_groups[0]["lr"])
            self.current_learning_iteration = checkpoint.get("iter", 0)
            if "command_range_x" in checkpoint:
                self.env.unwrapped.command_manager.get_term("twist").cfg.ranges.lin_vel_x = tuple(checkpoint["command_range_x"])
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        model = self.alg.actor_critic.to(device or self.device).eval()
        def policy(obs):
            with torch.inference_mode():
                return model.act_inference(obs["actor"])
        return policy

    def export_policy_to_onnx(self, path, filename="policy.onnx", verbose=False):
        import copy
        from torch import nn
        class Policy(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, history):
                return self.model.act_inference(history)
        model = Policy(copy.deepcopy(self.alg.actor_critic).cpu().eval())
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.onnx.export(model, (torch.zeros(1,270),), str(Path(path)/filename), input_names=["history"], output_names=["actions"], opset_version=17, dynamo=False, verbose=verbose)
