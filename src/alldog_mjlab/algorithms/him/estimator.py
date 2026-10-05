"""HIM estimator：source encoder + target encoder + prototype representation objective。

数学行为移植自 official InternRobotics/HIMLoco
``rsl_rl/rsl_rl/modules/him_estimator.py``（commit ef289ac，2024-05-14），只做必要适配：

- 维度由 ``HIMSpec`` 提供，不出现 Black 固定维度；
- target encoder 输入显式化为 ``next_estimator_input``（frame 去掉 command + velocity）
  与 ``next_estimator_velocity``，不再从 critic privileged observation 里做 packing slice；
- optimizer 归属显式化：estimator optimizer 独立持有 source encoder / target encoder /
  prototypes，PPO optimizer 不包含它们（official 依赖 no_grad 而不是分账，见报告）。

official 默认网络（保持 released code，而不是论文里的 extractor 描述）：

```text
source encoder: history_dim -> 128 -> 64 -> velocity_dim + latent_dim
target encoder: target_dim  -> 128 -> 64 -> latent_dim
prototypes:     32 × latent_dim（temperature 3.0）
optimizer:      Adam, lr 1e-3, max_grad_norm 10.0
```
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules import MLP
from rsl_rl.utils import resolve_optimizer

from .spec import HIMSpec


class HIMEstimator(nn.Module):
    """HIM estimator（source encoder / target encoder / prototypes / loss）。"""

    def __init__(
        self,
        spec: HIMSpec,
        *,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        target_encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        num_prototypes: int = 32,
        temperature: float = 3.0,
        optimizer: str = "adam",
    ) -> None:
        super().__init__()
        if len(encoder_hidden_dims) < 1 or len(target_encoder_hidden_dims) < 1:
            raise ValueError("encoder_hidden_dims and target_encoder_hidden_dims must be non-empty.")

        self.spec = spec
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.num_prototypes = num_prototypes

        # Source encoder：输出 estimated velocity + raw latent。
        self.encoder = MLP(
            spec.history_dim,
            spec.velocity_dim + spec.latent_dim,
            hidden_dims=list(encoder_hidden_dims),
            activation=activation,
        )
        # Target encoder：输出 target latent。
        self.target = MLP(
            spec.target_encoder_input_dim,
            spec.latent_dim,
            hidden_dims=list(target_encoder_hidden_dims),
            activation=activation,
        )
        # Prototypes。
        self.proto = nn.Embedding(num_prototypes, spec.latent_dim)

        # Estimator optimizer 只拥有本模块参数（encoder / target / proto）。
        self.learning_rate = learning_rate
        self.optimizer = resolve_optimizer(optimizer)(self.parameters(), lr=learning_rate)

    def encode(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """source encoder：canonical history → (estimated velocity, L2-normalized latent)。"""
        parts = self.encoder(history)
        velocity = parts[..., : self.spec.velocity_dim]
        latent = F.normalize(parts[..., self.spec.velocity_dim :], dim=-1, p=2)
        return velocity, latent

    def encode_target(self, next_input: torch.Tensor) -> torch.Tensor:
        """target encoder：successor target input → L2-normalized target latent。"""
        return F.normalize(self.target(next_input), dim=-1, p=2)

    @torch.no_grad()
    def predict(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """推理用：estimated velocity + normalized latent（无梯度）。"""
        return self.encode(history)

    def losses(
        self,
        history: torch.Tensor,
        next_input: torch.Tensor,
        next_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """official HIM estimator objective：(velocity loss, representation swap loss)。

        official 语义：
        - prototype 权重在每次更新前先做一次 L2 normalize（in-place，no_grad）；
        - Sinkhorn 只在 no_grad 下产生软分配 q_s / q_t（swapped prediction 目标）；
        - swap loss = -0.5 * (q_s · log p_t + q_t · log p_s).mean()，
          其中 log p 是 temperature 缩放后的 log-softmax；
        - velocity loss = MSE(estimated velocity, successor true scaled base lin vel)。
        """
        velocity, source_latent = self.encode(history)
        target_latent = self.encode_target(next_input)

        with torch.no_grad():
            normalized_proto = F.normalize(self.proto.weight.data.clone(), dim=-1, p=2)
            self.proto.weight.copy_(normalized_proto)

        score_s = source_latent @ self.proto.weight.T
        score_t = target_latent @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(velocity, next_velocity)
        return estimation_loss, swap_loss

    def update(
        self,
        history: torch.Tensor,
        next_input: torch.Tensor,
        next_velocity: torch.Tensor,
        lr: float | None = None,
    ) -> tuple[float, float]:
        """official estimator update：loss → zero_grad → backward → clip → step。"""
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        estimation_loss, swap_loss = self.losses(history, next_input, next_velocity)
        loss = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    """official HIM Sinkhorn normalization（逐行/逐列交替归一化，iters 次）。"""
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for _ in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T
