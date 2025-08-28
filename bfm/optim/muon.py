"""Muon is like the newest and coolest optimizer that works better than Adam."""

import torch


def _zeroth_power_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    abc: tuple = (3.4445, -4.7750, 2.0315)
) -> torch.Tensor:
    assert len(G.shape) == 2
    a, b, c = abc
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


def orthogonalize(G: torch.Tensor) -> torch.Tensor:
    return _zeroth_power_via_newtonschulz5(G, steps=10, eps=1e-8, abc=(3, -3.2, 1.2))


class Muon(torch.optim.Optimizer):
    """
    Muon is a new optimizer that works better than Adam.

    Args:
        params (torch.nn.Parameter): Parameters to optimize.
        lr (float): Learning rate.
        momentum (float): Momentum factor.
        nesterov (bool): Whether to use Nesterov momentum.
        weight_decay (float): Weight decay (L2 penalty).
        backend (str): Backend to use for optimization.
        backend_steps (int): Number of steps for backend optimization.
    """

    def __init__(
        self,
        params: torch.nn.Parameter,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        backend: str = "newtonschulz5",
        backend_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            backend=backend,
            backend_steps=backend_steps,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            zeropower_backend = _zeroth_power_via_newtonschulz5

            for i, p in enumerate(group["params"]):
                g = p.grad
                # assert g is not None
                # print(f"Param {i} has shape {p.shape}, and grad % of nan values: {torch.sum(torch.isnan(g)).item() / g.numel() * 100:.2f}%, and % of zeros: {torch.sum(g == 0).item() / g.numel() * 100:.2f}%")

                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_backend(g, steps=group["backend_steps"])
                # g.mul_(0.2 * max(g.shape[0], g.shape[1])**0.5) --- from moonlight paper
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["weight_decay"] * lr)
                # p.data.add_(g, alpha=-lr * max(1, (g.shape[-2] / g.shape[-1]))**0.5) #XXX

                # print(f" == After Muon, % of zeros in grad: {torch.sum(g == 0).item() / g.numel() * 100:.2f}%")
                p.data.add_(g, alpha=-lr)
