"""Scaled Conjugate Gradient (SCG) Optimizer for PyTorch.

Implementation based on:
  "A Scaled Conjugate Gradient Algorithm for Fast Supervised Learning"
  by Martin Føslette Møller, Neural Networks, 6(4), 525-533, 1993

Author: Based on Møller 1993 and NETLAB by Ian Nabney
"""

from __future__ import annotations

import math
import sys
from typing import Callable, Iterator

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer


class SCG(Optimizer):
    """Scaled Conjugate Gradient optimizer following Møller 1993.

    SCG is a second-order optimization method that avoids expensive line searches
    by using a Levenberg-Marquardt approach to scale the step size. It's particularly
    effective for deterministic (full-batch) optimization problems.

    Args:
        params: Iterable of parameters to optimize
        sigma0: Initial scale for second-order approximation (default: 1e-6)
        lambda_init: Initial trust region parameter (default: 1e-6)
        lambda_min: Minimum trust region parameter (default: 1e-15)
        lambda_max: Maximum trust region parameter (default: 1e20)

    Example:
        >>> model = nn.Linear(10, 1)
        >>> optimizer = SCG(model.parameters())
        >>> X, y = get_batch()
        >>>
        >>> def compute_loss():
        ...     return nn.functional.mse_loss(model(X), y)
        >>>
        >>> for _ in range(100):
        ...     loss = optimizer.step(compute_loss)
        ...     print(f"Loss: {loss:.4f}")

    Note:
        Unlike standard PyTorch optimizers, SCG.step() requires a loss function
        (not a closure that calls backward()). The optimizer handles all gradient
        computations internally.
    """

    def __init__(
        self,
        params: Iterator[Parameter],
        sigma0: float = 1e-6,
        lambda_init: float = 1e-6,
        lambda_min: float = 1e-15,
        lambda_max: float = 1e20,
    ):
        defaults = dict(
            sigma0=sigma0,
            lambda_init=lambda_init,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )
        super().__init__(params, defaults)

        # Validate single param group (SCG operates on all params as one vector)
        if len(self.param_groups) != 1:
            raise ValueError("SCG doesn't support per-parameter options (param groups)")

        self._params = self.param_groups[0]["params"]
        self._numel = sum(p.numel() for p in self._params)

    def _gather_flat_grad(self) -> Tensor:
        """Flatten gradients into a single vector."""
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new_zeros(p.numel())
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _get_flat_params(self) -> Tensor:
        """Get all parameters as a flat vector."""
        return torch.nn.utils.parameters_to_vector(self._params)

    def _set_flat_params(self, flat_params: Tensor) -> None:
        """Set parameters from a flat vector."""
        torch.nn.utils.vector_to_parameters(flat_params, self._params)

    def _compute_grad(self, loss_fn: Callable[[], Tensor]) -> tuple[Tensor, Tensor]:
        """Compute loss and negative gradient (search direction)."""
        for p in self._params:
            if p.grad is not None:
                p.grad.zero_()

        loss = loss_fn()
        loss.backward()

        # Return negative gradient (descent direction) and loss value
        return -self._gather_flat_grad(), loss.detach()

    def _compute_loss(self, loss_fn: Callable[[], Tensor]) -> Tensor:
        """Compute loss without gradient."""
        with torch.no_grad():
            return loss_fn().detach()

    @torch.no_grad()
    def step(self, loss_fn: Callable[[], Tensor]) -> Tensor:
        """Perform a single optimization step.

        Args:
            loss_fn: A callable that computes the loss (forward pass only).
                     The optimizer will call backward() internally as needed.

        Returns:
            The loss value after the step (or current loss if step was rejected).
        """
        sigma0 = self.defaults["sigma0"]
        lambda_min = self.defaults["lambda_min"]
        lambda_max = self.defaults["lambda_max"]

        # Initialize state on first call
        state = self.state.setdefault("global", {})
        if not state:
            with torch.enable_grad():
                r, f = self._compute_grad(loss_fn)

            state["step"] = 0
            state["lambda"] = self.defaults["lambda_init"]
            state["lambda_bar"] = 0.0
            state["success"] = True
            state["n_success"] = 0
            state["r"] = r  # r = -E'(w), the negative gradient
            state["f"] = f  # f = E(w), the loss
            state["p"] = r.clone()  # p = search direction
            state["p_norm_sq"] = (r @ r).item()  # |p|²

        step = state["step"]
        lam = state["lambda"]
        lam_bar = state["lambda_bar"]
        success = state["success"]
        n_success = state["n_success"]
        r = state["r"]
        f = state["f"]
        p = state["p"]
        p_norm_sq = state["p_norm_sq"]

        w = self._get_flat_params()

        # Step 2: Calculate second-order information if previous step was successful
        if success:
            if p_norm_sq < sys.float_info.epsilon:
                # Gradient is essentially zero, we've converged
                return f

            sigma = sigma0 / math.sqrt(p_norm_sq)

            # Compute gradient at perturbed position: E'(w + σp)
            self._set_flat_params(w + sigma * p)
            with torch.enable_grad():
                r_perturbed, _ = self._compute_grad(loss_fn)
            self._set_flat_params(w)

            # s ≈ H·p via finite difference: s = (E'(w+σp) - E'(w)) / σ
            # Note: r = -E', so s = (r - r_perturbed) / σ = -(E'_perturbed - E') / σ
            # Actually: r_perturbed = -E'(w+σp), r = -E'(w)
            # s = (E'(w+σp) - E'(w)) / σ = (-r_perturbed - (-r)) / σ = (r - r_perturbed) / σ
            s = (r - r_perturbed) / sigma

            # δ = p·s = p·(H·p) ≈ curvature along p
            delta = (p @ s).item()
        else:
            # Reuse delta from previous iteration
            delta = state.get("delta", 0.0)

        # Step 3: Scale delta with trust region parameter
        delta = delta + (lam - lam_bar) * p_norm_sq

        # Step 4: Make Hessian positive definite if needed
        if delta <= 0:
            lam_bar = 2 * (lam - delta / p_norm_sq)
            delta = -delta + lam * p_norm_sq
            lam = lam_bar

        if delta == 0:
            # Edge case: can't determine step size
            return f

        # Step 5: Calculate step size
        mu = (p @ r).item()  # μ = p·r
        alpha = mu / delta  # α = μ/δ

        # Compute loss at candidate position
        w_new = w + alpha * p
        self._set_flat_params(w_new)
        f_new = self._compute_loss(loss_fn)

        # Step 6: Calculate comparison ratio
        # Δ = 2δ(E(w) - E(w + αp)) / μ²
        Delta = 2 * delta * (f.item() - f_new.item()) / (mu * mu)

        # Step 7: Update if successful
        if Delta >= 0:
            success = True
            lam_bar = 0.0
            n_success += 1

            # Accept the new position
            r_old = r.clone()
            f = f_new

            # Compute gradient at new position
            with torch.enable_grad():
                r, _ = self._compute_grad(loss_fn)

            # Restart or update search direction
            if n_success % self._numel == 0:
                # Restart: reset to steepest descent
                p = r.clone()
                n_success = 0
            else:
                # Polak-Ribière formula: β = (|r_new|² - r_new·r_old) / μ
                # Note: mu = p·r_old (from before the update)
                beta = ((r @ r).item() - (r @ r_old).item()) / mu
                p = r + beta * p

            p_norm_sq = (p @ p).item()

            # Reduce trust region parameter if very successful
            if Delta >= 0.75:
                lam = max(0.25 * lam, lambda_min)
        else:
            # Reject step, restore parameters
            self._set_flat_params(w)
            success = False
            lam_bar = lam

        # Step 8: Increase trust region parameter if not very successful
        if Delta < 0.25 and p_norm_sq > 0:
            lam = min(lam + delta * (1 - Delta) / p_norm_sq, lambda_max)

        # Update state
        state["step"] = step + 1
        state["lambda"] = lam
        state["lambda_bar"] = lam_bar
        state["success"] = success
        state["n_success"] = n_success
        state["r"] = r
        state["f"] = f
        state["p"] = p
        state["p_norm_sq"] = p_norm_sq
        state["delta"] = delta

        return f

    def reset_state(self) -> None:
        """Reset optimizer state (useful when starting a new optimization)."""
        self.state.clear()
