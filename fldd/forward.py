import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedForwardProcess(nn.Module):
    """Learned element-wise forward (noising) process for binary data.

    For each timestep t in {1, ..., T}, we learn a flip probability alpha_t
    such that:
        q_phi(z_t = x | x) = 1 - alpha_t
        q_phi(z_t != x | x) = alpha_t

    Parameterization
    ----------------
    Unconstrained logits are mapped to monotone alphas in (alpha_min, 0.5):

        alpha_t = 0.5 * sigmoid( cumsum(softplus(logits)) + sigmoid_offset )

    The resulting alpha range is (0.5 * sigmoid(sigmoid_offset), 0.5).

    The original code used sigmoid_offset = -2.0, giving a floor of
    alpha_min = 0.5 * sigmoid(-2) ~= 0.0596. In all of our trained E2 runs
    the optimizer drove every "early" alpha exactly to this floor (max
    deviation 5e-5 across 18 ckpts), which is the parameterization
    saturating, not the data choosing alpha ~ 0.06. To allow the schedule
    actual freedom, pass a more-negative sigmoid_offset (e.g. -6 gives
    alpha_min ~= 0.00124). Set fixed_alphas instead to bypass learning
    altogether (E4 frozen-schedule reruns).

    Backward compatibility: when loading an existing checkpoint, callers
    that don't pass sigmoid_offset will keep the historical -2.0 default
    so older results reproduce exactly.

    Fixed alphas
    ------------
    If fixed_alphas is provided, the schedule is non-trainable: get_alphas
    always returns the given tensor. This is the simplest way to control
    for the schedule when comparing block sizes (E4 T=2 confound, B4).
    """

    HISTORICAL_OFFSET = -2.0  # original code's value, kept for ckpt compat

    def __init__(self, T, sigmoid_offset=HISTORICAL_OFFSET, fixed_alphas=None):
        super().__init__()
        self.T = T
        self.sigmoid_offset = float(sigmoid_offset)

        if fixed_alphas is not None:
            alphas = torch.as_tensor(fixed_alphas, dtype=torch.float32)
            if alphas.shape != (T,):
                raise ValueError(
                    f"fixed_alphas must have shape (T,) = ({T},), "
                    f"got {tuple(alphas.shape)}"
                )
            if (alphas <= 0).any() or (alphas > 0.5).any():
                raise ValueError(
                    f"fixed_alphas must lie in (0, 0.5]; got min={alphas.min()}"
                    f" max={alphas.max()}"
                )
            if not torch.all(alphas[1:] >= alphas[:-1]):
                # warn but allow: monotone is a modelling choice, not a hard
                # requirement of the math
                import warnings
                warnings.warn("fixed_alphas is not monotone-increasing")
            self.register_buffer("_fixed_alphas", alphas)
            # still need a logits param so old code paths that touch it
            # don't error; flagged non-trainable
            self.logits = nn.Parameter(torch.zeros(T), requires_grad=False)
            self._is_fixed = True
        else:
            # NOTE: do NOT register a _fixed_alphas buffer here so that the
            # state_dict matches the original (logits-only) format. This
            # keeps load_state_dict(..., strict=True) working for existing
            # checkpoints saved by the old code.
            self.logits = nn.Parameter(torch.zeros(T))
            self._is_fixed = False

    @property
    def alpha_floor(self):
        """The parameterization floor on alpha_1 (smallest reachable alpha)."""
        import math
        return 0.5 / (1.0 + math.exp(-self.sigmoid_offset))

    def get_alphas(self):
        """Return flip probabilities alpha_1, ..., alpha_T.

        Monotonically increasing in t via cumulative softplus.

        Range: (alpha_floor, 0.5) where alpha_floor = 0.5 * sigmoid(offset).
        With the historical offset of -2.0, floor ~= 0.0596.
        With offset = -6.0, floor ~= 0.00124.
        """
        if self._is_fixed:
            return self._fixed_alphas

        increments = F.softplus(self.logits)
        cumulative = torch.cumsum(increments, dim=0)
        alphas = 0.5 * torch.sigmoid(cumulative + self.sigmoid_offset)
        return alphas

    def q_zt_given_x(self, x, t_idx):
        """Compute q(z_t = 1 | x) for binary x.

        Args:
            x: binary tensor (B, 1, H, W), values in {0, 1}
            t_idx: integer timestep index (0-based, so t=1 is index 0)

        Returns:
            prob_one: probability that z_t = 1, shape (B, 1, H, W)
        """
        alphas = self.get_alphas()
        alpha_t = alphas[t_idx]
        prob_one = x * (1.0 - alpha_t) + (1.0 - x) * alpha_t
        return prob_one

    def sample_zt(self, x, t_idx):
        """Sample z_t ~ q(z_t | x). Returns (z_t, prob_one)."""
        prob_one = self.q_zt_given_x(x, t_idx)
        z_t = torch.bernoulli(prob_one)
        return z_t, prob_one

    # NOTE: there is no q_posterior method on this class because FLDD's
    # forward process is non-Markovian: q(z_t | x) is defined directly
    # per t, so z_s and z_t are conditionally independent given x, and
    # q(z_s | z_t, x) collapses to q(z_s | x) — call q_zt_given_x at
    # s_idx instead. (Earlier versions of this code exposed a
    # q_posterior wrapper that did exactly that and was never called;
    # removed for clarity.)

    def kl_prior(self, x):
        """KL[q(z_T | x) || p(z_T)] with p(z_T) = Uniform Bernoulli(0.5).

        Per-pixel: q*log(2q) + (1-q)*log(2(1-q)).
        """
        prob_one = self.q_zt_given_x(x, self.T - 1)
        eps = 1e-8
        p = prob_one.clamp(eps, 1.0 - eps)
        kl = p * torch.log(2.0 * p) + (1.0 - p) * torch.log(2.0 * (1.0 - p))
        return kl.sum(dim=(1, 2, 3)).mean()


def make_forward_process(T, sigmoid_offset=None, fixed_alphas=None):
    """Convenience constructor.

    If sigmoid_offset is None, falls back to the historical -2.0 (for
    reproducing existing results). For new training, pass an explicit
    value like -6.0 (alpha_min ~= 0.00124) to give the schedule real
    freedom.
    """
    if sigmoid_offset is None:
        sigmoid_offset = LearnedForwardProcess.HISTORICAL_OFFSET
    return LearnedForwardProcess(
        T=T, sigmoid_offset=sigmoid_offset, fixed_alphas=fixed_alphas
    )
