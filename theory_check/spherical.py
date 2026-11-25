"""Signed angular learning and reproducible Gaussian training/fork streams.

Spherical rules use theta = amplitude / sqrt(ns). The explicit *_decay variants
reuse the legacy library updates without normalization. Full sharing uses one
common synapse ordering, so the physical Gram matrix is simply W W.T.
"""

from dataclasses import dataclass
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtri


RULE_NAMES = ("threshold", "nwta", "variable_target", "softmax_int", "softmax_nonint",
              "threshold_decay", "nwta_decay", "variable_target_decay")
SUPPORTED_ND = (1, 2, 3, 4, 5, 10, 12, 15, 20, 30, 50, 60, 75, 100, 150,
                300, 400, 750, 1000, 1500)


@dataclass(frozen=True)
class Rule:
    """Fixed learning parameters; unused parameters remain None."""

    name: str
    nd: int
    ns: int
    n: float = 10
    b: float | None = None
    kappa: float | None = None
    eta: float | None = None
    beta_select: float | None = None
    beta_decay: float | None = None

    @property
    def is_decay(self):
        """Whether this rule uses the original weight-decay implementation."""
        return self.name.endswith("_decay")

    @property
    def base_name(self):
        """Selection/target family shared by spherical and decay variants."""
        return self.name.removesuffix("_decay")

    @property
    def eligibility(self):
        """Gaussian eligibility quantile; n == nd selects every branch."""
        return float(ndtri(1 - self.n / self.nd))


def target_terms(u, rule, xp=np):
    """Return the selected mask and target, retaining zero-amplitude selections.

    Decay acts on selected branches even if a variable target equals activation.
    Its mask therefore cannot be reconstructed from nonzero amplitudes.
    """
    name = rule.base_name
    if name == "nwta":
        if xp is jnp:
            # Avoid inverse argsort: GPU lowering can mix int32 scatter updates
            # with int64 buffers when x64 is enabled. top_k uses int32 indices.
            # Reversal retains stable ascending-sort selection at ties: the
            # larger original index wins. A cutoff mask avoids scatter entirely.
            values, indices = jax.lax.top_k(u if rule.is_decay else u[..., ::-1], int(rule.n))
            cutoff = values[..., -1:]
            positions = jnp.arange(rule.nd, dtype=jnp.int32)
            # The library's top_k gives lower-index branches priority at ties.
            tied = positions <= indices[..., -1:] if rule.is_decay else positions >= jnp.int32(rule.nd - 1) - indices[..., -1:]
            mask = (u > cutoff) | ((u == cutoff) & tied)
        else:
            order = np.argsort(-u if rule.is_decay else u, axis=-1, kind="stable")
            ranks = np.argsort(order, axis=-1)
            mask = ranks < int(rule.n) if rule.is_decay else ranks >= rule.nd - int(rule.n)
        target = rule.b + rule.kappa
        return mask & (u < target), target
    mask = u > rule.eligibility
    if name == "threshold":
        target = rule.b + rule.kappa
        return mask & (u < target), target
    if name != "variable_target":
        raise ValueError(f"Unknown rule: {rule.name}")
    count = xp.sum(mask, axis=-1, keepdims=True)
    fallback = xp.arange(rule.nd) == xp.argmax(u, axis=-1, keepdims=True)
    mask = xp.where(count == 0, fallback, mask)
    target = rule.b + rule.n * rule.kappa / xp.maximum(count, 1)
    return mask, target


def amplitudes(u, rule, xp=np):
    """Angular amplitudes or leading input coefficients for legacy decay rules."""
    if rule.name == "softmax_int":
        logits = rule.beta_select * (u - xp.max(u, axis=-1, keepdims=True))
        p = xp.exp(logits)
        return rule.eta * p / xp.sum(p, axis=-1, keepdims=True)
    if rule.name == "softmax_nonint":
        return rule.eta / rule.nd * xp.exp(rule.beta_select * u - rule.beta_select**2 / 2)
    mask, target = target_terms(u, rule, xp)
    # Unlike the fixed-target rules, selected variable-target branches may move downward.
    return mask * (target - u)


@lru_cache(maxsize=None)
def legacy_neuron(rule):
    """Construct the existing library model once for each fixed parameter set."""
    from lib.neurons import (HyperConfig1, Neuron_Decay_NonInt, Neuron_Decay_Int,
                             Neuron_Decay_Int_Variable_n_min1)

    classes = {"threshold_decay": Neuron_Decay_NonInt, "nwta_decay": Neuron_Decay_Int,
               "variable_target_decay": Neuron_Decay_Int_Variable_n_min1}
    n = int(rule.n) if rule.base_name == "nwta" else rule.n
    return classes[rule.name](HyperConfig1(rule.nd, rule.ns, n, rule.b,
                                         rule.b + rule.kappa, rule.beta_decay))


def update_weights(w, x, rule):
    """Dispatch to the exact legacy update or a unit-norm spherical rotation."""
    if not rule.is_decay:
        return rotate(w, x, rule)
    from lib.neurons import HyperDynConfig1, State1

    dynamic = HyperDynConfig1(rule.b, rule.b + rule.kappa, rule.beta_decay)
    # Legacy methods index inputs by dendrite, even for fully shared inputs.
    x = jnp.broadcast_to(x, w.shape)
    return legacy_neuron(rule).update(dynamic, State1(w=w), x).w


def initialize_weights(key, rule, dtype):
    """Match each implementation's initialization with paired Gaussian draws."""
    w = jax.random.normal(key, (rule.nd, rule.ns), dtype=dtype)
    return w / np.sqrt(rule.ns) if rule.is_decay else w / jnp.linalg.norm(w, axis=-1, keepdims=True)


def rotate(w, x, rule):
    """Apply a leading-angle spherical update, including degenerate tangents."""
    u = jnp.sum(w * x, axis=-1)
    a = amplitudes(u, rule, jnp)
    tangent = x - u[..., None] * w
    length = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
    tolerance = 8 * jnp.finfo(w.dtype).eps * jnp.linalg.norm(x, axis=-1, keepdims=True)
    valid = length > tolerance
    direction = tangent / jnp.where(valid, length, 1)
    theta = a[..., None] / np.sqrt(rule.ns)
    updated = w * jnp.cos(theta) + direction * jnp.sin(theta)
    updated /= jnp.linalg.norm(updated, axis=-1, keepdims=True)
    return jnp.where(valid, updated, w)


def pattern(key, rule, condition, dtype):
    """Draw independent local inputs or one fully shared axonal input."""
    shape = (rule.nd, rule.ns) if condition == "independent" else (rule.ns,)
    return jax.random.normal(key, shape, dtype=dtype)


def indexed_key(root, index):
    """Derive streams by identity so chunk sizes do not change the experiment."""
    return jax.random.fold_in(root, jnp.asarray(index, dtype=jnp.uint32))


@partial(jax.jit, static_argnames=("rule", "condition", "steps"))
def train_segment(w, root, start, *, rule, condition, steps):
    """Train through a segment; acquisition times are one-based update counts."""
    def step(weights, offset):
        x = pattern(indexed_key(root, start + offset), rule, condition, weights.dtype)
        return update_weights(weights, x, rule), None

    return jax.lax.scan(step, w, jnp.arange(steps, dtype=jnp.uint32))[0]


@partial(jax.jit, static_argnames=("rule", "condition", "steps"))
def fork_batch(w, old, root, indices, *, rule, condition, steps):
    """Return endpoint overlap increments for independently restarted forks.

    Every fork starts at w. Within a fork the weights evolve sequentially, and
    the same trajectory evaluates every stored pattern. No trajectory is saved.
    """
    def one(index):
        key = indexed_key(root, index)
        final = train_segment(w, key, 0, rule=rule, condition=condition, steps=steps)
        difference = final - w
        if condition == "shared":
            return jnp.einsum("rs,ps->pr", difference, old)
        return jnp.einsum("rs,prs->pr", difference, old)

    return jax.vmap(one)(indices)


def overlaps(w, old, condition):
    """Compute pattern-by-dendrite overlaps in the actual coordinate system."""
    if condition == "shared":
        return np.einsum("rs,ps->pr", w, old)
    return np.einsum("rs,prs->pr", w, old)
