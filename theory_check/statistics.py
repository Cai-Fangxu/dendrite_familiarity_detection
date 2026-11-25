"""Streaming conditional moments and independent-batch uncertainty estimates."""

import numpy as np

from theory_check.spherical import amplitudes, target_terms


class Moments:
    """Merge centered moments for samples shaped (fork, pattern, dendrite)."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def add(self, values):
        """Accumulate a batch in float64 using the parallel covariance identity."""
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite fork increments; inspect learning amplitudes")
        mean = values.mean(axis=0)
        centered = values - mean
        m2 = np.einsum("kpr,kps->prs", centered, centered)
        self.merge(len(values), mean, m2)

    def merge(self, count, mean, m2):
        """Merge sufficient statistics without treating batch means as raw forks."""
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean.copy(), m2.copy()
            return
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + np.einsum("pr,ps->prs", delta, delta) * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    @property
    def covariance(self):
        """Unbiased centered covariance with K - 1 degrees of freedom."""
        if self.count < 2:
            raise ValueError("Covariance requires at least two forks")
        return self.m2 / (self.count - 1)


def balanced_sizes(total, groups):
    """Split a fixed sample count into near-equal independent groups."""
    return np.full(groups, total // groups, dtype=int) + (np.arange(groups) < total % groups)


def batch_se(values, counts):
    """Approximate SE from independent, near-equal groups, retaining dependence.

    Aggregate ages or dendrites within each group *before* calling this function
    to account for their shared fork/theory inputs. This is conditional on W.
    """
    values = np.asarray(values)
    weights = np.asarray(counts, dtype=float) / np.sum(counts)
    shape = (-1,) + (1,) * (values.ndim - 1)
    mean = np.sum(weights.reshape(shape) * values, axis=0)
    variance = np.sum(weights.reshape(shape)**2 * (values - mean)**2, axis=0)
    return np.sqrt(variance / (1 - np.sum(weights**2)))


def gram_factor(q):
    """Factor the realized PSD Gram matrix and use the same cutoff for Q+.

    No ridge or identity substitution is used, including when nd >= ns.
    """
    eigenvalues, vectors = np.linalg.eigh((q + q.T) / 2)
    cutoff = max(q.shape) * np.finfo(float).eps * max(eigenvalues[-1], 1)
    if eigenvalues[0] < -100 * cutoff:
        raise ValueError("Gram matrix is not positive semidefinite")
    keep = eigenvalues > cutoff
    basis = vectors[:, keep]
    return basis * np.sqrt(eigenvalues[keep]), (basis / eigenvalues[keep]) @ basis.T, int(keep.sum())


def drift_matrix(rule, condition, c, f, selection_probability, inverse):
    """Local leading drift for spherical rotations or selected weight decay."""
    if rule.is_decay:
        return f @ inverse - np.diag(rule.beta_decay * selection_probability)
    if condition == "independent":
        return -0.5 * c
    return f @ inverse - np.diag(np.diag(f) + 0.5 * np.diag(c))


def theory_moments(rule, condition, q, m, rng, samples, groups, chunk_size):
    """Estimate local coefficients with a separate Gaussian MC stream.

    Batch predictions carry numerical uncertainty through H and each full m.
    For spherical independent inputs exchangeability pools a_r^2 across dendrites
    within each Gaussian vector, without counting dendrites as independent draws.
    Decay uses the actual activation covariance, a = mask*(target-u), and
    H = E[a u.T] Q+ - beta_decay*diag(E[mask]); no spherical radial term is used.
    This is a large-ns reference: the input-length denominator and the finite-ns
    beta*u/ns compensation in the exact update are not replaced in simulation.
    With disjoint inputs, F stores only its local diagonal because cross-branch
    activation/plasticity moments do not contribute to the old-pattern drift.
    """
    factor, inverse, rank = gram_factor(q)
    counts = balanced_sizes(samples, groups)
    c_total = np.zeros_like(q)
    f_total = np.zeros_like(q)
    p_total = np.zeros(rule.nd)
    batch_c, batch_f, batch_h, batch_a = [], [], [], []
    batch_p = []
    for count in counts:
        c = np.zeros_like(q)
        f = np.zeros_like(q)
        p = np.zeros(rule.nd)
        for offset in range(0, count, chunk_size):
            size = min(chunk_size, count - offset)
            z = rng.normal(size=(size, rule.nd if condition == "independent" else rank))
            u = z if condition == "independent" else z @ factor.T
            if rule.is_decay:
                if condition == "independent":
                    u = z * np.sqrt(np.diag(q))
                mask, target = target_terms(u, rule)
                a = mask * (target - u)
                p += mask.sum(axis=0)
            else:
                a = amplitudes(u, rule)
            if not np.isfinite(a).all():
                raise FloatingPointError("Nonfinite theory amplitudes")
            if condition == "independent":
                if rule.is_decay:
                    c += np.diag(np.sum(a * a, axis=0))
                    f += np.diag(np.sum(a * u, axis=0))
                else:
                    c += np.eye(rule.nd) * np.sum(a * a) / rule.nd
            else:
                c += a.T @ a
                f += a.T @ u
        c /= count
        f /= count
        p /= count
        h = drift_matrix(rule, condition, c, f, p, inverse)
        if not np.isfinite(c).all() or not np.isfinite(h).all():
            raise FloatingPointError("Nonfinite theory moments; inspect learning amplitudes")
        batch_c.append(c)
        batch_f.append(f)
        batch_h.append(h)
        batch_p.append(p)
        batch_a.append(m @ h.T)
        c_total += c * count / samples
        f_total += f * count / samples
        p_total += p * count / samples
    h = drift_matrix(rule, condition, c_total, f_total, p_total, inverse)
    result = dict(C=c_total, F=f_total, H=h, A=m @ h.T, counts=counts,
                  C_se=batch_se(batch_c, counts), F_se=batch_se(batch_f, counts),
                  H_se=batch_se(batch_h, counts), batch_C=np.asarray(batch_c),
                  batch_A=np.asarray(batch_a), rank=rank)
    if rule.is_decay:
        result.update(selection_probability=p_total, selection_probability_se=batch_se(batch_p, counts))
    return result


def choose_pairs(c, maximum):
    """Prespecified theory-only rule: evenly spaced ranks of predicted covariance.

    Include both extrema and spread the remaining pairs across the ordering.
    Selection never uses measured fork agreement. Signed variable-target
    amplitudes can yield negative entries; nonnegative amplitudes cannot.
    """
    rows, cols = np.triu_indices(len(c), 1)
    if not len(rows) or maximum == 0:
        return np.empty((0, 2), dtype=int)
    order = np.argsort(c[rows, cols], kind="stable")
    indices = order[np.linspace(0, len(order) - 1, min(maximum, len(order)), dtype=int)]
    return np.column_stack((rows[indices], cols[indices]))
