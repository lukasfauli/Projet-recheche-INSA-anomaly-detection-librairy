import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable


# ============================================================
# Kernel utilities
# ============================================================

def rbf_kernel(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    gamma: Optional[float] = None
) -> np.ndarray:
    """
    Gaussian / RBF kernel:
        K(x, y) = exp(-gamma * ||x - y||^2)
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    X_norm = np.sum(X ** 2, axis=1)[:, None]
    Y_norm = np.sum(Y ** 2, axis=1)[None, :]
    dist2 = X_norm + Y_norm - 2.0 * X @ Y.T
    dist2 = np.maximum(dist2, 0.0)
    return np.exp(-gamma * dist2)


def polynomial_kernel(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    degree: int = 2,
    coef0: float = 1.0
) -> np.ndarray:
    """
    Polynomial kernel:
        K(x, y) = (x^T y + coef0)^degree
    """
    if Y is None:
        Y = X
    return (X @ Y.T + coef0) ** degree


def stable_eigh_psd(M: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric eigendecomposition sorted descending.
    """
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    vals[vals < eps] = 0.0
    keep = vals > eps
    return vals[keep], vecs[:, keep]


def center_rows(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center a matrix whose columns are samples and rows are coordinates.
    """
    mu = np.mean(Z, axis=1, keepdims=True)
    return Z - mu, mu


# ============================================================
# Greedy KPCA subset selection
# ============================================================

def greedy_kpca_subset(
    X_block: np.ndarray,
    kernel_fn: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    max_subset: Optional[int] = None,
    mse_tol: Optional[float] = None,
    min_pivot: float = 1e-12,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy KPCA subset selection using pivoted kernel residual updates.

    Returns
    -------
    selected : array of selected indices in the block
    R : array of shape (s, n_block)
    d : final residual diagonal
    """
    K = kernel_fn(X_block, X_block)
    n = K.shape[0]

    if max_subset is None:
        max_subset = n

    diagK = np.clip(np.diag(K).copy(), 0.0, None)
    d = diagK.copy()
    R = np.zeros((max_subset, n), dtype=float)
    selected: List[int] = []

    for t in range(max_subset):
        j = int(np.argmax(d))
        pivot = d[j]

        if verbose:
            print(
                f"[greedy] step={t+1}, pivot_idx={j}, "
                f"pivot_residual={pivot:.6e}, mse={d.mean():.6e}"
            )

        if pivot <= min_pivot:
            break

        selected.append(j)

        if t == 0:
            R[t, :] = K[j, :] / np.sqrt(pivot)
        else:
            proj = R[:t, j] @ R[:t, :]
            R[t, :] = (K[j, :] - proj) / np.sqrt(pivot)

        d = np.maximum(d - R[t, :] ** 2, 0.0)

        if mse_tol is not None and d.mean() <= mse_tol:
            R = R[:t + 1, :]
            break

    if len(selected) == 0:
        raise RuntimeError("Greedy selection returned no pivots. Check kernel or tolerances.")

    if R.shape[0] != len(selected):
        R = R[:len(selected), :]

    return np.array(selected, dtype=int), R, d


# ============================================================
# Block-level GKPCA model
# ============================================================

@dataclass
class BlockModel:
    X_subset: np.ndarray
    subset_idx: np.ndarray
    beta: np.ndarray
    Z: np.ndarray
    mu: np.ndarray
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    residual_diag: np.ndarray


def block_gkpca(
    X_block: np.ndarray,
    kernel_fn: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    max_subset: Optional[int] = None,
    mse_tol: Optional[float] = None,
    min_pivot: float = 1e-12,
    var_keep: Optional[float] = None,
    max_kpcs: Optional[int] = None,
    verbose: bool = False
) -> BlockModel:
    """
    Run Greedy KPCA on one block.
    """
    subset_idx, _, residual_diag = greedy_kpca_subset(
        X_block,
        kernel_fn=kernel_fn,
        max_subset=max_subset,
        mse_tol=mse_tol,
        min_pivot=min_pivot,
        verbose=verbose
    )

    Xs = X_block[subset_idx]

    K_ss = kernel_fn(Xs, Xs)
    eigvals, eigvecs = stable_eigh_psd(K_ss)

    if len(eigvals) == 0:
        raise RuntimeError("Degenerate subset kernel matrix.")

    beta = eigvecs / np.sqrt(eigvals)[None, :]

    K_sb = kernel_fn(Xs, X_block)
    Z = beta.T @ K_sb

    Zc, mu = center_rows(Z)

    C = Zc @ Zc.T
    evals_c, U = stable_eigh_psd(C)
    S = np.sqrt(np.maximum(evals_c, 0.0))

    if len(S) == 0:
        raise RuntimeError("Degenerate block covariance in reduced coordinates.")

    Vt = (U.T @ Zc) / S[:, None]

    if max_kpcs is not None:
        q = min(max_kpcs, len(S))
    elif var_keep is not None:
        cum = np.cumsum(S ** 2)
        q = int(np.searchsorted(cum / cum[-1], var_keep) + 1)
    else:
        q = len(S)

    U = U[:, :q]
    S = S[:q]
    Vt = Vt[:q, :]

    return BlockModel(
        X_subset=Xs,
        subset_idx=subset_idx,
        beta=beta,
        Z=Z,
        mu=mu,
        U=U,
        S=S,
        Vt=Vt,
        residual_diag=residual_diag
    )


# ============================================================
# Global mult-GKPCA model
# ============================================================

@dataclass
class MultiGKPCAModel:
    kernel_name: str
    kernel_params: dict
    X_library: np.ndarray
    alpha: np.ndarray
    singular_values: np.ndarray
    mean_coeff: np.ndarray

    def kernel_fn(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if self.kernel_name == "rbf":
            return rbf_kernel(X, Y, **self.kernel_params)
        if self.kernel_name == "poly":
            return polynomial_kernel(X, Y, **self.kernel_params)
        raise NotImplementedError(f"Unsupported kernel '{self.kernel_name}'.")


# ============================================================
# Merge logic
# ============================================================

def _expand_to_union(
    old_n: int,
    new_n: int,
    A_old: Optional[np.ndarray],
    A_new: Optional[np.ndarray]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Expand coefficients into union library coordinates.
    """
    A_old_u = None
    A_new_u = None

    if A_old is not None:
        A_old_u = np.vstack([A_old, np.zeros((new_n, A_old.shape[1]))])

    if A_new is not None:
        A_new_u = np.vstack([np.zeros((old_n, A_new.shape[1])), A_new])

    return A_old_u, A_new_u


def merge_mult_gkpca_models(
    model: Optional[MultiGKPCAModel],
    block: BlockModel,
    kernel_name: str = "rbf",
    kernel_params: Optional[dict] = None,
    var_keep: Optional[float] = 0.9999,
    max_kpcs: Optional[int] = None
) -> MultiGKPCAModel:
    """
    Merge one block-GKPCA model into the current global model.
    """
    if kernel_params is None:
        kernel_params = {}

    alpha_b = block.beta @ block.U
    mean_b = block.beta @ block.mu
    S_b = block.S
    Xb = block.X_subset

    if model is None:
        return MultiGKPCAModel(
            kernel_name=kernel_name,
            kernel_params=kernel_params,
            X_library=Xb.copy(),
            alpha=alpha_b.copy(),
            singular_values=S_b.copy(),
            mean_coeff=mean_b.copy()
        )

    Xa = model.X_library
    na = Xa.shape[0]
    nb = Xb.shape[0]

    X_union = np.vstack([Xa, Xb])

    alpha_a_u, alpha_b_u = _expand_to_union(na, nb, model.alpha, alpha_b)
    mean_a_u, mean_b_u = _expand_to_union(na, nb, model.mean_coeff, mean_b)

    if kernel_name == "rbf":
        K_union = rbf_kernel(X_union, X_union, **kernel_params)
    elif kernel_name == "poly":
        K_union = polynomial_kernel(X_union, X_union, **kernel_params)
    else:
        raise NotImplementedError(f"Unsupported kernel '{kernel_name}'.")

    dm = mean_b_u - mean_a_u
    Basis = np.hstack([alpha_a_u, alpha_b_u, dm])

    qa = alpha_a_u.shape[1]
    qb = alpha_b_u.shape[1]

    Wa = np.diag(model.singular_values ** 2)
    Wb = np.diag(S_b ** 2)

    W = np.block([
        [Wa, np.zeros((qa, qb)), np.zeros((qa, 1))],
        [np.zeros((qb, qa)), Wb, np.zeros((qb, 1))],
        [np.zeros((1, qa)), np.zeros((1, qb)), np.ones((1, 1))]
    ])

    M = Basis.T @ K_union @ Basis
    Csmall = M @ W @ M.T

    evals, Usmall = stable_eigh_psd(Csmall)
    if len(evals) == 0:
        return model

    S_new = np.sqrt(evals)
    alpha_new = Basis @ Usmall

    if max_kpcs is not None:
        q = min(max_kpcs, len(S_new))
    elif var_keep is not None:
        cum = np.cumsum(S_new ** 2)
        q = int(np.searchsorted(cum / cum[-1], var_keep) + 1)
    else:
        q = len(S_new)

    alpha_new = alpha_new[:, :q]
    S_new = S_new[:q]

    mean_new = 0.5 * (mean_a_u + mean_b_u)

    return MultiGKPCAModel(
        kernel_name=kernel_name,
        kernel_params=kernel_params,
        X_library=X_union,
        alpha=alpha_new,
        singular_values=S_new,
        mean_coeff=mean_new
    )


# ============================================================
# Main mult-GKPCA class
# ============================================================

class MultiGKPCA:
    """
    mult-GKPCA:
      1) split data into blocks
      2) run GKPCA on each block
      3) merge blockwise KPCA incrementally
      4) optionally deflate after each merge
    """

    def __init__(
        self,
        n_blocks: int = 10,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        degree: int = 2,
        coef0: float = 1.0,
        max_subset_per_block: Optional[int] = None,
        mse_tol: Optional[float] = None,
        min_pivot: float = 1e-12,
        block_var_keep: Optional[float] = None,
        global_var_keep: Optional[float] = 0.9999,
        max_block_kpcs: Optional[int] = None,
        max_global_kpcs: Optional[int] = None,
        shuffle: bool = True,
        random_state: int = 42,
        verbose: bool = False
    ):
        self.n_blocks = n_blocks
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_subset_per_block = max_subset_per_block
        self.mse_tol = mse_tol
        self.min_pivot = min_pivot
        self.block_var_keep = block_var_keep
        self.global_var_keep = global_var_keep
        self.max_block_kpcs = max_block_kpcs
        self.max_global_kpcs = max_global_kpcs
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

        self.model_: Optional[MultiGKPCAModel] = None
        self.block_models_: List[BlockModel] = []
        self.block_indices_: List[np.ndarray] = []

    def _kernel_fn(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if self.kernel == "rbf":
            return rbf_kernel(X, Y, gamma=self.gamma)
        if self.kernel == "poly":
            return polynomial_kernel(X, Y, degree=self.degree, coef0=self.coef0)
        raise NotImplementedError(f"Unsupported kernel '{self.kernel}'.")

    def _kernel_params(self) -> dict:
        if self.kernel == "rbf":
            return {"gamma": self.gamma}
        if self.kernel == "poly":
            return {"degree": self.degree, "coef0": self.coef0}
        raise NotImplementedError

    def fit(self, X: np.ndarray):
        """
        Fit mult-GKPCA on training data X.
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n)
        if self.shuffle:
            rng.shuffle(indices)

        blocks = np.array_split(indices, self.n_blocks)

        self.block_models_ = []
        self.block_indices_ = []
        self.model_ = None

        for b, idx in enumerate(blocks):
            Xb = X[idx]

            block_model = block_gkpca(
                Xb,
                kernel_fn=self._kernel_fn,
                max_subset=self.max_subset_per_block,
                mse_tol=self.mse_tol,
                min_pivot=self.min_pivot,
                var_keep=self.block_var_keep,
                max_kpcs=self.max_block_kpcs,
                verbose=self.verbose
            )

            self.block_models_.append(block_model)
            self.block_indices_.append(idx)

            self.model_ = merge_mult_gkpca_models(
                self.model_,
                block_model,
                kernel_name=self.kernel,
                kernel_params=self._kernel_params(),
                var_keep=self.global_var_keep,
                max_kpcs=self.max_global_kpcs
            )

            if self.verbose:
                print(
                    f"[block {b+1}/{len(blocks)}] "
                    f"size={len(idx)} | subset={block_model.X_subset.shape[0]} | "
                    f"block_kpcs={len(block_model.S)} | "
                    f"global_lib={self.model_.X_library.shape[0]} | "
                    f"global_kpcs={self.model_.alpha.shape[1]}"
                )

        return self

    def transform(self, X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """
        Project new data into the learned mult-GKPCA score space.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit first.")

        X = np.asarray(X, dtype=float)
        q = self.model_.alpha.shape[1] if n_components is None else min(
            n_components,
            self.model_.alpha.shape[1]
        )

        K = self.model_.kernel_fn(self.model_.X_library, X)

        mean_shift = self.model_.alpha[:, :q].T @ (
            self.model_.kernel_fn(self.model_.X_library, self.model_.X_library)
            @ self.model_.mean_coeff
        )

        scores = (self.model_.alpha[:, :q].T @ K - mean_shift).T
        return scores

    def transform_batches(self, X: np.ndarray, batch_size: int = 10000, n_components: Optional[int] = None):
        """
        Transform X in batches.
        """
        X = np.asarray(X, dtype=float)
        Z_parts = []
        n = X.shape[0]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            print(f"Transforming batch {start}:{end} / {n}")
            Z_batch = self.transform(X[start:end], n_components=n_components)
            Z_parts.append(Z_batch)

        return np.vstack(Z_parts)

    def fit_transform(self, X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        self.fit(X)
        return self.transform(X, n_components=n_components)

    @property
    def library_size_(self) -> int:
        if self.model_ is None:
            raise RuntimeError("Call fit first.")
        return self.model_.X_library.shape[0]

    @property
    def n_kpcs_(self) -> int:
        if self.model_ is None:
            raise RuntimeError("Call fit first.")
        return self.model_.alpha.shape[1]


# ============================================================
# Convenience anomaly-detection helpers
# ============================================================

