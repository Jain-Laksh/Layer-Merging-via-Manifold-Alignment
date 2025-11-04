# aryan/scripts/utils_merging.py
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


def diffusion_kernel_embed(X: np.ndarray, sigma=8.0, alpha=0.5, d=2):
    """
    X: (N, D) activations for a layer
    returns: (N, d) low-dim embedding
    closely follows the paper idea
    """
    # affinity
    W = pairwise_kernels(X, metric="rbf", gamma=1.0 / (2 * sigma * sigma))
    # row-normalize
    D = np.sum(W, axis=1, keepdims=True)
    P = W / (D + 1e-8)

    # eigen
    eigvals, eigvecs = np.linalg.eig(P)
    # sort by eigenvalue desc
    idx = np.argsort(-eigvals.real)
    eigvals = eigvals.real[idx]
    eigvecs = eigvecs.real[:, idx]
    # skip the first trivial eigenvector
    emb = []
    for k in range(1, d + 1):
        emb.append((eigvals[k] ** alpha) * eigvecs[:, k])
    emb = np.stack(emb, axis=1)  # (N, d)
    return emb


def normalized_mi_gaussian(A: np.ndarray, B: np.ndarray):
    """
    A: (N, d), B: (N, d)
    NMI (paper-style, approximate)
    """
    # covariances
    cov_A = np.cov(A, rowvar=False)
    cov_B = np.cov(B, rowvar=False)
    AB = np.concatenate([A, B], axis=1)
    cov_AB = np.cov(AB, rowvar=False)

    def logdet(mat):
        return np.log(np.linalg.det(mat + 1e-6 * np.eye(mat.shape[0])))

    I = 0.5 * (logdet(cov_A) + logdet(cov_B) - logdet(cov_AB))
    H_A = 0.5 * logdet(cov_A)
    H_B = 0.5 * logdet(cov_B)
    nmi = I / (np.sqrt(H_A * H_B) + 1e-8)
    return nmi


def compute_similarity_matrix(layer_embeddings: dict):
    """
    layer_embeddings: {layer_idx: np.array(N, d)}
    returns: (L, L) similarity matrix
    """
    layers = sorted(layer_embeddings.keys())
    L = len(layers)
    sim = np.zeros((L, L))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            if j < i:
                sim[i, j] = sim[j, i]
                continue
            if i == j:
                sim[i, j] = 1.0
            else:
                nmi = normalized_mi_gaussian(layer_embeddings[li], layer_embeddings[lj])
                sim[i, j] = nmi
    return sim
