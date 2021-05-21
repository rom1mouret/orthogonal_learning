import torch.nn as nn
import torch

def safe_svd(x: torch.Tensor, compute_uv: bool=True):
    if torch.isnan(x).any():
        raise Exception("SVD won't work on matrices with NaN values")
    # note: in future version of Pytorch, compute_uv=False might return a single element
    for n in range(8):
        try:
            return torch.svd(x, compute_uv=compute_uv)
        except:
            if n == 0:
                logging.exception("SVD failed")
            x = x + 0.0001 * x.mean() * torch.rand_like(x)
    raise Exception("maximum SVD attempt")


class LowPlasticity:
    def __init__(self, unit_vectors: bool=True) -> None:
        self._n_samples = 0
        self._cov = 0
        self._unit_vectors = unit_vectors

    def register_input(self, x: torch.Tensor) -> None:
        self._cov = x.t() @ x + self._cov
        self._n_samples += x.size(0)

    def compute_projection(self) -> torch.Tensor:
        # approximation of the null space
        actual_cov = self._cov / self._n_samples
        singular_vecs, singular_vals, _ = safe_svd(actual_cov.double())

        i = len(singular_vals)
        while singular_vals[i-1] == 0:
            i -= 1
        singular_vecs = singular_vecs[:i]

        if self._unit_vectors:
            singular_vecs /= singular_vecs.norm(dim=0, keepdim=True)

        id = torch.eye(singular_vecs.size(0), device=singular_vecs.device)
        proj = id - (singular_vecs @ singular_vecs.t()).float()

        # check idempotence
        diff = (proj - proj @ proj).abs().sum().item()
        print("idempotence diff: %.4f" % diff)

        return proj
