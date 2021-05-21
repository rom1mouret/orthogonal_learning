from bisect import bisect_left
from typing import Callable
import torch.nn as nn
import torch
import logging

# Training Networks in Null Space of Feature Covariance for Continual Learning
# Ref: https://arxiv.org/pdf/2103.07113.pdf

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


class NullSpace:
    def __init__(self, R: float=0.05, a: float=-1, unit_vectors: bool=True) -> None:
        self._n_samples = 0
        self._cov = 0
        self._R = R
        self._a = a
        self._unit_vectors = unit_vectors

    def register_input(self, x: torch.Tensor) -> None:
        self._cov = x.t() @ x + self._cov
        self._n_samples += x.size(0)

    def compute_projection(self) -> torch.Tensor:
        # approximation of the null space
        actual_cov = self._cov / self._n_samples
        singular_vecs, singular_vals, _ = safe_svd(actual_cov.double())

        if self._a > 1:
            # as in the paper
            cut = self._a * singular_vals[-1]
            position = len(singular_vals) - bisect_left(list(reversed(singular_vals)), cut)
        else:
            # auto-mode
            # (singular vecs should account for less than 0.05 of the total values)
            Rs = singular_vals.cumsum(dim=0) / singular_vals.sum()
            position = min(1 + bisect_left(Rs.tolist(), 1 - self._R), len(Rs)-1)

        print("position:", position, "/", singular_vals.size(0))
        approx = singular_vecs[:, position:]
        if self._unit_vectors:
            # not always unitary bc of svd's lack of precision
            approx /= approx.norm(dim=0, keepdim=True)
        #print("u2 norm", approx.norm(dim=0))
        proj = (approx @ approx.t()).float()
        #print("**** null space position:", position, "/", len(singular_vals))

        # check idempotence
        diff = (proj - proj @ proj).abs().sum().item()
        print("idempotence diff: %.4f" % diff)

        v = torch.randn(256, self._cov.size(0), device=proj.device)
        p1 = v @ proj
        p2 = v @ (proj @ proj)
        diff = (p1 - p2).abs().sum().item()
        print("random proj idempotence diff: %.4f" % diff)

        return proj
