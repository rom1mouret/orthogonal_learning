from bisect import bisect_left
from typing import Callable
import numpy as np
import torch.nn as nn
import torch
from contextlib import contextmanager

# Ref: GRADIENT PROJECTION MEMORY LEARNING FOR CONTINUAL (ICRL 2021)
# https://openreview.net/pdf?id=3AOj0RCNC2

# handy: https://web.stanford.edu/class/cs168/l/l9.pdf


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


class GPM:
    def __init__(self, max_samples: int=4096*8, R: float=0.05, unit_vectors: bool=True) -> None:
        self._max_samples = max_samples
        self._n_samples = 0
        self._vecs = None
        self._proj = None
        self._R = R
        self._unit_vectors = unit_vectors
        self._reset()

    def _reset(self):
        self._sampling_rate = 1.0  # this will be adjusted on-the-fly
        self._n_samples = 0
        self._X = None

    def register_input(self, x: torch.Tensor) -> None:
        """ We cannot keep all the inputs so we have to random sample it.
        Another problem: we cannot assume the input to be shuffled when it comes from
        image patches from the same image. """
        self._device = x.device
        self._n_samples += x.size(0)
        if self._X is None:
            if x.size(0) > self._max_samples:
                self._sampling_rate = self._max_samples / x.size(0)
                mask = np.random.binomial(1, self._sampling_rate, x.size(0)).astype(np.bool)
                x = x[mask, :]
            self._X = x.to("cpu")
        elif self._X.size(0) <= self._max_samples:
            # this might grow a bit overboard but no more than a x2 factor
            self._X = torch.cat([self._X, x.to("cpu")], dim=0)
        else:
            new_rate = self._max_samples / self._n_samples
            # re-sample the rows already stored to match the new sampling rate
            rate_ratio = new_rate / self._sampling_rate
            mask = np.random.binomial(1, rate_ratio, self._X.size(0)).astype(np.bool)
            self._X = self._X[mask, :]
            # sample the new rows with the new sampling rate
            mask = np.random.binomial(1, new_rate, x.size(0)).astype(np.bool)
            self._sampling_rate = new_rate
            if mask.any():
                # put everything together
                x = x[mask, :]
                self._X = torch.cat([self._X, x.to("cpu")], dim=0)

    def compute_projection(self) -> torch.Tensor:
        features_as_rows = self._X.t()
        # compute SVD on raw self._X only to get the sum of the singular values
        _, singular_vals, _ = safe_svd(features_as_rows, compute_uv=False)
        total_singular_vals = singular_vals.pow(2).sum()

        if self._proj is not None:
            features_as_rows -= self._proj @ features_as_rows

        # compute SVD on CPU
        left_singular, singular_vals, _ = safe_svd(features_as_rows)
        singular_val_square = singular_vals.pow(2)
        diff = total_singular_vals - singular_val_square.sum()
        Rs = (diff + singular_val_square.cumsum(dim=0)) / total_singular_vals
        position = bisect_left(Rs.tolist(), 1-self._R)
        new_vecs = left_singular[:, :position]

        # transform to unit-length vecs because of SVD's lack of precision
        if self._unit_vectors:
            new_vecs /= new_vecs.norm(dim=0, keepdim=True)

        if self._vecs is None:
            self._vecs = new_vecs
        else:
            self._vecs = torch.cat([self._vecs, new_vecs], dim=1)
        print("newvec size", new_vecs.size(), "total vec size", self._vecs.size())
        self._proj = self._vecs @ self._vecs.t()  # this stays on CPU

        diff = (self._proj - self._proj @ self._proj).abs().sum().item()
        print("idempotence:", diff)

        self._reset()

        # in the paper, they do "grad - P x grad", but in the null space
        # paper, it's "P x grad".
        # To make both papers compatible and thanks to the distributivity of
        # matrix multiplication, we change "grad - P x grad" to "grad x (1 - P)"
        ret = self._proj.to(self._device)
        return torch.eye(ret.size(0), device=ret.device) - ret
