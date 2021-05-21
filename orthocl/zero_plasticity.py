import torch


class ZeroPlasticity:
    def __init__(self) -> None:
        self._proj = None

    def register_input(self, x: torch.Tensor) -> None:
        if self._proj is None:
            self._proj = torch.zeros(x.size(1), x.size(1), device=x.device)

    def compute_projection(self) -> torch.Tensor:
        return self._proj
