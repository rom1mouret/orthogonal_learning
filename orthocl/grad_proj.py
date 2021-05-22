from typing import Callable
import torch.nn as nn
import torch
from contextlib import contextmanager

class GradProjLayer(nn.Module):
    """ single-parameter layer """
    def __init__(self, proj_calculator) -> None:
        super(GradProjLayer, self).__init__()
        self._collecting_on = False
        self._proj_calculator = proj_calculator
        self._proj = None
        self._enable = True
        self._t = 0
        self._double_proj = False
        self._post_proj = True
        self._snapshot = None
        self.reset()

    def two_proj(self, enable: bool=True) -> "GradProjLayer":
        """ this will project the matrix onto the gradients twice.
            Could be relevant for Adam and RMSProp """
        self._double_proj = enable
        return self

    def post_proj(self, enable: bool=True) -> "GradProjLayer":
        self._post_proj = enable
        return self

    def data_collecting_on(self) -> None:
        self._collecting_on = True

    def data_collecting_off(self) -> None:
        self._collecting_on = False
        self._proj = self._proj_calculator.compute_projection()

    def random_step(self, amplitude: float) -> None:
        """ random step in the null space -> can be used to generate more
            training data """
        params = self.parameters()
        for param in params:
            break

        # random noise
        delta = amplitude * torch.randn_like(param.data)
        # projects onto the null space
        delta = self.project_grad(delta)
        # update param
        param.data += delta

    def disable(self):
        self._enable = False
        return self

    def reset(self) -> None:
        self._1st_moment = 0
        self._2nd_moment = 0
        self._prev_step = 0

    def take_snapshot(self, reset: bool=True) -> None:
        if reset:
            self.reset()
        self._displacement = 0
        for param in self.parameters():
            self._snapshot = param.data.clone()
            break  # only one param

    def l1dist(self) -> torch.Tensor:
        if self._snapshot is None or self._proj is None:
            # first task
            for param in self.parameters():
                return torch.zeros(1, device=param.device)
        for param in self.parameters():
            return (param - self._snapshot).abs().sum()

    def adam_step(self, lr: float, **args) -> None:
        self._step(self._adam, lr, **args)

    def rms_step(self, lr: float, **args) -> None:
        self._step(self._rmsprop, lr, **args)

    def sgd_step(self, lr: float, momentum=0, **args) -> None:
        self._step(self._sgd, lr, momentum, **args)

    def _step(self, optimizer: Callable, lr: float, momentum: float, **args) -> None:
        self._t += 1
        params = self.parameters()
        do_projections = self._proj is not None and self._enable
        for param in params:
            grad = param.grad
            if grad is None:
                raise Exception("step() cannot be called before backward()")
            if do_projections and self._double_proj:
                # this is okay only if the projection matrix is idempotent!
                grad = self.project_grad(grad)
            # adjust the gradients with adam, sgd or another optimizer
            adjusted_grad = optimizer(grad, **args)
            if self._post_proj:
                self._displacement = adjusted_grad + self._displacement
                if not do_projections:
                    # first task: no nullspace/GPM yet
                    safe_displacement = self._displacement
                else:
                    # project the gradients onto the nullspace
                    safe_displacement = self.project_grad(self._displacement)

                param.data.copy_(self._snapshot - lr * safe_displacement)
            else:
                # classic gradient descent
                if not do_projections:
                    # first task: no nullspace/GPM yet
                    safe_grad = adjusted_grad
                else:
                    # project the gradients onto the null space
                    safe_grad = self.project_grad(adjusted_grad)
                step = -lr * safe_grad + momentum * self._prev_step
                self._prev_step = step
                param.data += step
            grad.data.zero_()
            break

        for param in params:
            raise Exception("this class doesn't work with more than one parameter")

    def _sgd(self, grad: torch.Tensor) -> None:
        return grad

    def _rmsprop(self, grad: torch.Tensor, beta=0.99, eps=1e-08):
        if self._t == 1:
            self._2nd_moment = grad.pow(2)
        else:
            self._2nd_moment = beta * self._2nd_moment + (1-beta) * grad.pow(2)

        return grad / (self._2nd_moment.sqrt() + eps)

    def _adam(self, grad: torch.Tensor, beta1=0.9, beta2=0.999, eps=1e-08):
        # REF: https://arxiv.org/pdf/1412.6980v8.pdf
        # biased first and second moment estimates
        self._1st_moment = beta1 * self._1st_moment + (1-beta1) * grad
        self._2nd_moment = beta2 * self._2nd_moment + (1-beta2) * grad.pow(2)

        # biased-corrected moment estimates
        c_1st_moment = self._1st_moment / (1 - beta1**self._t)
        c_2nd_moment = self._2nd_moment / (1 - beta2**self._t)

        # resulting gradients
        return c_1st_moment / (c_2nd_moment.sqrt() + eps)


class GradProjLinear(GradProjLayer):
    def __init__(self, in_dim: int, out_dim: int, proj_calculator) -> None:
        super(GradProjLinear, self).__init__(proj_calculator)
        self._layer = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._collecting_on:
            return self._layer(x)
        self._proj_calculator.register_input(x)
        with torch.no_grad():
            return self._layer(x)

    def project_grad(self, grad):
        # PyTorch's linear layer: y = Batch x A.t()
        # We need to project the gradient vectors of A.t(), i.e. grad.t(),
        # and then repack them into A format: (Proj x grad.t()).t()
        # (Proj x grad.t()).t() = grad x Proj.t() = Grad x Proj
        # since Proj is symmetric.
        return grad @ self._proj

class GradProjConv2d(GradProjLayer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            proj_calculator,
            padding=(0, 0),
            stride=1):
        super(GradProjConv2d, self).__init__(proj_calculator)
        self._padding = padding
        self._stride = stride
        self._layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                padding=padding, stride=stride, bias=False)
        self._kernel_size = kernel_size
        self._in_channels = in_channels
        if type(kernel_size) is tuple:
            self._flat_dim = in_channels * kernel_size[0] * kernel_size[1]
        else:
            self._flat_dim =  in_channels * kernel_size**2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._collecting_on:
            return self._layer(x)
        # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        patches = nn.functional.unfold(x,self._kernel_size,
                        padding=self._padding, stride=self._stride)
        # patches size: (n_images, n_features, n_patches),
        # but we need: (n_images * n_patches, n_features)
        patches = patches.permute(0, 2, 1).reshape(-1, self._flat_dim)
        self._proj_calculator.register_input(patches)

        with torch.no_grad():
            return self._layer(x)

    def project_grad(self, grad):
        # grad size: (out_channels, in_channels, kernel_size, kernel_size)
        # features are unfolded in the same order:
        # (in_channels, kernel_size, kernel_size) -> in_channels x kernel_size x kernel_size
        # Like for linear layers, we need to project gradient vectors of dimension
        # in_dim from a matrix of dimensions in_dim x out_dim.
        # The same logic applied as to why we end up with grad x proj.
        grad2d = grad.view(grad.size(0), -1)
        projected2d =  grad2d @ self._proj
        projected3d = projected2d.view_as(grad)

        return projected3d


def gradproj_layers(module: nn.Module):
    for submodule in module.modules():
        if isinstance(submodule, GradProjLayer):
            yield submodule


@contextmanager
def proj_computation(network: nn.Module):
    for nullspace_l in gradproj_layers(network):
        nullspace_l.data_collecting_on()

    network.eval()
    with torch.no_grad():
        yield

    for nullspace_l in gradproj_layers(network):
        nullspace_l.data_collecting_off()
