PyTorch implementation of

[[1](https://arxiv.org/abs/2103.07113)] Shipeng Wang and Xiaorong Li and Jian Sun and Zongben Xu.
Training Networks in Null Space of Feature Covariance for Continual Learning, CVPR 2021
2021

[[2](https://arxiv.org/pdf/2103.09762.pdf)] Gobinda Saha, Isha Garg & Kaushik Roy.
Gradient Projection Memory for Continual Learning, ICLR 2021

Example of network equipped for continual learning:

```python3
from orthocl import (
    GradProjLinear, GradProjConv2d, NullSpace, gradproj_layers, proj_computation
)

net = nn.Sequential(
  GradProjConv2d(3, channels, (w, h), NullSpace(R=0.005)),
  nn.Flatten(),
  nn.LeakyReLU(0.25),
  GradProjLinear(channels * img_dim ** 2, n_classes, NullSpace(R=0.005)),
)
```

Before training a new task:
```python3
for layer in gradproj_layers(net):
  layer.take_snapshot()
```

After the first task, call `net.eval()` to freeze batchnorms and dropouts.

While training:
```python3
loss.backward()
for layer in gradproj_layers(net):
  layer.sgd_step(lr=0.01, momentum=0.5)
```

This should replace:
```
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

After training, before the next task:
```python3
with proj_computation(net):
  for batch in dataset:
    net(batch)
```

## Options

#### GPM[2]
```python3
GradProjConv2d(3, channels, (w, h), GPM(R=0.005))
GradProjLinear(in_dim, out_dim, GPM(R=0.005))
```

#### Zero plasticity

Disables weight update for the layers of your choice.

```python3
GradProjConv2d(3, channels, (w, h), ZeroPlasticity())
GradProjLinear(in_dim, out_dim, ZeroPlasticity())
```

### Zero stability / Full plasticity

Disables the algorithm. (recommended as baseline)

```python3
GradProjConv2d(3, channels, (w, h), NullSpace()).disable()
GradProjLinear(in_dim, out_dim, NullSpace()).disable()
```

### Minimum plasticity

This computes the null space projection matrix of the uncentered covariance matrix
with `1 - U x tr(U)` where `U,S,V = SVD(cov)`.

In this way, the null space is not approximated, but it doesn't leave much wiggle
room to weight updates.

```python3
GradProjConv2d(3, channels, (w, h), LowPlasticity()).disable()
GradProjLinear(in_dim, out_dim, LowPlasticity()).disable()
```

### Double projection

As Adam and RMSprop accumulate second-order moments from destructive gradients,
it might be worth projecting the gradients *prior to* updating the second-order moments,
and projecting again the gradients after adjusting them with the moments.

In theory, if the nullspace is well approximated, the resulting projection matrix
will be idempotent, thus gradients can be projected multiple times.
However, in practice, as the approximation is not perfect, this could make
things worse.

Use this option at your own peril.


```python3
GradProjConv2d(3, channels, (w, h), NullSpace()).two_proj()
GradProjLinear(in_dim, out_dim, NullSpace()).two_proj()
```

### Post projection

I've noticed difference in precision between SVD on CPU and SVD on GPU with
my version of PyTorch.

If you want to be sure to avoid accumulating errors, you can ask the algorithm
to keep around the sum of the unprojected gradients and to project the sum of
gradients of the entire task onto the nullspace at every optimization step.

```python3
GradProjConv2d(3, channels, (w, h), NullSpace()).post_proj()
GradProjLinear(in_dim, out_dim, NullSpace()).post_proj()
```

### Adam, RMSprop

```python3
for layer in gradproj_layers(net):
  layer.adam_step(lr=0.01)
```

```python3
for layer in gradproj_layers(net):
  layer.rms_step(lr=0.01)
```

### Bonus

coming soon

### Recommended versions

| tool |  version|
|--------|--------|
| python | 3.8.5  |
| torch  | 1.8.0  |
| numpy  | 1.20.1 |

### Installation

```bash
python3 setup.py install --user
```

or

```bash
sudo python3 setup.py install
```
