PyTorch implementation of

[[1](https://arxiv.org/abs/2103.07113)] Shipeng Wang and Xiaorong Li and Jian Sun and Zongben Xu.
Training Networks in Null Space of Feature Covariance for Continual Learning, CVPR 2021 (oral paper)

[[2](https://arxiv.org/pdf/2103.09762.pdf)] Gobinda Saha, Isha Garg & Kaushik Roy.
Gradient Projection Memory for Continual Learning, ICLR 2021

Example of network equipped for continual learning with orthocl:

```python3
from orthocl import (
    GradProjLinear, GradProjConv2d, NullSpace, gradproj_layers, proj_computation
)

net = nn.Sequential(
  GradProjConv2d(3, channels, (h, w), NullSpace(R=0.005)),
  nn.Flatten(),
  nn.LeakyReLU(0.25),
  GradProjLinear(channels * img_dim ** 2, n_classes, NullSpace(R=0.005)),
)
```

Before training the network on a new task, execute:
```python3
for layer in gradproj_layers(net):
  layer.take_snapshot()
```

After the first task, call `net.eval()` to freeze batchnorms and dropouts.

In the training loop, you don't need an optimizer. Simply update the weights with `sgd_step`:
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

After training, and before the next task, execute:
```python3
with proj_computation(net):
  for batch in dataset:
    net(batch)
```

## Example

You can try a number of different scenarios with [rotated_mnist.py](rotated_mnist.py).

#### Results with `NullSpace(R=0.01)`

| angle | accuracy with null space (%) | baseline accuracy (%) |
|-------|-----|----- |
| 0     | 86  | 56   |
| 90    | 86  | 59   |
| -45   | 90  | 38   |
| 45    | 88  | 97   |

The accuracy is measured after training the 4 tasks.
As you can see, the baseline algorithm forgets old tasks as it learns new tasks.

(I haven't tuned any hyper-parameter so I do not recommend that you use those numbers to compare orthocl with other CL algorithms)

## Options

##### GPM[[2](https://arxiv.org/pdf/2103.09762.pdf)]
```python3
m = 32768  # requires to avoid running out of memory
GradProjConv2d(3, channels, (h, w), GPM(R=0.05,  max_samples=m))
GradProjLinear(in_dim, out_dim, GPM(R=0.05,  max_samples=m))
```

##### Zero plasticity

Disables weight update for the layers of your choice.

```python3
GradProjConv2d(3, channels, (h, w), ZeroPlasticity())
GradProjLinear(in_dim, out_dim, ZeroPlasticity())
```

##### Zero stability / full plasticity

Disables the projection. (recommended as baseline)

```python3
GradProjConv2d(3, channels, (h, w), NullSpace()).disable()
GradProjLinear(in_dim, out_dim, NullSpace()).disable()
```

##### Minimum plasticity

This computes the null space projection matrix of the uncentered covariance matrix
with `1 - U x U.t()` where `U,S,V = SVD(cov)`.

In this way, the null space is not approximated. On the downside, it doesn't leave much wiggle
room to weight updates.

```python3
GradProjConv2d(3, channels, (h, w), LowPlasticity())
GradProjLinear(in_dim, out_dim, LowPlasticity())
```

##### Double projection

As Adam and RMSprop accumulate second-order moments from destructive gradients,
it might be worth projecting the gradients *prior to* updating the second-order moments,
and projecting again the gradients after adjusting them with the moments.

In theory, if the nullspace is well approximated, the resulting projection matrix
should be idempotent, thus gradients can be projected multiple times.
However, in practice, as the approximation is not perfect, this could make
things worse.

Use this option at your own peril.


```python3
GradProjConv2d(3, channels, (h, w), NullSpace()).two_proj()
GradProjLinear(in_dim, out_dim, NullSpace()).two_proj()
```

##### Post projection

I've noticed a difference in precision between SVD on CPU and SVD on GPU with
my version of PyTorch.

If you want to be sure to avoid accumulating errors, you can have the algorithm
keep around the sum of the unprojected gradients and project the gradient sum
at every optimization step.

```python3
GradProjConv2d(3, channels, (h, w), NullSpace()).post_proj()
GradProjLinear(in_dim, out_dim, NullSpace()).post_proj()
```

##### L1 distance

Parameters are saved when you call `take_snapshot`.
You can compute the differentiable L1 distance between the current parameters and the saved parameters by calling `l1dist` on each layer.

```python3
l1loss = 0
for layer in gradproj_layers(net):
  l1loss = layer.l1dist() + l1loss
```

note: the distance is calculated with a `sum`, not a `mean`.

##### Adam, RMSprop

```python3
for layer in gradproj_layers(net):
  layer.adam_step(lr=0.01)
```

```python3
for layer in gradproj_layers(net):
  layer.rms_step(lr=0.01)
```


### Recommended versions

| tool   | version| required by |
|--------|--------| ------------|
| python | 3.8.5  | orthocl     |
| torch  | 1.8.0  | orthocl     |
| numpy  | 1.20.1 | orthocl     |
| tqdm   | 4.46.1 | rotated_mnist.py |
| torchvision | 0.9.0 | rotated_mnist.py |

### Installation

```bash
python3 setup.py install --user
```

or

```bash
sudo python3 setup.py install
```
