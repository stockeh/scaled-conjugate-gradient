# Scaled Conjugate Gradient in PyTorch

A PyTorch implementation of the Scaled Conjugate Gradient (SCG) optimizer, based on the 1993 paper by Martin Føslette Møller. SCG is a second-order optimization method that uses a Levenberg-Marquardt approach to scale step sizes, avoiding expensive line searches. It is particularly effective for deterministic (full-batch) optimization problems.

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management. Install the project dependencies:

```bash
uv sync
```

## Usage

SCG differs from standard PyTorch optimizers in that `step()` takes a loss function (maybe closure for future):

```python
from scg import SCG

optimizer = SCG(model.parameters())
loss = optimizer.step(lambda: loss_fn(model(X), y))
```

## Examples

Run the included examples:

```bash
# Test on Rosenbrock function
uv run rosenbrock.py

# Benchmark on MNIST classification
uv run mnist.py
```