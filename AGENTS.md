# AGENTS.md - Scaled Conjugate Gradient Project

## Project Overview

This project implements the **Scaled Conjugate Gradient (SCG)** optimizer for PyTorch, based on the 1993 paper by Martin Føslette Møller. SCG is a second-order optimization method that avoids expensive line searches by using a Levenberg-Marquardt approach to scale the step size. It's particularly effective for deterministic (full-batch) optimization problems, typically relevant with small networks and datasets.

### Limitations

- SCG falls behind Adam and other stochastic optimizers with minibatch gradient descent
- 

## Main Dependencies

- **torch** >= 2.9.1 PyTorch for neural networks
- **torchvision** >= 0.24.1 For MNIST dataset loading
- **matplotlib** >= 3.10.8 For plotting results

## Optimizer Details

The SCG algorithm:
1. Computes gradient and search direction
2. Calculates second-order information via finite differences
3. Scales curvature with trust region parameter
4. Makes Hessian positive definite if needed
5. Calculates step size
6. Evaluates step quality using comparison ratio
7. Updates or rejects step based on quality
8. Adjusts trust region parameter

### `scg.py`
- **Purpose**: Core SCG optimizer implementation
- **Class**: `SCG(Optimizer)` - PyTorch optimizer following Møller 1993 algorithm
- **Key Features**:
  - Unlike standard PyTorch optimizers, `step()` takes a loss function (not a closure)
  - Handles all gradient computations internally
  - Uses trust region parameter (lambda) to scale step size
  - Implements Polak-Ribière conjugate gradient updates
  - Restarts to steepest descent every N iterations (where N = number of parameters)

### Optimizer Usage
```python
# SCG requires a loss function, not a closure
optimizer = SCG(model.parameters())
loss = optimizer.step(lambda: loss_fn(model(X), y))

# Standard optimizers use backward()
optimizer.zero_grad()
loss = loss_fn(model(X), y)
loss.backward()
optimizer.step()
```

## Development Tips

- This is a uv project so use: uv add, uv run, etc. 
- Strive for elegance and simplicity, thinking about mathematical principles best practices

## Testing

Run examples:
- `uv run rosenbrock.py` - Test on Rosenbrock function
- `uv run mnist.py` - Benchmark on MNIST classification

