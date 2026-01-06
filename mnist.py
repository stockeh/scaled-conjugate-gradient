import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from tqdm import tqdm

from scg import SCG

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_mnist(device):
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, transform=transform)

    Xtrain = (train_ds.data.float() / 255.0).to(device)
    Ttrian = train_ds.targets.to(device)
    Xval = (test_ds.data.float() / 255.0).to(device)
    Tval = test_ds.targets.to(device)

    return Xtrain, Ttrian, Xval, Tval


def make_batches(X, y, batch_size):
    """Split data into batches. batch_size=-1 means full batch."""
    n = X.shape[0]
    if batch_size == -1:
        return [(X, y)]
    return [
        (X[i : i + batch_size], y[i : i + batch_size]) for i in range(0, n, batch_size)
    ]


def compute_accuracy(model, X, y):
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item() * 100


def train(
    model,
    Xtrain,
    Ttrian,
    Xval,
    Tval,
    n_epochs,
    batch_size,
    optimizer_name,
    lr=0.001,
):
    """Train model with specified optimizer."""
    if optimizer_name == "SCG":
        optimizer = SCG(model.parameters())
    else:
        optimizer = Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    batches = make_batches(Xtrain, Ttrian, batch_size)

    for _ in tqdm(range(n_epochs), desc=optimizer_name):
        epoch_loss = 0.0

        for X_batch, y_batch in batches:
            if optimizer_name == "SCG":
                loss = optimizer.step(lambda: F.cross_entropy(model(X_batch), y_batch))
            else:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(batches))

        with torch.no_grad():
            val_loss = F.cross_entropy(model(Xval), Tval)
            val_losses.append(val_loss.item())

    train_acc = compute_accuracy(model, Xtrain, Ttrian)
    val_acc = compute_accuracy(model, Xval, Tval)

    return train_losses, val_losses, train_acc, val_acc


def run_experiment(Xtrain, Ttrian, Xval, Tval, batch_size, n_epochs, device):
    results = {}

    for name in ["SCG", "Adam"]:
        torch.manual_seed(42)
        model = MLP().to(device)
        train_losses, val_losses, train_acc, val_acc = train(
            model, Xtrain, Ttrian, Xval, Tval, n_epochs, batch_size, name
        )
        results[name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }

    return results


def plot_results(all_results, batch_sizes):
    n_batches = len(batch_sizes)
    fig, axes = plt.subplots(n_batches, 2, figsize=(12, 4 * n_batches))
    if n_batches == 1:
        axes = axes.reshape(1, -1)

    colors = {"SCG": "black", "Adam": "blue"}

    for i, (bs, results) in enumerate(zip(batch_sizes, all_results)):
        bs_label = "Full" if bs == -1 else bs

        for name, data in results.items():
            d = torch.exp(-torch.tensor(data["train_losses"]))
            axes[i, 0].plot(d, label=name, color=colors[name])
        axes[i, 0].set_title(f"Training (batch={bs_label})")
        axes[i, 0].set_xlabel("Epoch")
        axes[i, 0].set_ylabel("Likelihood")
        axes[i, 0].legend()

        for name, data in results.items():
            d = torch.exp(-torch.tensor(data["val_losses"]))
            axes[i, 1].plot(d, label=name, color=colors[name])
        axes[i, 1].set_title(f"Validation (batch={bs_label})")
        axes[i, 1].set_xlabel("Epoch")
        axes[i, 1].set_ylabel("Likelihood")
        axes[i, 1].legend()

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print(f"Testing SCG on MNIST | Device: {DEVICE}")
    print("=" * 60)

    Xtrain, Ttrian, Xval, Tval = load_mnist(DEVICE)
    print(f"Train: {Xtrain.shape}, Val: {Xval.shape}")

    n_epochs = 200
    batch_sizes = [-1]

    all_results = []
    for bs in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch size: {'Full' if bs == -1 else bs}")
        print("=" * 60)
        results = run_experiment(Xtrain, Ttrian, Xval, Tval, bs, n_epochs, DEVICE)
        all_results.append(results)

        print(
            f"\n{'Optimizer':<10} {'Train Acc':>12} {'Val Acc':>12} {'Train Loss':>12} {'Val Loss':>12}"
        )
        print("-" * 60)
        for name, data in results.items():
            print(
                f"{name:<10} {data['train_acc']:>11.2f}% {data['val_acc']:>11.2f}% "
                f"{data['train_losses'][-1]:>12.4f} {data['val_losses'][-1]:>12.4f}"
            )

    plot_results(all_results, batch_sizes)


if __name__ == "__main__":
    main()
