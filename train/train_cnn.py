"""
train/train_cnn.py

Train a 1D CNN regressor on J1–J2 energies.

This script ties together:
- QuantumJ1J2Solver (physics)
- CNN input features (x, y, z channels along the chain)
- J1J2CNNRegressor1D (custom CNN)
- Optimization and training loop

It saves loss + MAE curves into the 'plots/' directory.
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from physics.j1j2_solver import QuantumJ1J2Solver
from models.cnn_regressor import J1J2CNNRegressor1D
from utils.feature_utils import angles_to_cnn_input
from utils.data_utils import train_val_test_split
from utils.training_utils import train_one_epoch, evaluate, build_optimizer


def main():
    # -----------------------------
    # 1. Configuration
    # -----------------------------
    OPTIMIZER_NAME = "adam"  # "adam", "sgd", "adamw"
    WEIGHT_DECAY = 1e-4      # L2 regularization

    USE_LR_SCHEDULER = True
    LR_FACTOR = 0.5
    LR_PATIENCE = 10

    USE_EARLY_STOPPING = True
    ES_PATIENCE = 20

    # Basic training hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    n_spins = 5
    J1 = 1.0
    J2 = 0.5
    n_samples = 4000
    batch_size = 64
    n_epochs = 100
    lr = 1e-3

    os.makedirs("plots", exist_ok=True)

    # -----------------------------
    # 2. Generate dataset
    # -----------------------------
    print("Building solver and generating data...")
    solver = QuantumJ1J2Solver(n_spins=n_spins, J1=J1, J2=J2)
    theta, phi, energies = solver.get_training_data(n_samples)

    x_all = angles_to_cnn_input(theta, phi)  # (B, 3, n_spins)
    y_all = energies

    print("Using CNN with input shape:", x_all.shape)

    # -----------------------------
    # 3. Train/val/test split
    # -----------------------------
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = train_val_test_split(
        x_all, y_all,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        shuffle=True,
        seed=42
    )

    print(f"Train size: {x_train.shape[0]}")
    print(f"Val size:   {x_val.shape[0]}")
    print(f"Test size:  {x_test.shape[0]}")

    # -----------------------------
    # 4. Build model, optimizer, scheduler
    # -----------------------------
    model = J1J2CNNRegressor1D(n_spins=n_spins, output_dim=1).to(device)

    criterion = nn.MSELoss()
    optimizer = build_optimizer(model, OPTIMIZER_NAME, lr=lr, weight_decay=WEIGHT_DECAY)

    if USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            verbose=True
        )
    else:
        scheduler = None

    train_losses = []
    val_losses = []
    val_maes = []

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    # -----------------------------
    # 5. Training loop
    # -----------------------------
    print("Starting CNN training...")
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            model, optimizer, criterion,
            x_train, y_train, batch_size, device
        )
        val_loss, val_mae = evaluate(model, criterion, x_val, y_val, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        # LR scheduler step
        if scheduler is not None:
            scheduler.step(val_loss)

        # Early stopping tracking
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Train MSE: {train_loss:.6f} | "
                  f"Val MSE: {val_loss:.6f} | "
                  f"Val MAE: {val_mae:.6f}")

        if USE_EARLY_STOPPING and epochs_no_improve >= ES_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # -----------------------------
    # 6. Final evaluation on test set
    # -----------------------------
    test_loss, test_mae = evaluate(model, criterion, x_test, y_test, device)
    print("\nFinal CNN test performance (best model):")
    print(f"Test MSE: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    # -----------------------------
    # 7. Plot and save curves
    # -----------------------------
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train MSE")
    plt.plot(epochs, val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"CNN ({OPTIMIZER_NAME}) Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/cnn_{OPTIMIZER_NAME}_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, val_maes, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.title(f"CNN ({OPTIMIZER_NAME}) Validation MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/cnn_{OPTIMIZER_NAME}_val_mae.png")
    plt.close()

    print("CNN training curves saved to 'plots/' directory.")


if __name__ == "__main__":
    main()
