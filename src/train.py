"""Production Training Pipeline with MLflow Tracking and NASA Asymmetric Scoring.

Trains the Multi-Scale CNN-Transformer on C-MAPSS FD001, tracking losses,
RMSE, MAE, and the NASA exponential penalty score in MLflow.
"""

from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import mlflow

from src.data_loader import load_cmapss_raw
from src.preprocess import CmapssPreprocessor
from src.models.cnn_transformer import CNNTransformer, PhysicsInformedRULLoss


def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes NASA asymmetric penalty score for turbofan RUL predictions.
    
    Late predictions (predicting engine has more life than it does) are penalized
    exponentially harder than early predictions due to catastrophic failure risk.
    """
    diff = y_pred - y_true
    scores = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(scores))


def train_model(
    dataset_id: str = "FD001",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    sequence_length: int = 30,
    d_model: int = 64,
    lambda_mono: float = 0.1,
    experiment_name: str = "Turbofan_RUL_CNN_Transformer",
) -> None:
    """Unified training loop with MLflow logging and checkpoint saving."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device} | Dataset: {dataset_id}")

    # 1. Load and Preprocess Data
    print(f"[Data] Loading C-MAPSS {dataset_id} benchmark data...")
    df_train, df_test, y_test = load_cmapss_raw(dataset_id)

    preprocessor = CmapssPreprocessor(sequence_length=sequence_length)
    X_train, y_train = preprocessor.fit_transform(df_train, use_sg=True)
    X_test = preprocessor.transform_test(df_test, use_sg=True)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. Instantiate Model and Loss
    model = CNNTransformer(
        in_features=X_train.shape[2],
        d_model=d_model,
        nhead=4,
        num_layers=2,
    ).to(device)

    criterion = PhysicsInformedRULLoss(lambda_monotonicity=lambda_mono)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Setup MLflow Tracking
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"CNN_Transformer_{dataset_id}"):
        mlflow.log_params({
            "dataset": dataset_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "sequence_length": sequence_length,
            "d_model": d_model,
            "lambda_monotonicity": lambda_mono,
            "device": str(device),
        })

        best_rmse = float("inf")
        checkpoint_dir = Path(__file__).resolve().parent.parent / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        best_model_path = checkpoint_dir / f"best_model_{dataset_id}.pt"

        # 4. Training Loop
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * len(batch_y)

            epoch_loss = running_loss / len(train_dataset)
            scheduler.step()

            # 5. Validation Evaluation on Test Engines
            model.eval()
            with torch.no_grad():
                test_tensor = torch.from_numpy(X_test).to(device)
                test_preds = model(test_tensor).cpu().numpy()

            mae = float(np.mean(np.abs(test_preds - y_test)))
            rmse = float(np.sqrt(np.mean((test_preds - y_test) ** 2)))
            nasa_score = compute_nasa_score(y_test, test_preds)

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {epoch_loss:.3f} | Test RMSE: {rmse:.2f} | MAE: {mae:.2f} | NASA Score: {nasa_score:.1f}"
            )

            mlflow.log_metrics(
                {"train_loss": epoch_loss, "val_rmse": rmse, "val_mae": mae, "nasa_score": nasa_score},
                step=epoch,
            )

            if rmse < best_rmse:
                best_rmse = rmse
                torch.save(model.state_dict(), best_model_path)
                # Also save standard best_model.pt for default API loading
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
                print(f"  --> Checkpoint saved: New best RMSE {best_rmse:.2f} at epoch {epoch}")

        mlflow.log_artifact(str(best_model_path), artifact_path="model_weights")
        print(f"[Done] Training complete. Best model saved to: {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-Transformer RUL Model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="C-MAPSS sub-dataset identifier (default: FD001)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_model(dataset_id=args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
