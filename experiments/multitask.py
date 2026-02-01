# Multi-task learning module
# Implements shared trunk + task-specific heads for Risk_Score and Save_Money_Yes
# Conservative architecture with strict controls

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, accuracy_score, precision_score, recall_score
from scipy.stats import spearmanr


class MultiTaskModel(nn.Module):
    """
    Multi-task neural network with shared trunk and task-specific heads.

    Architecture:
    - Shared trunk: 1-2 hidden layers (conservative)
    - Risk head: Regression (Huber/MSE loss)
    - Savings head: Binary classification (BCEWithLogitsLoss)
    """

    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3, activation='relu'):
        super(MultiTaskModel, self).__init__()

        # Shared trunk
        trunk_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            trunk_layers.append(nn.Linear(prev_dim, hidden_dim))
            trunk_layers.append(nn.ReLU() if activation == 'relu' else nn.GELU())
            trunk_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.shared_trunk = nn.Sequential(*trunk_layers)

        # Task-specific heads
        self.risk_head = nn.Linear(prev_dim, 1)  # Regression output
        self.savings_head = nn.Linear(prev_dim, 1)  # Binary classification (logits)

    def forward(self, x):
        """
        Forward pass through shared trunk and both heads.

        Returns:
            risk_pred: (batch_size, 1) - regression predictions
            savings_logits: (batch_size, 1) - classification logits
        """
        shared_features = self.shared_trunk(x)
        risk_pred = self.risk_head(shared_features)
        savings_logits = self.savings_head(shared_features)

        return risk_pred, savings_logits


class MultiTaskTrainer:
    """
    Trainer for multi-task model with early stopping and gradient clipping.
    """

    def __init__(self, model, device='cpu', lr=0.001, weight_decay=0.01,
                 risk_loss_fn='huber', clip_grad=1.0, patience=10):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.clip_grad = clip_grad
        self.patience = patience

        # Loss functions
        if risk_loss_fn == 'huber':
            self.risk_loss_fn = nn.HuberLoss()
        else:
            self.risk_loss_fn = nn.MSELoss()

        self.savings_loss_fn = nn.BCEWithLogitsLoss()

    def train_epoch(self, train_loader):
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0

        for X_batch, y_risk_batch, y_savings_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_risk_batch = y_risk_batch.to(self.device)
            y_savings_batch = y_savings_batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            risk_pred, savings_logits = self.model(X_batch)

            # Compute losses
            risk_loss = self.risk_loss_fn(risk_pred.squeeze(), y_risk_batch)
            savings_loss = self.savings_loss_fn(savings_logits.squeeze(), y_savings_batch)

            # Combined loss (equal weighting)
            loss = risk_loss + savings_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """Validate and return loss + metrics."""
        self.model.eval()
        total_loss = 0.0

        all_risk_preds = []
        all_risk_true = []
        all_savings_preds = []
        all_savings_true = []

        with torch.no_grad():
            for X_batch, y_risk_batch, y_savings_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_risk_batch = y_risk_batch.to(self.device)
                y_savings_batch = y_savings_batch.to(self.device)

                # Forward pass
                risk_pred, savings_logits = self.model(X_batch)

                # Compute losses
                risk_loss = self.risk_loss_fn(risk_pred.squeeze(), y_risk_batch)
                savings_loss = self.savings_loss_fn(savings_logits.squeeze(), y_savings_batch)
                loss = risk_loss + savings_loss

                total_loss += loss.item()

                # Store predictions
                all_risk_preds.extend(risk_pred.squeeze().cpu().numpy())
                all_risk_true.extend(y_risk_batch.cpu().numpy())

                savings_probs = torch.sigmoid(savings_logits.squeeze())
                all_savings_preds.extend((savings_probs > 0.5).cpu().numpy().astype(int))
                all_savings_true.extend(y_savings_batch.cpu().numpy().astype(int))

        # Compute metrics
        metrics = self._compute_metrics(
            np.array(all_risk_true),
            np.array(all_risk_preds),
            np.array(all_savings_true),
            np.array(all_savings_preds)
        )

        avg_loss = total_loss / len(val_loader)
        return avg_loss, metrics

    def _compute_metrics(self, y_risk_true, y_risk_pred, y_savings_true, y_savings_pred):
        """Compute all evaluation metrics."""
        metrics = {}

        # Risk metrics (regression)
        metrics['risk_mae'] = mean_absolute_error(y_risk_true, y_risk_pred)
        metrics['risk_rmse'] = np.sqrt(mean_squared_error(y_risk_true, y_risk_pred))
        metrics['risk_spearman'] = spearmanr(y_risk_true, y_risk_pred)[0]

        ss_res = np.sum((y_risk_true - y_risk_pred) ** 2)
        ss_tot = np.sum((y_risk_true - np.mean(y_risk_true)) ** 2)
        metrics['risk_r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Savings metrics (classification)
        metrics['savings_macro_f1'] = f1_score(y_savings_true, y_savings_pred, average='macro', zero_division=0)
        metrics['savings_accuracy'] = accuracy_score(y_savings_true, y_savings_pred)
        metrics['savings_precision'] = precision_score(y_savings_true, y_savings_pred, average='macro', zero_division=0)
        metrics['savings_recall'] = recall_score(y_savings_true, y_savings_pred, average='macro', zero_division=0)

        return metrics

    def fit(self, train_loader, val_loader, max_epochs=100):
        """
        Train with early stopping.

        Returns:
            best_metrics: dict of best validation metrics
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_metrics = None

        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        return best_metrics

    def predict(self, X):
        """
        Predict both tasks for input X.

        Returns:
            risk_preds: np.array - regression predictions
            savings_preds: np.array - binary predictions
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            risk_pred, savings_logits = self.model(X_tensor)
            savings_probs = torch.sigmoid(savings_logits.squeeze())
            savings_preds = (savings_probs > 0.5).cpu().numpy().astype(int)
            risk_preds = risk_pred.squeeze().cpu().numpy()

        return risk_preds, savings_preds


def train_single_task_risk(X_train, y_train, X_val, y_val, config, seed):
    """Train risk-only model (ablation baseline)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_train.shape[1]
    hidden_dims = config.get('hidden_dims', [64, 32])
    dropout = config.get('dropout', 0.3)

    # Create model with only risk head
    model = MultiTaskModel(input_dim, hidden_dims, dropout)

    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.zeros(len(y_train))  # Dummy savings labels
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
        torch.zeros(len(y_val))
    )

    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32))

    # Train
    trainer = MultiTaskTrainer(
        model,
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.01),
        risk_loss_fn=config.get('risk_loss', 'huber'),
        clip_grad=config.get('clip_grad', 1.0),
        patience=config.get('patience', 10)
    )

    metrics = trainer.fit(train_loader, val_loader, max_epochs=config.get('max_epochs', 100))

    # Extract only risk metrics
    risk_metrics = {k: v for k, v in metrics.items() if k.startswith('risk_')}
    return risk_metrics, trainer.model


def train_single_task_savings(X_train, y_train, X_val, y_val, config, seed):
    """Train savings-only model (ablation baseline)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_train.shape[1]
    hidden_dims = config.get('hidden_dims', [64, 32])
    dropout = config.get('dropout', 0.3)

    # Create model with only savings head
    model = MultiTaskModel(input_dim, hidden_dims, dropout)

    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.zeros(len(y_train)),  # Dummy risk labels
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.zeros(len(y_val)),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32))

    # Train
    trainer = MultiTaskTrainer(
        model,
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.01),
        risk_loss_fn=config.get('risk_loss', 'huber'),
        clip_grad=config.get('clip_grad', 1.0),
        patience=config.get('patience', 10)
    )

    metrics = trainer.fit(train_loader, val_loader, max_epochs=config.get('max_epochs', 100))

    # Extract only savings metrics
    savings_metrics = {k: v for k, v in metrics.items() if k.startswith('savings_')}
    return savings_metrics, trainer.model


def train_multitask(X_train, y_risk_train, y_savings_train,
                   X_val, y_risk_val, y_savings_val,
                   config, seed):
    """Train multi-task model with both heads."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_train.shape[1]
    hidden_dims = config.get('hidden_dims', [64, 32])
    dropout = config.get('dropout', 0.3)

    # Create full multi-task model
    model = MultiTaskModel(input_dim, hidden_dims, dropout)

    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_risk_train),
        torch.FloatTensor(y_savings_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_risk_val),
        torch.FloatTensor(y_savings_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32))

    # Train
    trainer = MultiTaskTrainer(
        model,
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.01),
        risk_loss_fn=config.get('risk_loss', 'huber'),
        clip_grad=config.get('clip_grad', 1.0),
        patience=config.get('patience', 10)
    )

    metrics = trainer.fit(train_loader, val_loader, max_epochs=config.get('max_epochs', 100))

    return metrics, trainer.model
