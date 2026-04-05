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

    CRITICAL FIX: Supports mode for proper single-task ablation.
    - mode="multitask": optimizes both losses (default)
    - mode="risk_only": optimizes only risk loss (savings head gets no gradient)
    - mode="savings_only": optimizes only savings loss (risk head gets no gradient)

    UPGRADES:
    - Loss balancing: loss = α * risk_loss + β * savings_loss
    - Gradient conflict detection: logs ||∇trunk L_risk|| vs ||∇trunk L_savings||
    - Threshold optimization: finds optimal threshold for savings on validation
    - PCGrad-lite: optional gradient projection when tasks conflict
    """

    def __init__(self, model, device='cpu', lr=0.001, weight_decay=0.01,
                 risk_loss_fn='huber', clip_grad=1.0, patience=10,
                 mode='multitask', risk_weight=1.0, savings_weight=1.0,
                 use_pcgrad=False, log_gradients=False,
                 dynamic_weighting=False, target_grad_ratio=1.0,
                 weight_update_rate=0.05, min_task_weight=0.2, max_task_weight=5.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.clip_grad = clip_grad
        self.patience = patience

        # Mode for ablation: "multitask", "risk_only", "savings_only"
        assert mode in ['multitask', 'risk_only', 'savings_only'], \
            f"Invalid mode: {mode}. Must be 'multitask', 'risk_only', or 'savings_only'"
        self.mode = mode

        # Loss weights for balancing (FIX: loss = α * risk_loss + β * savings_loss)
        self.risk_weight = risk_weight
        self.savings_weight = savings_weight

        # Gradient conflict handling
        self.use_pcgrad = use_pcgrad
        self.log_gradients = log_gradients
        self.gradient_logs = []  # Store gradient norms per epoch

        # Optional dynamic weighting to prevent savings domination.
        self.dynamic_weighting = dynamic_weighting
        self.target_grad_ratio = target_grad_ratio
        self.weight_update_rate = weight_update_rate
        self.min_task_weight = min_task_weight
        self.max_task_weight = max_task_weight

        # Loss functions
        if risk_loss_fn == 'huber':
            self.risk_loss_fn = nn.HuberLoss()
        else:
            self.risk_loss_fn = nn.MSELoss()

        self.savings_loss_fn = nn.BCEWithLogitsLoss()

        # Threshold optimization state
        self.optimal_threshold = 0.5  # Will be updated during training
        self.threshold_history = []

    def _update_task_weights(self, grad_ratio: float):
        """Adapt task weights based on observed risk/savings trunk gradient ratio."""
        if not self.dynamic_weighting or self.mode != 'multitask':
            return

        safe_ratio = max(float(grad_ratio), 1e-6)
        target = max(float(self.target_grad_ratio), 1e-6)
        # If risk gradients are too small, increase risk weight and reduce savings weight.
        factor = target / safe_ratio
        bounded = float(np.clip(factor, 0.8, 1.25))
        step = self.weight_update_rate

        risk_mult = 1.0 + step * (bounded - 1.0)
        savings_mult = 1.0 + step * ((1.0 / bounded) - 1.0)

        self.risk_weight = float(np.clip(self.risk_weight * risk_mult, self.min_task_weight, self.max_task_weight))
        self.savings_weight = float(np.clip(self.savings_weight * savings_mult, self.min_task_weight, self.max_task_weight))

    def _compute_trunk_gradients(self, loss, task_name):
        """Compute gradient norms on shared trunk for a specific task loss."""
        # Clear existing gradients
        self.optimizer.zero_grad()

        # Backward to compute gradients
        loss.backward(retain_graph=True)

        # Compute norm of trunk gradients
        trunk_grad_norm = 0.0
        trunk_grads = []
        for param in self.model.shared_trunk.parameters():
            if param.grad is not None:
                trunk_grad_norm += param.grad.norm().item() ** 2
                trunk_grads.append(param.grad.clone())
        trunk_grad_norm = trunk_grad_norm ** 0.5

        return trunk_grad_norm, trunk_grads

    def _pcgrad_project(self, grad1, grad2):
        """
        PCGrad-lite: Project grad1 to remove conflicting component with grad2.
        If grad1 · grad2 < 0 (conflict), project grad1 onto plane perpendicular to grad2.
        """
        # Flatten gradients
        g1_flat = torch.cat([g.flatten() for g in grad1])
        g2_flat = torch.cat([g.flatten() for g in grad2])

        # Compute dot product
        dot = torch.dot(g1_flat, g2_flat)

        if dot < 0:  # Conflict detected
            # Project g1 onto plane perpendicular to g2
            g2_norm_sq = torch.dot(g2_flat, g2_flat)
            if g2_norm_sq > 1e-8:
                proj = (dot / g2_norm_sq) * g2_flat
                g1_flat = g1_flat - proj

        # Unflatten back to original shapes
        projected = []
        offset = 0
        for g in grad1:
            size = g.numel()
            projected.append(g1_flat[offset:offset + size].view(g.shape))
            offset += size

        return projected

    def train_epoch(self, train_loader):
        """Train one epoch with loss balancing and optional gradient conflict handling."""
        self.model.train()
        total_loss = 0.0
        epoch_risk_grad_norms = []
        epoch_savings_grad_norms = []

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

            # LOSS GATING based on mode
            if self.mode == 'risk_only':
                loss = self.risk_weight * risk_loss
            elif self.mode == 'savings_only':
                loss = self.savings_weight * savings_loss
            else:  # multitask with balancing
                # FIX: Weighted loss combination
                loss = self.risk_weight * risk_loss + self.savings_weight * savings_loss

                # Optional: Log gradient norms for debugging
                if self.log_gradients and len(epoch_risk_grad_norms) < 3:  # Sample first 3 batches
                    risk_norm, risk_grads = self._compute_trunk_gradients(risk_loss, 'risk')
                    savings_norm, savings_grads = self._compute_trunk_gradients(savings_loss, 'savings')
                    epoch_risk_grad_norms.append(risk_norm)
                    epoch_savings_grad_norms.append(savings_norm)

                    # PCGrad: project gradients if in conflict
                    if self.use_pcgrad and risk_grads and savings_grads:
                        # Check for conflict (negative dot product)
                        risk_grads_proj = self._pcgrad_project(risk_grads, savings_grads)
                        savings_grads_proj = self._pcgrad_project(savings_grads, risk_grads)

                        # Apply projected gradients
                        self.optimizer.zero_grad()
                        idx = 0
                        for param in self.model.shared_trunk.parameters():
                            if param.grad is not None or risk_grads_proj[idx] is not None:
                                param.grad = risk_grads_proj[idx] + savings_grads_proj[idx]
                            idx += 1
                        # Continue to apply gradients to heads normally
                        loss.backward()
                    else:
                        # Normal backward
                        self.optimizer.zero_grad()
                        loss.backward()
                else:
                    loss.backward()

            # Gradient clipping
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()

            total_loss += loss.item()

        # Log gradient norms for this epoch
        if self.log_gradients and epoch_risk_grad_norms:
            ratio = np.mean(epoch_risk_grad_norms) / (np.mean(epoch_savings_grad_norms) + 1e-8)
            self.gradient_logs.append({
                'risk_grad_norm': np.mean(epoch_risk_grad_norms),
                'savings_grad_norm': np.mean(epoch_savings_grad_norms),
                'ratio': ratio,
                'risk_weight': self.risk_weight,
                'savings_weight': self.savings_weight,
            })
            self._update_task_weights(ratio)

        return total_loss / len(train_loader)

    def _find_optimal_threshold(self, savings_probs, y_true):
        """
        Find optimal threshold that maximizes macro-F1 on validation.
        Tests thresholds from 0.1 to 0.9 in steps of 0.05.
        """
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in np.arange(0.1, 0.91, 0.05):
            preds = (savings_probs > threshold).astype(int)
            f1 = f1_score(y_true, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold, best_f1

    def validate(self, val_loader, optimize_threshold=True):
        """Validate and return loss + metrics with loss gating and threshold optimization."""
        self.model.eval()
        total_loss = 0.0

        all_risk_preds = []
        all_risk_true = []
        all_savings_probs = []  # Store probs for threshold optimization
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

                # LOSS GATING based on mode
                if self.mode == 'risk_only':
                    loss = self.risk_weight * risk_loss
                elif self.mode == 'savings_only':
                    loss = self.savings_weight * savings_loss
                else:  # multitask
                    loss = self.risk_weight * risk_loss + self.savings_weight * savings_loss

                total_loss += loss.item()

                # Store predictions
                all_risk_preds.extend(risk_pred.squeeze().cpu().numpy())
                all_risk_true.extend(y_risk_batch.cpu().numpy())

                savings_probs = torch.sigmoid(savings_logits.squeeze()).cpu().numpy()
                all_savings_probs.extend(savings_probs)
                all_savings_true.extend(y_savings_batch.cpu().numpy().astype(int))

        # Convert to arrays
        all_savings_probs = np.array(all_savings_probs)
        all_savings_true = np.array(all_savings_true)

        # FIX: Threshold optimization for savings
        if optimize_threshold and self.mode != 'risk_only':
            optimal_thresh, optimal_f1 = self._find_optimal_threshold(all_savings_probs, all_savings_true)
            self.optimal_threshold = optimal_thresh
            self.threshold_history.append(optimal_thresh)
        else:
            optimal_thresh = 0.5

        # Apply optimal threshold for predictions
        all_savings_preds = (all_savings_probs > optimal_thresh).astype(int)

        # Compute metrics
        metrics = self._compute_metrics(
            np.array(all_risk_true),
            np.array(all_risk_preds),
            all_savings_true,
            all_savings_preds
        )

        # Add threshold info to metrics
        metrics['savings_threshold'] = optimal_thresh

        avg_loss = total_loss / len(val_loader)
        return avg_loss, metrics

    def _compute_metrics(self, y_risk_true, y_risk_pred, y_savings_true, y_savings_pred):
        """Compute all evaluation metrics."""
        metrics = {}

        # Risk metrics (regression)
        metrics['risk_mae'] = mean_absolute_error(y_risk_true, y_risk_pred)
        metrics['risk_rmse'] = np.sqrt(mean_squared_error(y_risk_true, y_risk_pred))

        # Spearman correlation: handle constant arrays gracefully
        # If predictions or true values are constant, correlation is undefined (set to 0)
        if np.std(y_risk_true) > 1e-8 and np.std(y_risk_pred) > 1e-8:
            try:
                metrics['risk_spearman'] = spearmanr(y_risk_true, y_risk_pred)[0]
            except Exception:
                metrics['risk_spearman'] = 0.0
        else:
            # One or both arrays are constant - correlation is undefined
            metrics['risk_spearman'] = 0.0

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
            best_metrics: dict of best validation metrics (includes gradient logs if enabled)
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_metrics = None
        best_threshold = 0.5
        training_log = []

        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader, optimize_threshold=True)

            # Track training progress
            training_log.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'risk_mae': val_metrics.get('risk_mae'),
                'savings_macro_f1': val_metrics.get('savings_macro_f1'),  # Consistent with plot expectations
                'savings_f1': val_metrics.get('savings_macro_f1'),  # Alias for compatibility
                'threshold': val_metrics.get('savings_threshold', 0.5)
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics.copy()
                best_threshold = val_metrics.get('savings_threshold', 0.5)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        # Store best threshold
        self.optimal_threshold = best_threshold

        # Add training metadata to metrics
        if best_metrics:
            best_metrics['_training_log'] = training_log
            best_metrics['_optimal_threshold'] = best_threshold
            if self.gradient_logs:
                best_metrics['_gradient_logs'] = self.gradient_logs.copy()

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
    """Train risk-only model (ablation baseline).

    CRITICAL: Uses mode='risk_only' so only risk loss is optimized.
    This ensures proper single-task ablation comparison.
    """
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
        torch.zeros(len(y_train))  # Dummy savings labels (not used due to mode)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
        torch.zeros(len(y_val))
    )

    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32))

    # Train with mode='risk_only' for proper ablation
    trainer = MultiTaskTrainer(
        model,
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.01),
        risk_loss_fn=config.get('risk_loss', 'huber'),
        clip_grad=config.get('clip_grad', 1.0),
        patience=config.get('patience', 10),
        mode='risk_only',  # CRITICAL: Only optimize risk loss
        risk_weight=config.get('risk_weight', 1.0),
        savings_weight=config.get('savings_weight', 1.0),
        use_pcgrad=False,
        log_gradients=False
    )

    metrics = trainer.fit(train_loader, val_loader, max_epochs=config.get('max_epochs', 100))

    # Extract only risk metrics
    risk_metrics = {k: v for k, v in metrics.items() if k.startswith('risk_')}
    return risk_metrics, trainer.model


def train_single_task_savings(X_train, y_train, X_val, y_val, config, seed):
    """Train savings-only model (ablation baseline).

    CRITICAL: Uses mode='savings_only' so only savings loss is optimized.
    This ensures proper single-task ablation comparison.
    """
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
        torch.zeros(len(y_train)),  # Dummy risk labels (not used due to mode)
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.zeros(len(y_val)),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32))

    # Train with mode='savings_only' for proper ablation
    trainer = MultiTaskTrainer(
        model,
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.01),
        risk_loss_fn=config.get('risk_loss', 'huber'),
        clip_grad=config.get('clip_grad', 1.0),
        patience=config.get('patience', 10),
        mode='savings_only',  # CRITICAL: Only optimize savings loss
        risk_weight=config.get('risk_weight', 1.0),
        savings_weight=config.get('savings_weight', 1.0),
        use_pcgrad=False,
        log_gradients=False
    )

    metrics = trainer.fit(train_loader, val_loader, max_epochs=config.get('max_epochs', 100))

    # Extract only savings metrics
    savings_metrics = {k: v for k, v in metrics.items() if k.startswith('savings_')}
    return savings_metrics, trainer.model


def train_multitask(X_train, y_risk_train, y_savings_train,
                   X_val, y_risk_val, y_savings_val,
                   config, seed):
    """Train multi-task model with both heads.

    Uses mode='multitask' to optimize both losses jointly.

    UPGRADES:
    - Loss balancing with risk_weight and savings_weight
    - Gradient logging for conflict detection
    - Threshold optimization for savings classification
    """
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

    # Get loss weights from config (default: equal weighting)
    risk_weight = config.get('risk_weight', 1.0)
    savings_weight = config.get('savings_weight', 1.0)
    use_pcgrad = config.get('use_pcgrad', False)
    log_gradients = config.get('log_gradients', True)  # Enable by default for debugging
    dynamic_weighting = config.get('dynamic_weighting', False)
    target_grad_ratio = config.get('target_grad_ratio', 1.0)
    weight_update_rate = config.get('weight_update_rate', 0.05)
    min_task_weight = config.get('min_task_weight', 0.2)
    max_task_weight = config.get('max_task_weight', 5.0)

    # Train with mode='multitask' for joint optimization
    trainer = MultiTaskTrainer(
        model,
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.01),
        risk_loss_fn=config.get('risk_loss', 'huber'),
        clip_grad=config.get('clip_grad', 1.0),
        patience=config.get('patience', 10),
        mode='multitask',  # Optimize both losses
        risk_weight=risk_weight,
        savings_weight=savings_weight,
        use_pcgrad=use_pcgrad,
        log_gradients=log_gradients,
        dynamic_weighting=dynamic_weighting,
        target_grad_ratio=target_grad_ratio,
        weight_update_rate=weight_update_rate,
        min_task_weight=min_task_weight,
        max_task_weight=max_task_weight,
    )

    metrics = trainer.fit(train_loader, val_loader, max_epochs=config.get('max_epochs', 100))

    return metrics, trainer.model
