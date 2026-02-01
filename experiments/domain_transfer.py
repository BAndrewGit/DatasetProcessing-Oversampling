# Domain Transfer Module
# Leverages GMSC dataset as auxiliary supervision for ADV risk modeling
# Architecture: Domain adapters -> Shared trunk -> Task-specific heads
#
# Key Design:
# - NO dataset merging (separate feature spaces)
# - Loss masking: GMSC does not update savings head
# - Evaluation on ADV real data only
# - Optional CORAL/MMD alignment

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    f1_score, accuracy_score, precision_score, recall_score
)
from scipy.stats import spearmanr
from typing import Tuple, Dict, Optional, List


# =============================================================================
# DOMAIN DATASETS
# =============================================================================

class ADVDataset(Dataset):
    """ADV dataset for risk and savings prediction."""

    def __init__(self, X: np.ndarray, y_risk: np.ndarray, y_savings: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y_risk = torch.FloatTensor(y_risk)
        self.y_savings = torch.FloatTensor(y_savings)
        self.domain = 0  # ADV domain identifier

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'risk_target': self.y_risk[idx],
            'savings_target': self.y_savings[idx],
            'domain': self.domain,
            'has_savings': True
        }


class GMSCDataset(Dataset):
    """GMSC dataset for auxiliary risk supervision."""

    def __init__(self, X: np.ndarray, y_dlq: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y_dlq = torch.FloatTensor(y_dlq)
        self.domain = 1  # GMSC domain identifier

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'risk_target': self.y_dlq[idx],  # Binary delinquency
            'savings_target': torch.tensor(0.0),  # Placeholder (not used)
            'domain': self.domain,
            'has_savings': False  # GMSC has no savings target
        }


class MixedDomainSampler(Sampler):
    """
    Sampler that creates mixed-domain batches with fixed ratio.
    E.g., 70% ADV / 30% GMSC samples per batch.
    """

    def __init__(self, adv_size: int, gmsc_size: int, batch_size: int,
                 adv_ratio: float = 0.7, seed: int = 42):
        self.adv_size = adv_size
        self.gmsc_size = gmsc_size
        self.batch_size = batch_size
        self.adv_ratio = adv_ratio
        self.seed = seed

        # Calculate samples per domain per batch
        self.adv_per_batch = int(batch_size * adv_ratio)
        self.gmsc_per_batch = batch_size - self.adv_per_batch

        # Total batches based on ADV dataset (primary)
        self.n_batches = adv_size // self.adv_per_batch

    def __iter__(self):
        rng = np.random.RandomState(self.seed)

        # Shuffle indices for each domain
        adv_indices = rng.permutation(self.adv_size)
        gmsc_indices = rng.permutation(self.gmsc_size)

        # Generate mixed batches
        for batch_idx in range(self.n_batches):
            # ADV samples
            adv_start = batch_idx * self.adv_per_batch
            adv_batch = adv_indices[adv_start:adv_start + self.adv_per_batch]

            # GMSC samples (cycle if needed)
            gmsc_start = (batch_idx * self.gmsc_per_batch) % self.gmsc_size
            gmsc_end = gmsc_start + self.gmsc_per_batch

            if gmsc_end <= self.gmsc_size:
                gmsc_batch = gmsc_indices[gmsc_start:gmsc_end]
            else:
                # Wrap around
                gmsc_batch = np.concatenate([
                    gmsc_indices[gmsc_start:],
                    gmsc_indices[:gmsc_end - self.gmsc_size]
                ])

            # Yield domain-tagged indices
            for idx in adv_batch:
                yield ('adv', int(idx))
            for idx in gmsc_batch:
                yield ('gmsc', int(idx))

    def __len__(self):
        return self.n_batches * self.batch_size


def mixed_collate_fn(batch_items: List[Tuple[str, int]],
                     adv_dataset: ADVDataset,
                     gmsc_dataset: GMSCDataset) -> Dict:
    """Custom collate function for mixed-domain batches."""
    adv_items = []
    gmsc_items = []

    for domain, idx in batch_items:
        if domain == 'adv':
            adv_items.append(adv_dataset[idx])
        else:
            gmsc_items.append(gmsc_dataset[idx])

    # Stack ADV batch
    if adv_items:
        adv_batch = {
            'features': torch.stack([item['features'] for item in adv_items]),
            'risk_target': torch.stack([item['risk_target'] for item in adv_items]),
            'savings_target': torch.stack([item['savings_target'] for item in adv_items]),
            'domain': torch.zeros(len(adv_items), dtype=torch.long),
            'has_savings': torch.ones(len(adv_items), dtype=torch.bool)
        }
    else:
        adv_batch = None

    # Stack GMSC batch
    if gmsc_items:
        gmsc_batch = {
            'features': torch.stack([item['features'] for item in gmsc_items]),
            'risk_target': torch.stack([item['risk_target'] for item in gmsc_items]),
            'savings_target': torch.zeros(len(gmsc_items)),
            'domain': torch.ones(len(gmsc_items), dtype=torch.long),
            'has_savings': torch.zeros(len(gmsc_items), dtype=torch.bool)
        }
    else:
        gmsc_batch = None

    return {'adv': adv_batch, 'gmsc': gmsc_batch}


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class DomainAdapter(nn.Module):
    """
    Domain-specific feature adapter.
    Maps domain features to shared latent dimension.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.adapter = nn.Sequential(*layers)

    def forward(self, x):
        return self.adapter(x)


class SharedTrunk(nn.Module):
    """
    Shared latent representation trunk.
    Processes adapted features from both domains.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x):
        return self.trunk(x)


class TaskHead(nn.Module):
    """Task-specific output head."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1,
                 dropout: float = 0.3, final_activation: Optional[str] = None):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        if final_activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class DomainTransferModel(nn.Module):
    """
    Domain Transfer Model for ADV + GMSC learning.

    Architecture:
        ADV features  -> ADV adapter  -┐
                                       ├-> Shared trunk -> Risk head (ADV regression)
        GMSC features -> GMSC adapter -┘                -> Risk head (GMSC classification)
                                                        -> Savings head (ADV only)
    """

    def __init__(self, adv_input_dim: int, gmsc_input_dim: int, config: Dict):
        super().__init__()

        dropout = config.get('dropout', 0.3)
        latent_dim = config.get('latent_dim', 32)

        # Domain adapters
        self.adv_adapter = DomainAdapter(
            input_dim=adv_input_dim,
            hidden_dims=config.get('adv_adapter_dims', [64]),
            output_dim=latent_dim,
            dropout=dropout
        )

        self.gmsc_adapter = DomainAdapter(
            input_dim=gmsc_input_dim,
            hidden_dims=config.get('gmsc_adapter_dims', [32]),
            output_dim=latent_dim,
            dropout=dropout
        )

        # Shared trunk
        self.shared_trunk = SharedTrunk(
            input_dim=latent_dim,
            hidden_dims=config.get('shared_trunk_dims', [64, 32]),
            dropout=dropout
        )

        trunk_output_dim = self.shared_trunk.output_dim

        # Task heads
        # Risk head for ADV (regression)
        self.adv_risk_head = TaskHead(
            input_dim=trunk_output_dim,
            hidden_dims=config.get('risk_head_dims', [16]),
            output_dim=1,
            dropout=dropout,
            final_activation=None  # No activation for regression
        )

        # Risk head for GMSC (classification)
        self.gmsc_risk_head = TaskHead(
            input_dim=trunk_output_dim,
            hidden_dims=config.get('risk_head_dims', [16]),
            output_dim=1,
            dropout=dropout,
            final_activation=None  # Use BCEWithLogitsLoss
        )

        # Savings head (ADV only)
        self.savings_head = TaskHead(
            input_dim=trunk_output_dim,
            hidden_dims=config.get('savings_head_dims', [16]),
            output_dim=1,
            dropout=dropout,
            final_activation=None  # Use BCEWithLogitsLoss
        )

    def forward_adv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for ADV domain."""
        adapted = self.adv_adapter(x)
        shared = self.shared_trunk(adapted)
        risk_out = self.adv_risk_head(shared)
        savings_out = self.savings_head(shared)
        return risk_out, savings_out, shared

    def forward_gmsc(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for GMSC domain."""
        adapted = self.gmsc_adapter(x)
        shared = self.shared_trunk(adapted)
        risk_out = self.gmsc_risk_head(shared)
        return risk_out, shared

    def forward(self, x: torch.Tensor, domain: int) -> Dict[str, torch.Tensor]:
        """
        Generic forward pass with domain specification.

        Args:
            x: Input features
            domain: 0 for ADV, 1 for GMSC
        """
        if domain == 0:
            risk, savings, shared = self.forward_adv(x)
            return {'risk': risk, 'savings': savings, 'shared': shared}
        else:
            risk, shared = self.forward_gmsc(x)
            return {'risk': risk, 'shared': shared}


# =============================================================================
# DOMAIN ALIGNMENT LOSSES
# =============================================================================

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CORAL (CORrelation ALignment) loss.
    Aligns second-order statistics between domains.
    """
    d = source.size(1)

    # Source covariance
    source_mean = source.mean(0, keepdim=True)
    source_centered = source - source_mean
    source_cov = source_centered.t() @ source_centered / (source.size(0) - 1)

    # Target covariance
    target_mean = target.mean(0, keepdim=True)
    target_centered = target - target_mean
    target_cov = target_centered.t() @ target_centered / (target.size(0) - 1)

    # Frobenius norm of covariance difference
    loss = (source_cov - target_cov).pow(2).sum() / (4 * d * d)

    return loss


def mmd_loss(source: torch.Tensor, target: torch.Tensor,
             kernel: str = 'rbf', bandwidth: float = 1.0) -> torch.Tensor:
    """
    Maximum Mean Discrepancy (MMD) loss.
    Measures distance between distributions in RKHS.
    """
    def rbf_kernel(x, y, bandwidth):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        sq_dist = (diff ** 2).sum(-1)
        return torch.exp(-sq_dist / (2 * bandwidth ** 2))

    if kernel == 'rbf':
        k_ss = rbf_kernel(source, source, bandwidth)
        k_tt = rbf_kernel(target, target, bandwidth)
        k_st = rbf_kernel(source, target, bandwidth)

        loss = k_ss.mean() + k_tt.mean() - 2 * k_st.mean()
    else:
        # Linear kernel
        loss = ((source.mean(0) - target.mean(0)) ** 2).sum()

    return loss


# =============================================================================
# TRAINER
# =============================================================================

class DomainTransferTrainer:
    """
    Trainer for domain transfer model.

    Handles:
    - Mixed-domain batches
    - Loss masking (GMSC doesn't update savings head)
    - Optional domain alignment
    - ADV-only evaluation
    """

    def __init__(self, model: DomainTransferModel, config: Dict, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('lr', 0.001),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # Loss functions
        risk_loss_type = config.get('risk_loss', 'huber')
        if risk_loss_type == 'huber':
            self.adv_risk_loss_fn = nn.HuberLoss()
        else:
            self.adv_risk_loss_fn = nn.MSELoss()

        self.gmsc_risk_loss_fn = nn.BCEWithLogitsLoss()
        self.savings_loss_fn = nn.BCEWithLogitsLoss()

        # Loss weights
        self.adv_risk_weight = config.get('adv_risk_weight', 1.0)
        self.gmsc_risk_weight = config.get('gmsc_risk_weight', 0.5)
        self.savings_weight = config.get('savings_weight', 1.0)

        # Domain alignment
        self.alignment_enabled = config.get('alignment_enabled', False)
        self.alignment_method = config.get('alignment_method', 'coral')
        self.alignment_weight = config.get('alignment_weight', 0.1)

        # Gradient clipping
        self.clip_grad = config.get('clip_grad', 1.0)

        # Early stopping
        self.patience = config.get('patience', 15)

    def train_step(self, adv_batch: Optional[Dict], gmsc_batch: Optional[Dict]) -> Dict:
        """Single training step with mixed-domain batch."""
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        loss_components = {}

        adv_shared = None
        gmsc_shared = None

        # Process ADV batch
        if adv_batch is not None:
            adv_features = adv_batch['features'].to(self.device)
            adv_risk_target = adv_batch['risk_target'].to(self.device)
            adv_savings_target = adv_batch['savings_target'].to(self.device)

            # Forward pass
            adv_output = self.model.forward(adv_features, domain=0)

            # Risk loss (regression)
            adv_risk_loss = self.adv_risk_loss_fn(adv_output['risk'], adv_risk_target)
            total_loss += self.adv_risk_weight * adv_risk_loss
            loss_components['adv_risk'] = adv_risk_loss.item()

            # Savings loss
            savings_loss = self.savings_loss_fn(adv_output['savings'], adv_savings_target)
            total_loss += self.savings_weight * savings_loss
            loss_components['savings'] = savings_loss.item()

            adv_shared = adv_output['shared']

        # Process GMSC batch
        if gmsc_batch is not None:
            gmsc_features = gmsc_batch['features'].to(self.device)
            gmsc_risk_target = gmsc_batch['risk_target'].to(self.device)

            # Forward pass (note: savings head is NOT updated)
            gmsc_output = self.model.forward(gmsc_features, domain=1)

            # Risk loss (classification)
            gmsc_risk_loss = self.gmsc_risk_loss_fn(gmsc_output['risk'], gmsc_risk_target)
            total_loss += self.gmsc_risk_weight * gmsc_risk_loss
            loss_components['gmsc_risk'] = gmsc_risk_loss.item()

            gmsc_shared = gmsc_output['shared']

        # Domain alignment loss
        if self.alignment_enabled and adv_shared is not None and gmsc_shared is not None:
            if self.alignment_method == 'coral':
                align_loss = coral_loss(adv_shared, gmsc_shared)
            elif self.alignment_method == 'mmd':
                align_loss = mmd_loss(adv_shared, gmsc_shared)
            else:
                align_loss = torch.tensor(0.0)

            total_loss += self.alignment_weight * align_loss
            loss_components['alignment'] = align_loss.item()

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

        self.optimizer.step()

        loss_components['total'] = total_loss.item()
        return loss_components

    def evaluate_adv(self, adv_loader: DataLoader) -> Dict:
        """
        Evaluate model on ADV data only.

        Returns metrics for both risk (regression) and savings (classification).
        """
        self.model.eval()

        all_risk_preds = []
        all_risk_true = []
        all_savings_preds = []
        all_savings_true = []

        with torch.no_grad():
            for batch in adv_loader:
                features = batch['features'].to(self.device)
                risk_target = batch['risk_target'].to(self.device)
                savings_target = batch['savings_target'].to(self.device)

                output = self.model.forward(features, domain=0)

                # Risk predictions
                all_risk_preds.extend(output['risk'].cpu().numpy())
                all_risk_true.extend(risk_target.cpu().numpy())

                # Savings predictions
                savings_probs = torch.sigmoid(output['savings'])
                all_savings_preds.extend((savings_probs > 0.5).cpu().numpy().astype(int))
                all_savings_true.extend(savings_target.cpu().numpy().astype(int))

        # Compute metrics
        metrics = self._compute_metrics(
            np.array(all_risk_true),
            np.array(all_risk_preds),
            np.array(all_savings_true),
            np.array(all_savings_preds)
        )

        return metrics

    def _compute_metrics(self, y_risk_true, y_risk_pred,
                         y_savings_true, y_savings_pred) -> Dict:
        """Compute all evaluation metrics."""
        metrics = {}

        # Risk metrics (regression)
        metrics['risk_mae'] = mean_absolute_error(y_risk_true, y_risk_pred)
        metrics['risk_rmse'] = np.sqrt(mean_squared_error(y_risk_true, y_risk_pred))

        # Handle edge cases for Spearman
        if np.std(y_risk_true) > 0 and np.std(y_risk_pred) > 0:
            metrics['risk_spearman'] = spearmanr(y_risk_true, y_risk_pred)[0]
        else:
            metrics['risk_spearman'] = 0.0

        # R² score
        ss_res = np.sum((y_risk_true - y_risk_pred) ** 2)
        ss_tot = np.sum((y_risk_true - np.mean(y_risk_true)) ** 2)
        metrics['risk_r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Savings metrics (classification)
        metrics['savings_macro_f1'] = f1_score(
            y_savings_true, y_savings_pred, average='macro', zero_division=0
        )
        metrics['savings_accuracy'] = accuracy_score(y_savings_true, y_savings_pred)
        metrics['savings_precision'] = precision_score(
            y_savings_true, y_savings_pred, average='macro', zero_division=0
        )
        metrics['savings_recall'] = recall_score(
            y_savings_true, y_savings_pred, average='macro', zero_division=0
        )

        return metrics

    def fit(self, train_batches: List, adv_val_loader: DataLoader,
            max_epochs: int = 100) -> Dict:
        """
        Train model with early stopping based on ADV validation loss.

        Args:
            train_batches: List of (adv_batch, gmsc_batch) tuples
            adv_val_loader: DataLoader for ADV validation data
            max_epochs: Maximum training epochs

        Returns:
            best_metrics: Best validation metrics
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_metrics = None
        best_state = None

        for epoch in range(max_epochs):
            # Training
            epoch_losses = []
            for adv_batch, gmsc_batch in train_batches:
                loss_components = self.train_step(adv_batch, gmsc_batch)
                epoch_losses.append(loss_components['total'])

            avg_train_loss = np.mean(epoch_losses)

            # Validation (ADV only)
            val_metrics = self.evaluate_adv(adv_val_loader)
            val_loss = val_metrics['risk_mae']  # Use MAE as validation criterion

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_metrics


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_adv_only(X_train: np.ndarray, y_risk_train: np.ndarray, y_savings_train: np.ndarray,
                   X_val: np.ndarray, y_risk_val: np.ndarray, y_savings_val: np.ndarray,
                   config: Dict, seed: int) -> Dict:
    """
    Train ADV-only baseline (no GMSC transfer).
    Uses same architecture but without GMSC adapter/data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model (GMSC adapter won't be used)
    model = DomainTransferModel(
        adv_input_dim=X_train.shape[1],
        gmsc_input_dim=10,  # Placeholder
        config=config
    )

    # Create data loaders
    train_dataset = ADVDataset(X_train, y_risk_train, y_savings_train)
    val_dataset = ADVDataset(X_val, y_risk_val, y_savings_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 64)
    )

    # Disable domain alignment for ADV-only
    trainer_config = config.copy()
    trainer_config['alignment_enabled'] = False

    # Create trainer
    trainer = DomainTransferTrainer(model, trainer_config, device)

    # Prepare training batches (ADV only)
    train_batches = []
    for batch in train_loader:
        train_batches.append((batch, None))

    # Train
    metrics = trainer.fit(
        train_batches,
        val_loader,
        max_epochs=config.get('max_epochs', 100)
    )

    # Return metrics and trained model for downstream saving
    return metrics, trainer.model


def train_with_gmsc_transfer(
    adv_X_train: np.ndarray, adv_y_risk_train: np.ndarray, adv_y_savings_train: np.ndarray,
    adv_X_val: np.ndarray, adv_y_risk_val: np.ndarray, adv_y_savings_val: np.ndarray,
    gmsc_X_train: np.ndarray, gmsc_y_train: np.ndarray,
    config: Dict, seed: int
) -> Dict:
    """
    Train with GMSC auxiliary supervision.
    Mixed-domain batches with loss masking.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = DomainTransferModel(
        adv_input_dim=adv_X_train.shape[1],
        gmsc_input_dim=gmsc_X_train.shape[1],
        config=config
    )

    # Create datasets
    adv_train_dataset = ADVDataset(adv_X_train, adv_y_risk_train, adv_y_savings_train)
    adv_val_dataset = ADVDataset(adv_X_val, adv_y_risk_val, adv_y_savings_val)
    gmsc_train_dataset = GMSCDataset(gmsc_X_train, gmsc_y_train)

    # Create validation loader (ADV only)
    val_loader = DataLoader(
        adv_val_dataset,
        batch_size=config.get('batch_size', 64)
    )

    # Create mixed-domain training batches
    batch_size = config.get('batch_size', 64)
    adv_ratio = config.get('adv_ratio', 0.7)
    adv_per_batch = int(batch_size * adv_ratio)
    gmsc_per_batch = batch_size - adv_per_batch

    # Generate training batches
    rng = np.random.RandomState(seed)
    n_batches = len(adv_train_dataset) // adv_per_batch

    adv_indices = rng.permutation(len(adv_train_dataset))
    gmsc_indices = rng.permutation(len(gmsc_train_dataset))

    train_batches = []
    for batch_idx in range(n_batches):
        # ADV batch
        adv_start = batch_idx * adv_per_batch
        adv_batch_indices = adv_indices[adv_start:adv_start + adv_per_batch]

        adv_batch = {
            'features': torch.stack([adv_train_dataset[i]['features'] for i in adv_batch_indices]),
            'risk_target': torch.stack([adv_train_dataset[i]['risk_target'] for i in adv_batch_indices]),
            'savings_target': torch.stack([adv_train_dataset[i]['savings_target'] for i in adv_batch_indices])
        }

        # GMSC batch (cycle if needed)
        gmsc_start = (batch_idx * gmsc_per_batch) % len(gmsc_train_dataset)
        gmsc_batch_indices = []
        for i in range(gmsc_per_batch):
            idx = (gmsc_start + i) % len(gmsc_train_dataset)
            gmsc_batch_indices.append(gmsc_indices[idx])

        gmsc_batch = {
            'features': torch.stack([gmsc_train_dataset[i]['features'] for i in gmsc_batch_indices]),
            'risk_target': torch.stack([gmsc_train_dataset[i]['risk_target'] for i in gmsc_batch_indices])
        }

        train_batches.append((adv_batch, gmsc_batch))

    # Enable domain alignment
    trainer_config = config.copy()
    trainer_config['alignment_enabled'] = config.get('alignment_enabled', True)
    trainer_config['alignment_method'] = config.get('alignment_method', 'coral')
    trainer_config['alignment_weight'] = config.get('alignment_weight', 0.1)

    # Create trainer
    trainer = DomainTransferTrainer(model, trainer_config, device)

    # Train
    metrics = trainer.fit(
        train_batches,
        val_loader,
        max_epochs=config.get('max_epochs', 100)
    )

    return metrics, trainer.model
