import os
import json
import argparse
import random
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, accuracy_score
)
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

FORBIDDEN_COLUMNS = ["Behavior_Risk_Level", "Confidence", "Outlier", "Cluster", "Auto_Label"]


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SharedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.encoder(x)


class RiskHead(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


class SavingsHead(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, hidden_dim, dropout)
        self.risk_head = RiskHead(hidden_dim)
        self.savings_head = SavingsHead(hidden_dim)

    def forward(self, x):
        shared = self.encoder(x)
        risk_out = self.risk_head(shared)
        savings_out = self.savings_head(shared)
        return risk_out, savings_out


class SingleTaskRiskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)


class SingleTaskSavingsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def load_and_prepare_data(dataset_path):
    if dataset_path is None:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        print("Select the dataset file...")
        dataset_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            parent=root
        )
        root.destroy()
        if not dataset_path:
            raise ValueError("No dataset selected")

    print(f"Loading: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Get targets
    if "Risk_Score" not in df.columns:
        raise ValueError("Risk_Score column not found")
    if "Save_Money_Yes" not in df.columns:
        raise ValueError("Save_Money_Yes column not found")

    y_risk = df["Risk_Score"].values.astype(np.float32)
    y_savings = df["Save_Money_Yes"].values.astype(np.float32)

    # Build features - exclude targets and forbidden columns
    exclude = FORBIDDEN_COLUMNS + ["Risk_Score", "Save_Money_Yes", "Save_Money_No"]
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].values.astype(np.float32)

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Risk_Score range: [{y_risk.min():.3f}, {y_risk.max():.3f}]")
    print(f"Save_Money distribution: {np.bincount(y_savings.astype(int))}")

    return X, y_risk, y_savings, feature_cols


def train_multitask(model, X_train, y_risk_train, y_sav_train, X_val, y_risk_val, y_sav_val,
                    epochs=200, lr=1e-3, weight_decay=1e-4, batch_size=32, patience=15,
                    risk_weight=1.0, savings_weight=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    early_stop = EarlyStopping(patience=patience)

    # Create dataloaders
    train_ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_risk_train),
        torch.tensor(y_sav_train)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val).to(device)
    y_risk_val_t = torch.tensor(y_risk_val).to(device)
    y_sav_val_t = torch.tensor(y_sav_val).to(device)

    best_model_state = None
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_risk, batch_sav in train_loader:
            batch_x = batch_x.to(device)
            batch_risk = batch_risk.to(device)
            batch_sav = batch_sav.to(device)

            optimizer.zero_grad()
            risk_pred, sav_pred = model(batch_x)

            loss_risk = mse_loss(risk_pred, batch_risk)
            loss_sav = bce_loss(sav_pred, batch_sav)
            loss = risk_weight * loss_risk + savings_weight * loss_sav

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            risk_pred_val, sav_pred_val = model(X_val_t)
            val_loss_risk = mse_loss(risk_pred_val, y_risk_val_t)
            val_loss_sav = bce_loss(sav_pred_val, y_sav_val_t)
            val_loss = risk_weight * val_loss_risk + savings_weight * val_loss_sav

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()

        if early_stop(val_loss.item()):
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def train_single_risk(model, X_train, y_train, X_val, y_val,
                      epochs=200, lr=1e-3, weight_decay=1e-4, batch_size=32, patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()
    early_stop = EarlyStopping(patience=patience)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val).to(device)
    y_val_t = torch.tensor(y_val).to(device)

    best_model_state = None
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = mse_loss(pred, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = mse_loss(val_pred, y_val_t)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()

        if early_stop(val_loss.item()):
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def train_single_savings(model, X_train, y_train, X_val, y_val,
                         epochs=200, lr=1e-3, weight_decay=1e-4, batch_size=32, patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce_loss = nn.BCELoss()
    early_stop = EarlyStopping(patience=patience)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val).to(device)
    y_val_t = torch.tensor(y_val).to(device)

    best_model_state = None
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = bce_loss(pred, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = bce_loss(val_pred, y_val_t)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()

        if early_stop(val_loss.item()):
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_risk(model, X, y, is_multitask=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X).to(device)
        if is_multitask:
            pred, _ = model(X_t)
        else:
            pred = model(X_t)
        pred = pred.cpu().numpy()

    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    spearman = spearmanr(y, pred)[0]
    r2 = r2_score(y, pred)

    return {'mae': mae, 'rmse': rmse, 'spearman': spearman, 'r2': r2}


def evaluate_savings(model, X, y, is_multitask=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X).to(device)
        if is_multitask:
            _, pred_proba = model(X_t)
        else:
            pred_proba = model(X_t)
        pred_proba = pred_proba.cpu().numpy()

    pred = (pred_proba >= 0.5).astype(int)

    macro_f1 = f1_score(y, pred, average='macro')
    accuracy = accuracy_score(y, pred)

    return {'macro_f1': macro_f1, 'accuracy': accuracy}


def run_ablation(dataset_path, seed=42, n_splits=5, n_repeats=5, hidden_dim=64, dropout=0.3):
    set_seeds(seed)

    print("=" * 70)
    print("MULTI-TASK ABLATION STUDY")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"CV: {n_splits} folds × {n_repeats} repeats")
    print("NO oversampling, NO synthetic data (ADV only)")
    print("=" * 70)

    # Load data
    X, y_risk, y_savings, feature_cols = load_and_prepare_data(dataset_path)
    input_dim = X.shape[1]

    # Results storage
    results = {
        'single_risk': {'mae': [], 'rmse': [], 'spearman': [], 'r2': []},
        'single_savings': {'macro_f1': [], 'accuracy': []},
        'multi_risk': {'mae': [], 'rmse': [], 'spearman': [], 'r2': []},
        'multi_savings': {'macro_f1': [], 'accuracy': []}
    }

    # CV loop
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    fold_idx = 0
    for train_idx, val_idx in cv.split(X):
        fold_idx += 1
        print(f"\nFold {fold_idx}/{n_splits * n_repeats}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_risk_train, y_risk_val = y_risk[train_idx], y_risk[val_idx]
        y_sav_train, y_sav_val = y_savings[train_idx], y_savings[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)

        # 1. Single-task Risk
        model_risk = SingleTaskRiskModel(input_dim, hidden_dim, dropout)
        model_risk = train_single_risk(model_risk, X_train_s, y_risk_train, X_val_s, y_risk_val)
        risk_metrics = evaluate_risk(model_risk, X_val_s, y_risk_val, is_multitask=False)
        for k, v in risk_metrics.items():
            results['single_risk'][k].append(v)

        # 2. Single-task Savings
        model_sav = SingleTaskSavingsModel(input_dim, hidden_dim, dropout)
        model_sav = train_single_savings(model_sav, X_train_s, y_sav_train, X_val_s, y_sav_val)
        sav_metrics = evaluate_savings(model_sav, X_val_s, y_sav_val, is_multitask=False)
        for k, v in sav_metrics.items():
            results['single_savings'][k].append(v)

        # 3. Multi-task
        model_multi = MultiTaskModel(input_dim, hidden_dim, dropout)
        model_multi = train_multitask(
            model_multi, X_train_s, y_risk_train, y_sav_train, X_val_s, y_risk_val, y_sav_val
        )

        multi_risk_metrics = evaluate_risk(model_multi, X_val_s, y_risk_val, is_multitask=True)
        for k, v in multi_risk_metrics.items():
            results['multi_risk'][k].append(v)

        multi_sav_metrics = evaluate_savings(model_multi, X_val_s, y_sav_val, is_multitask=True)
        for k, v in multi_sav_metrics.items():
            results['multi_savings'][k].append(v)

        print(f"  Single Risk MAE: {risk_metrics['mae']:.4f}, Multi Risk MAE: {multi_risk_metrics['mae']:.4f}")
        print(f"  Single Sav F1:   {sav_metrics['macro_f1']:.4f}, Multi Sav F1:   {multi_sav_metrics['macro_f1']:.4f}")

    # Compute summary statistics
    summary = {}
    for model_name, metrics in results.items():
        summary[model_name] = {}
        for metric_name, values in metrics.items():
            summary[model_name][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values]
            }

    return summary, results


def print_ablation_summary(summary):
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- RISK PREDICTION (Risk_Score) ---")
    print(f"{'Model':<20} {'MAE':>12} {'RMSE':>12} {'Spearman':>12} {'R²':>12}")
    print("-" * 70)

    for model in ['single_risk', 'multi_risk']:
        m = summary[model]
        label = "Single-Task" if "single" in model else "Multi-Task"
        print(f"{label:<20} {m['mae']['mean']:>8.4f}±{m['mae']['std']:.4f} "
              f"{m['rmse']['mean']:>8.4f}±{m['rmse']['std']:.4f} "
              f"{m['spearman']['mean']:>8.4f}±{m['spearman']['std']:.4f} "
              f"{m['r2']['mean']:>8.4f}±{m['r2']['std']:.4f}")

    print("\n--- SAVINGS PREDICTION (Save_Money_Yes) ---")
    print(f"{'Model':<20} {'Macro-F1':>15} {'Accuracy':>15}")
    print("-" * 50)

    for model in ['single_savings', 'multi_savings']:
        m = summary[model]
        label = "Single-Task" if "single" in model else "Multi-Task"
        print(f"{label:<20} {m['macro_f1']['mean']:>10.4f}±{m['macro_f1']['std']:.4f} "
              f"{m['accuracy']['mean']:>10.4f}±{m['accuracy']['std']:.4f}")

    # Decision
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    risk_degrades = summary['multi_risk']['mae']['mean'] > summary['single_risk']['mae']['mean'] * 1.05
    sav_degrades = summary['multi_savings']['macro_f1']['mean'] < summary['single_savings']['macro_f1']['mean'] * 0.95

    risk_diff = (summary['multi_risk']['mae']['mean'] - summary['single_risk']['mae']['mean']) / summary['single_risk']['mae']['mean'] * 100
    sav_diff = (summary['multi_savings']['macro_f1']['mean'] - summary['single_savings']['macro_f1']['mean']) / summary['single_savings']['macro_f1']['mean'] * 100

    print(f"Risk MAE change:    {risk_diff:+.2f}% ({'worse' if risk_diff > 0 else 'better'})")
    print(f"Savings F1 change:  {sav_diff:+.2f}% ({'worse' if sav_diff < 0 else 'better'})")

    if risk_degrades or sav_degrades:
        print("\n⚠️  MULTI-TASK DEGRADES PERFORMANCE")
        print("   Recommendation: Use SINGLE-TASK models")
        print("   (Multi-task reported as negative ablation)")
        recommendation = "single_task"
    else:
        print("\n✅ MULTI-TASK ACCEPTABLE")
        print("   Recommendation: Use MULTI-TASK model")
        recommendation = "multi_task"

    return recommendation


def save_ablation_results(summary, recommendation, output_dir="runs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"ablation_multitask_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save summary JSON
    results_data = {
        'summary': summary,
        'recommendation': recommendation,
        'timestamp': timestamp
    }

    with open(os.path.join(run_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results_data, f, indent=2)

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Risk comparison
    ax = axes[0]
    models = ['Single-Task', 'Multi-Task']
    mae_means = [summary['single_risk']['mae']['mean'], summary['multi_risk']['mae']['mean']]
    mae_stds = [summary['single_risk']['mae']['std'], summary['multi_risk']['mae']['std']]

    bars = ax.bar(models, mae_means, yerr=mae_stds, capsize=5, color=['steelblue', 'coral'])
    ax.set_ylabel('MAE (lower is better)')
    ax.set_title('Risk Prediction (Risk_Score)')
    ax.set_ylim(0, max(mae_means) * 1.3)
    for bar, mean in zip(bars, mae_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.4f}', ha='center', va='bottom')

    # Savings comparison
    ax = axes[1]
    f1_means = [summary['single_savings']['macro_f1']['mean'], summary['multi_savings']['macro_f1']['mean']]
    f1_stds = [summary['single_savings']['macro_f1']['std'], summary['multi_savings']['macro_f1']['std']]

    bars = ax.bar(models, f1_means, yerr=f1_stds, capsize=5, color=['steelblue', 'coral'])
    ax.set_ylabel('Macro-F1 (higher is better)')
    ax.set_title('Savings Prediction (Save_Money)')
    ax.set_ylim(0, 1.0)
    for bar, mean in zip(bars, f1_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.4f}', ha='center', va='bottom')

    plt.suptitle(f'Ablation Study: Single-Task vs Multi-Task\nRecommendation: {recommendation.upper()}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'ablation_comparison.png'), dpi=150)
    plt.close()

    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Risk MAE distribution
    ax = axes[0, 0]
    ax.hist(summary['single_risk']['mae']['values'], bins=15, alpha=0.6, label='Single-Task', color='steelblue')
    ax.hist(summary['multi_risk']['mae']['values'], bins=15, alpha=0.6, label='Multi-Task', color='coral')
    ax.set_xlabel('MAE')
    ax.set_ylabel('Frequency')
    ax.set_title('Risk MAE Distribution')
    ax.legend()

    # Risk Spearman distribution
    ax = axes[0, 1]
    ax.hist(summary['single_risk']['spearman']['values'], bins=15, alpha=0.6, label='Single-Task', color='steelblue')
    ax.hist(summary['multi_risk']['spearman']['values'], bins=15, alpha=0.6, label='Multi-Task', color='coral')
    ax.set_xlabel('Spearman Correlation')
    ax.set_ylabel('Frequency')
    ax.set_title('Risk Spearman Distribution')
    ax.legend()

    # Savings F1 distribution
    ax = axes[1, 0]
    ax.hist(summary['single_savings']['macro_f1']['values'], bins=15, alpha=0.6, label='Single-Task', color='steelblue')
    ax.hist(summary['multi_savings']['macro_f1']['values'], bins=15, alpha=0.6, label='Multi-Task', color='coral')
    ax.set_xlabel('Macro-F1')
    ax.set_ylabel('Frequency')
    ax.set_title('Savings F1 Distribution')
    ax.legend()

    # Savings Accuracy distribution
    ax = axes[1, 1]
    ax.hist(summary['single_savings']['accuracy']['values'], bins=15, alpha=0.6, label='Single-Task', color='steelblue')
    ax.hist(summary['multi_savings']['accuracy']['values'], bins=15, alpha=0.6, label='Multi-Task', color='coral')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Frequency')
    ax.set_title('Savings Accuracy Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'cv_distributions.png'), dpi=150)
    plt.close()

    print(f"\nResults saved to: {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='Multi-task ablation study: Risk + Savings')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                       help='Path to dataset CSV')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--n-splits', type=int, default=5,
                       help='Number of CV splits')
    parser.add_argument('--n-repeats', type=int, default=5,
                       help='Number of CV repeats')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension size')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    args = parser.parse_args()

    summary, results = run_ablation(
        args.dataset,
        seed=args.seed,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )

    recommendation = print_ablation_summary(summary)
    save_ablation_results(summary, recommendation)


if __name__ == "__main__":
    main()

