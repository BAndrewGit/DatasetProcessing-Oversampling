import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from config import TARGET, DPI

sns.set_theme(style="whitegrid")

#Univariate analysis
def generate_univariate_plots(df, group_map, plots_dir):
    print("Starting Univariate Analysis...")

    # Helper to find groups of one-hot encoded columns
    def find_one_hot_groups(cols):
        groups = {}
        for col in cols:
            # Check for pattern like "Feature_Value"
            parts = col.rsplit('_', 1)
            if len(parts) == 2:
                prefix, suffix = parts
                # Heuristic: if we have multiple columns with same prefix in the list
                if sum(1 for c in cols if c.startswith(prefix + '_')) > 1:
                    if prefix not in groups:
                        groups[prefix] = []
                    groups[prefix].append(col)
        return groups

    for group, cols in group_map.items():
        # Identify one-hot groups within this category
        one_hot_groups = find_one_hot_groups(cols)
        processed_cols = set()

        # Plot one-hot groups as single bar charts
        for prefix, group_cols in one_hot_groups.items():
            # Calculate frequencies (percentages)
            means = df[group_cols].mean().sort_values(ascending=False)
            # Clean labels: remove prefix
            labels = [c.replace(prefix + '_', '') for c in means.index]

            plt.figure(figsize=(10, 6))
            sns.barplot(x=means.values, y=labels, palette="viridis")
            plt.title(f"Distribution of {prefix} (Frequency)", fontsize=14)
            plt.xlabel("Frequency (0-1)")
            plt.xlim(0, 1)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "univariate", f"{group}_{prefix}_summary.png"),
                        dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f"- {group}/{prefix} summary saved")

            processed_cols.update(group_cols)

        # Plot remaining individual columns
        for col in cols:
            if col in processed_cols:
                continue

            if col not in df.columns or df[col].nunique() <= 1:
                continue

            fig = plt.figure(figsize=(12, 5))
            fig.suptitle(f"{group} — {col}", fontsize=14, y=0.98)

            plt.subplot(1, 2, 1)
            # Check if binary/categorical
            if df[col].nunique() <= 10:
                 # Bar plot for categorical/low cardinality
                 counts = df[col].value_counts(normalize=True).sort_index()
                 sns.barplot(x=counts.index, y=counts.values, color='steelblue')
                 plt.ylabel("Frequency")
                 plt.title("Distribution (Categorical)", pad=10)
            else:
                 # Hist/KDE for continuous
                 sns.histplot(df[col].dropna(), kde=True, bins=30, color='steelblue', stat="density")
                 plt.title("Distribution (Continuous)", pad=10)

            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col], color='salmon', showfliers=True,
                        flierprops={"marker": "o", "color": "black", "alpha": 0.6})
            plt.title("Boxplot", pad=10)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(plots_dir, "univariate", f"{group}_{col}.png"),
                        dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f"- {group}/{col} saved")


#Bivariate analysis
def generate_bivariate_plots(df, group_map, plots_dir):
    print("Starting Bivariate Analysis...")
    for group, cols in group_map.items():
        numeric_cols = [c for c in cols if c in df.select_dtypes(include=np.number).columns]
        if len(numeric_cols) < 2:
            continue

        corr = df[numeric_cols].corr()
        high_pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
            .head(3)
        )

        # Heatmap
        fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols) * 0.8),
                                        max(8, len(numeric_cols) * 0.7)))
        short_names = [c.replace("Impulse_Buying_", "IB_").replace("Financial_", "Fin_")[:20]
                       for c in numeric_cols]
        corr_renamed = corr.copy()
        corr_renamed.columns = short_names
        corr_renamed.index = short_names

        sns.heatmap(corr_renamed, annot=True, cmap="coolwarm", fmt=".2f",
                    annot_kws={"size": 8}, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f"Correlation Heatmap — {group}", fontsize=14, pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "bivariate", f"{group}_heatmap.png"),
                    dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"- {group} heatmap saved")

        # Scatter plots pentru top perechi
        for (f1, f2), val in high_pairs.items():
            fig = plt.figure(figsize=(7, 6))
            sns.regplot(x=df[f1], y=df[f2], scatter_kws={"alpha": 0.6},
                        line_kws={"color": "red"})
            plt.title(f"{group}: {f1} vs {f2}\n(r={val:.2f})", fontsize=12, pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "bivariate", f"{group}_{f1}_vs_{f2}.png"),
                        dpi=DPI, bbox_inches='tight')
            plt.close()

#Target variable analysis
def generate_target_plots(df, plots_dir):
    print("Starting Analysis vs Target...")

    # Plot Target Distribution
    if TARGET in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[TARGET], kde=True, color='purple', stat="density")
        plt.title(f"Distribution of Target: {TARGET}", fontsize=14)
        plt.xlabel(TARGET)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "vs_target", f"Target_Distribution.png"),
                    dpi=DPI, bbox_inches='tight')
        plt.close()

    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != TARGET]

    corr_with_target = df[numeric_cols].corrwith(df[TARGET]).abs().sort_values(ascending=False)
    top5 = corr_with_target.head(5)

    # Scatter plots
    for col in top5.index:
        fig = plt.figure(figsize=(7, 6))
        sns.regplot(x=df[col], y=df[TARGET], scatter_kws={"alpha": 0.6},
                    line_kws={"color": "red"})
        plt.title(f"{col} vs {TARGET}\n(corr={corr_with_target[col]:.2f})",
                  fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "vs_target", f"{col}_vs_{TARGET}.png"),
                    dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"- {col} vs {TARGET} saved")

    # Feature importance barplot
    fig = plt.figure(figsize=(9, 6))
    ax = sns.barplot(x=top5.values, y=top5.index, orient="h", hue=top5.index,
                     palette="Blues_r", legend=False)
    ax.set_title("Top 5 Features correlated with Risk_Score", fontsize=14, pad=15)
    ax.set_xlabel("Absolute Correlation", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "vs_target", "feature_importance.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print("- Feature importance saved")
