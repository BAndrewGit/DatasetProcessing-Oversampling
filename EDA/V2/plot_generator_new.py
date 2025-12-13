import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import TARGET, DPI

sns.set_theme(style="whitegrid")


# Removed binary group helper functions (is_binary_group_column, get_group_prefix, plot_binary_group_frequencies)
# to enforce the strategy of aggregating features upstream before plotting.

def generate_univariate_plots(df, group_map, plots_dir):
    """
    Generates univariate plots for all columns in group_map.
    Handles Numeric (Hist+Box) and Categorical (Bar) data automatically.
    """
    print("Starting Univariate Analysis...")
    plotted_columns = set()

    for group, cols in group_map.items():
        for col in cols:
            if col not in df.columns or col in plotted_columns:
                continue

            # Skip columns with no data
            if df[col].dropna().empty:
                continue

            fig = plt.figure(figsize=(12, 5))
            fig.suptitle(f"{group} — {col}", fontsize=14, y=0.98)

            # Check data type
            is_numeric = pd.api.types.is_numeric_dtype(df[col])

            if is_numeric:
                # Numeric: Histogram + Boxplot
                plt.subplot(1, 2, 1)
                sns.histplot(df[col].dropna(), kde=True, bins=30, color='steelblue')
                plt.title("Distribution", pad=10)
                plt.xlabel(col)

                plt.subplot(1, 2, 2)
                sns.boxplot(x=df[col], color='salmon', showfliers=True,
                            flierprops={"marker": "o", "color": "black", "alpha": 0.6})
                plt.title("Boxplot", pad=10)
                plt.xlabel(col)
            else:
                # Categorical/Ordinal: Horizontal Bar Chart
                plt.subplot(1, 1, 1)
                # Calculate relative frequencies
                val_counts = df[col].value_counts(normalize=True).sort_values(ascending=True)

                # If too many categories, take top 20
                if len(val_counts) > 20:
                    val_counts = val_counts.tail(20)
                    plt.title(f"Top 20 Categories - {col}", pad=10)
                else:
                    plt.title(f"Frequency Distribution - {col}", pad=10)

                val_counts.plot(kind='barh', color='steelblue')
                plt.xlabel("Relative Frequency")

                # Add percentage labels
                for i, (idx, val) in enumerate(val_counts.items()):
                    plt.text(val, i, f' {val * 100:.1f}%', va='center', fontsize=9)

                plt.xlim(0, 1.1)  # Make room for labels

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(plots_dir, "univariate", f"{group}_{col}.png"),
                        dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f"- {group}/{col} saved")
            plotted_columns.add(col)


def generate_bivariate_plots(df, group_map, plots_dir):
    print("Starting Bivariate Analysis...")
    for group, cols in group_map.items():
        # Only correlate numeric columns
        numeric_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if len(numeric_cols) < 2:
            continue

        corr = df[numeric_cols].corr()

        # Find high correlations for scatter plots
        high_pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
            .head(3)
        )

        # Heatmap
        fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols) * 0.8),
                                        max(8, len(numeric_cols) * 0.7)))

        # Shorten names for heatmap readability
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

        # Scatter plots for top correlated pairs
        for (f1, f2), val in high_pairs.items():
            fig = plt.figure(figsize=(7, 6))
            sns.regplot(x=df[f1], y=df[f2], scatter_kws={"alpha": 0.6},
                        line_kws={"color": "red"})
            plt.title(f"{group}: {f1} vs {f2}\n(r={val:.2f})", fontsize=12, pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "bivariate", f"{group}_{f1}_vs_{f2}.png"),
                        dpi=DPI, bbox_inches='tight')
            plt.close()


def generate_target_plots(df, plots_dir):
    print("Starting Analysis vs Target...")

    if TARGET not in df.columns:
        print(f"Target {TARGET} not found in dataframe.")
        return

    # Normalize target variable distribution
    if pd.api.types.is_numeric_dtype(df[TARGET]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Target Variable: {TARGET}", fontsize=14, y=0.98)

        # Histogram with normalized density
        ax1.hist(df[TARGET].dropna(), bins=30, density=True, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel(TARGET)
        ax1.set_ylabel('Normalized Density')
        ax1.set_title('Normalized Distribution')
        ax1.grid(axis='y', alpha=0.3)

        # KDE plot
        try:
            df[TARGET].dropna().plot(kind='kde', ax=ax2, color='darkblue', linewidth=2)
            ax2.set_xlabel(TARGET)
            ax2.set_ylabel('Density')
            ax2.set_title('Kernel Density Estimate')
            ax2.grid(axis='y', alpha=0.3)
        except Exception as e:
            print(f"Could not plot KDE for target: {e}")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(plots_dir, "vs_target", f"{TARGET}_distribution.png"),
                    dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"- {TARGET} normalized distribution saved")

        # Feature correlation with target (Numeric only)
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != TARGET]
        if numeric_cols:
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
            ax.set_title(f"Top 5 Features correlated with {TARGET}", fontsize=14, pad=15)
            ax.set_xlabel("Absolute Correlation", fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "vs_target", "feature_importance.png"),
                        dpi=DPI, bbox_inches='tight')
            plt.close()
            print("- Feature importance saved")
    else:
        print(f"Target {TARGET} is not numeric, skipping correlation plots.")
