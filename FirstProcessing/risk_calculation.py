import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import IsolationForest
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier
import joblib

#Fitting + save scaler
def fit_and_save_scaler(df, columns, scaler_path):
    scaler = StandardScaler()
    scaler.fit(df[columns])
    joblib.dump(scaler, scaler_path)
    df[columns] = scaler.transform(df[columns])
    return df

#Transform using scaler
def apply_existing_scaler(df, columns, scaler_path):
    scaler = joblib.load(scaler_path)
    df[columns] = scaler.transform(df[columns])
    return df

# Basic risk scoring via GMM clustering over weighted feature
def calculate_risk_clusters(df, cluster_range=(2, 8)):
    config = {
        'weights': {
            'Debt_Level': 0.25,
            'Impulse_Buying_Frequency': 0.15,
            'Essential_Needs_Percentage': -0.2,
            'Savings_Goal_Emergency_Fund': 0.1,
            'Bank_Account_Analysis_Frequency': -0.1
        }
    }

    def calculate_risk_score(df_local):
        scaler = RobustScaler()
        scaled = scaler.fit_transform(df_local[config['weights'].keys()])
        return pd.Series(np.dot(scaled, list(config['weights'].values())), index=df_local.index)

    df = df.copy()
    df['Risk_Score'] = calculate_risk_score(df)

    best_score = -1
    best_n_clusters = None
    best_model = None
    best_labels = None

    X = df[['Risk_Score']]

    # Find optimal number of clusters using Silhouette Score
    for n_clusters in range(cluster_range[0], cluster_range[1] + 1):
        model = BayesianGaussianMixture(n_components=n_clusters, max_iter=500, random_state=42)
        labels = model.fit_predict(X)

        try:
            score = silhouette_score(X, labels)
            print(f"Clusters={n_clusters}: Silhouette Score={score:.3f}")
        except Exception as e:
            print(f"Silhouette calculation failed for {n_clusters} clusters: {e}")
            continue

        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_model = model
            best_labels = labels

    if best_model is None:
        raise Exception("No valid clustering found.")

    print(f"\nBest clustering: {best_n_clusters} clusters with Silhouette Score={best_score:.3f}")

    df['Cluster'] = best_labels

    # Label safest cluster (lowest average risk) as 0
    safe_cluster = df.groupby('Cluster')['Risk_Score'].mean().idxmin()
    df['Behavior_Risk_Level'] = np.where(df['Cluster'] == safe_cluster, 0, 1)

    print("\nFinal Risk distribution:")
    print(df['Behavior_Risk_Level'].value_counts())

    return df

# Advanced multi-stage risk labeling: outliers + clustering + iterative ML
def calculate_risk_advanced(df, gmm_clusters=3, confidence_threshold=0.9, iterations=2):
    config = {
        'weights': {
            'Debt_Level': 0.25,
            'Impulse_Buying_Frequency': 0.15,
            'Essential_Needs_Percentage': -0.2,
            'Savings_Goal_Emergency_Fund': 0.1,
            'Bank_Account_Analysis_Frequency': -0.1
        },
        'outlier_contamination': 0.05,
        'gmm_clusters': gmm_clusters,
        'knn_neighbors': 7
    }

    def calculate_risk_score(df_local):
        scaler = RobustScaler()
        scaled = scaler.fit_transform(df_local[config['weights'].keys()])
        return pd.Series(np.dot(scaled, list(config['weights'].values())), index=df_local.index)

    df = df.copy()
    df['Risk_Score'] = calculate_risk_score(df)

    # Outlier detection using Isolation Forest
    iso = IsolationForest(contamination=config['outlier_contamination'], random_state=42)
    df['Outlier'] = np.where(iso.fit_predict(df[['Risk_Score']]) == -1, 1, 0)

    # Cluster using Bayesian GMM
    bgmm = BayesianGaussianMixture(n_components=config['gmm_clusters'],
                                   weight_concentration_prior=0.1,
                                   max_iter=500, random_state=42)
    df['Cluster'] = bgmm.fit_predict(df[['Risk_Score']])

    safe_cluster = np.argmin(bgmm.means_.flatten())
    df['Auto_Label'] = np.where(df['Cluster'] == safe_cluster, 0, 1)

    # Predict labels for outliers using KNN
    knn = KNeighborsClassifier(n_neighbors=config['knn_neighbors'])
    knn.fit(df[df['Outlier'] == 0][['Risk_Score']], df[df['Outlier'] == 0]['Auto_Label'])
    df.loc[df['Outlier'] == 1, 'Auto_Label'] = knn.predict(df[df['Outlier'] == 1][['Risk_Score']])

    df['Confidence'] = 1.0
    df['Behavior_Risk_Level'] = -1

    features = df.columns.difference(['Behavior_Risk_Level', 'Confidence', 'Auto_Label'])
    xgb = XGBClassifier(
        scale_pos_weight=1.5,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        learning_rate=0.05,
        n_estimators=100,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='logloss',
        random_state=42
    )

    # Iteratively refine high-confidence labels using XGBoost + SHAP
    for iteration in range(iterations):
        train = df[df['Behavior_Risk_Level'] != -1]
        if len(train) < 10:
            train = df[df['Confidence'] > confidence_threshold]

        if len(train) < 10:
            print(f"Iteration {iteration + 1}: not enough high-confidence samples, stopping...")
            break

        print(f"Iteration {iteration + 1}: training on {len(train)} samples...")
        xgb.fit(train[features], train['Auto_Label'])

        probas = xgb.predict_proba(df.loc[df['Behavior_Risk_Level'] == -1, features])
        confidences = np.max(probas, axis=1)
        predictions = xgb.predict(df.loc[df['Behavior_Risk_Level'] == -1, features])

        high_confidence_indices = df.loc[df['Behavior_Risk_Level'] == -1].index[confidences >= confidence_threshold]

        df.loc[high_confidence_indices, 'Behavior_Risk_Level'] = predictions[confidences >= confidence_threshold]
        df.loc[high_confidence_indices, 'Confidence'] = confidences[confidences >= confidence_threshold]

        # Explainability (SHAP importance summary)
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(train[features])
        shap_summary = dict(zip(features, np.abs(shap_values).mean(axis=0)))
        print(f"Iteration {iteration + 1} SHAP Importances:", shap_summary)

        if len(high_confidence_indices) == 0:
            print(f"Iteration {iteration + 1}: no new high-confidence labels assigned, stopping...")
            break

    # Apply fallback rule-based logic
    def apply_rules(row):
        score = 0
        if row['Income_Category'] >= 15000:
            score += 2
        if row['Debt_Level'] == 4:
            score -= 3
        if row['Debt_Level'] >= 2 and row['Income_Category'] < 12000:
             score -= 2
        if row.get('Financial_Investments_Yes, regularly', 0) == 1:
            score += 2
        if row.get('Budget_Planning_Plan budget in detail', 0) == 1:
            score += 1
        if row.get('Save_Money_Yes', 0) == 1:
            score += 1
        else:
            if row['Income_Category'] >= 5000:
                score -= 1
            else:
                score -= 0.5
        if row.get('Relationship_Status_In a relationship/married with children', 0) == 1:
            score += 1

        if score >= 2:
            return 0
        elif score <= -2:
            return 1
        return row['Behavior_Risk_Level']

    df['Behavior_Risk_Level'] = df.apply(apply_rules, axis=1)

    labeled = df[df['Behavior_Risk_Level'] != -1]
    unlabeled = df[df['Behavior_Risk_Level'] == -1]

    print(f"\nFinal labeled count: {len(labeled)}, Unlabeled (needs manual review): {len(unlabeled)}")

    # Final scoring summary
    if len(labeled) > 0:
        silhouette = silhouette_score(labeled[['Risk_Score']], labeled['Behavior_Risk_Level'])
        print(f"\nSilhouette Score (final labeled data): {silhouette:.2f}")

        print("\nClassification Report (final labeled data):")
        print(classification_report(labeled['Auto_Label'], labeled['Behavior_Risk_Level']))
    else:
        print("No labeled data available for final scoring.")

    return df


def scale_numeric_columns(df, columns, scaler_path=None, fit=True):
    if fit:
        if scaler_path:
            return fit_and_save_scaler(df, columns, scaler_path)
        else:
            scaler = StandardScaler()
            df = df.copy()
            df[columns] = scaler.fit_transform(df[columns])
            return df
    else:
        if not scaler_path:
            raise ValueError("scaler_path required when fit=False")
        return apply_existing_scaler(df, columns, scaler_path)


# Backwards-compatible wrapper expected by old augmentation scripts
def calculate_risk_score(df, method='advanced', **kwargs):
    """
    Wrapper to produce (df_with_risk, metadata) to be compatible with older scripts.
    method: 'advanced' (default) or 'clusters'
    Returns: (df, info_dict)
    """
    if method == 'clusters':
        df_out = calculate_risk_clusters(df, **{k: v for k, v in kwargs.items() if k in ['cluster_range']})
        return df_out, {'method': 'clusters'}
    else:
        df_out = calculate_risk_advanced(df, **{k: v for k, v in kwargs.items() if k in ['gmm_clusters', 'confidence_threshold', 'iterations']})
        return df_out, {'method': 'advanced'}

