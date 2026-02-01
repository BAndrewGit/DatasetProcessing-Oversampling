# DEPRECATED: This script is in the Old/ folder and should not be used.
# It causes experimental drift due to iterative augmentation practices.
# For reproducible experiments, use run_experiment.py with configs/*.yaml

import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

from FirstProcessing.risk_calculation import calculate_risk_score

# Importăm funcțiile comune din scriptul WCGAN_Augmentation
from WCGAN_Augmentation import preprocess_data, safe_int, apply_constraints

# Citim pragul global din fișierul TXT
if os.path.exists("global_risk_threshold.txt"):
    with open("global_risk_threshold.txt", "r") as f:
        threshold_value = float(f.read().strip())
    print(f"Loaded global risk threshold: {threshold_value:.2f}")
else:
    print("Global risk threshold file not found. Using None.")
    threshold_value = None

# Variabile pentru Income_Category – se vor seta în main ca float
INCOME_MIN = None
INCOME_MAX = None

# Lista finală de coloane în ordinea dorită
FINAL_COL_ORDER = [
    'Age', 'Family_Status', 'Gender', 'Income_Category', 'Essential_Needs_Percentage',
    'Financial_Attitude', 'Budget_Planning', 'Save_Money', 'Product_Lifetime_Clothing',
    'Product_Lifetime_Tech', 'Product_Lifetime_Appliances', 'Product_Lifetime_Cars',
    'Impulse_Buying_Frequency', 'Impulse_Buying_Category', 'Impulse_Buying_Reason',
    'Debt_Level', 'Financial_Investments', 'Bank_Account_Analysis_Frequency',
    'Savings_Goal_Emergency_Fund', 'Savings_Goal_Major_Purchases', 'Savings_Goal_Child_Education',
    'Savings_Goal_Vacation', 'Savings_Goal_Retirement', 'Savings_Goal_Other',
    'Savings_Obstacle_Insufficient_Income', 'Savings_Obstacle_Other_Expenses',
    'Savings_Obstacle_Not_Priority', 'Savings_Obstacle_Other',
    'Expense_Distribution_Food', 'Expense_Distribution_Housing', 'Expense_Distribution_Transport',
    'Expense_Distribution_Entertainment', 'Expense_Distribution_Health',
    'Expense_Distribution_Personal_Care', 'Expense_Distribution_Child_Education',
    'Expense_Distribution_Other', 'Credit_Essential_Needs', 'Credit_Major_Purchases',
    'Credit_Unexpected_Expenses', 'Credit_Personal_Needs', 'Credit_Never_Used',
    'Behavior_Risk_Level'
]

# Constantă pentru numărul total de rânduri dorit
TOTAL_ROWS = 1000


# Modificare a funcției enforce_constraints pentru a trata cazurile în care Income_Category nu e numeric
def enforce_constraints(row):
    # Constrângeri pentru Savings
    savings_goal_cols = [
        'Savings_Goal_Emergency_Fund', 'Savings_Goal_Major_Purchases', 'Savings_Goal_Child_Education',
        'Savings_Goal_Vacation', 'Savings_Goal_Retirement', 'Savings_Goal_Other'
    ]
    savings_obstacle_cols = [
        'Savings_Obstacle_Insufficient_Income', 'Savings_Obstacle_Other_Expenses',
        'Savings_Obstacle_Not_Priority', 'Savings_Obstacle_Other'
    ]
    if row.get('Save_Money', 'missing') == "Yes":
        for col in savings_obstacle_cols:
            row[col] = '0'
        if sum([safe_int(row, col) for col in savings_goal_cols]) < 1:
            row[np.random.choice(savings_goal_cols)] = '1'
    elif row.get('Save_Money', 'missing') == "No":
        for col in savings_goal_cols:
            row[col] = '0'
        if sum([safe_int(row, col) for col in savings_obstacle_cols]) != 1:
            for col in savings_obstacle_cols:
                row[col] = '0'
            row[np.random.choice(savings_obstacle_cols)] = '1'

    # Constrângeri pentru Expense Distribution
    expense_distribution_cols = [
        'Expense_Distribution_Food', 'Expense_Distribution_Housing', 'Expense_Distribution_Transport',
        'Expense_Distribution_Entertainment', 'Expense_Distribution_Health',
        'Expense_Distribution_Personal_Care', 'Expense_Distribution_Child_Education',
        'Expense_Distribution_Other'
    ]
    if sum([safe_int(row, col) for col in expense_distribution_cols]) != 1:
        for col in expense_distribution_cols:
            row[col] = '0'
        row[np.random.choice(expense_distribution_cols)] = '1'

    # Constrângeri pentru Credit
    credit_cols = [
        'Credit_Essential_Needs', 'Credit_Major_Purchases', 'Credit_Unexpected_Expenses',
        'Credit_Personal_Needs', 'Credit_Never_Used'
    ]
    if sum([safe_int(row, col) for col in credit_cols]) != 1:
        for col in credit_cols:
            row[col] = '0'
        row[np.random.choice(credit_cols)] = '1'

    # Age
    try:
        age = int(float(row.get('Age', 18)))
    except:
        age = 18
    row['Age'] = str(max(18, min(age, 100)))

    # Essential_Needs_Percentage
    try:
        enp = float(row.get('Essential_Needs_Percentage', 0))
    except:
        enp = 0.0
    row['Essential_Needs_Percentage'] = str(int(round(max(0, min(enp, 100)))))

    # Income_Category: verificăm dacă valoarea e numerică, altfel atribuim o valoare aleatorie
    try:
        income_val = row.get('Income_Category', None)
        income_val_numeric = pd.to_numeric(income_val, errors='coerce')
        if pd.isna(income_val_numeric):
            raise ValueError("Non-numeric income")
        income = income_val_numeric
    except Exception:
        min_val = INCOME_MIN if INCOME_MIN is not None else 0
        max_val = INCOME_MAX if INCOME_MAX is not None else 10000
        income = np.random.uniform(min_val, max_val)
    row['Income_Category'] = str(int(round(income)))

    # Product Lifetime: pentru Product_Lifetime_Cars convertim 'months' în 'years'
    for col in ['Product_Lifetime_Clothing', 'Product_Lifetime_Tech', 'Product_Lifetime_Appliances',
                'Product_Lifetime_Cars']:
        val = row.get(col, '')
        if isinstance(val, str) and 'months' in val and col == 'Product_Lifetime_Cars':
            try:
                months = int(val.split()[0])
                years = max(1, round(months / 12))
                row[col] = f"{years} years"
            except Exception:
                row[col] = "5 years"

    # Valori implicite pentru categorice
    allowed_family_status = ["Single, no children", "In a relationship/married without children",
                             "In a relationship/married with children"]
    if row.get('Family_Status', '') not in allowed_family_status:
        row['Family_Status'] = allowed_family_status[0]
    allowed_financial_attitude = ["I try to find a balance", "I am disciplined in saving"]
    if row.get('Financial_Attitude', '') not in allowed_financial_attitude:
        row['Financial_Attitude'] = allowed_financial_attitude[0]
    allowed_impulse_reason = ["Self-reward", "Discounts or promotions"]
    if row.get('Impulse_Buying_Reason', '') not in allowed_impulse_reason:
        row['Impulse_Buying_Reason'] = allowed_impulse_reason[0]

    if row.get('Debt_Level', '') == "Absent":
        allowed_financial_investments = ["Yes, occasionally", "No, but interested"]
        if row.get('Financial_Investments', '') not in allowed_financial_investments:
            row['Financial_Investments'] = np.random.choice(allowed_financial_investments)
    elif row.get('Debt_Level', '') in ["Low", "Manageable"]:
        if row.get('Financial_Investments', '') == "Yes, regularly":
            row['Financial_Investments'] = "No, but interested"

    if row.get('Impulse_Buying_Frequency', 'missing') == "Very rarely" and row.get('Impulse_Buying_Reason',
                                                                                   'missing') == "Self-reward":
        row['Impulse_Buying_Reason'] = "Discounts or promotions"

    return row


# Ne asigurăm că apply_constraints folosește noua versiune de enforce_constraints
def apply_constraints(df):
    return df.apply(enforce_constraints, axis=1)


def main():
    global INCOME_MIN, INCOME_MAX

    # Citim datele din fișierul DatasetOriginal.csv
    df = pd.read_csv('../DatasetOriginal.csv')
    df, categorical_cols, numeric_cols = preprocess_data(df)

    # Convertim Income_Category la numeric și completăm eventualele valori lipsă
    df['Income_Category'] = pd.to_numeric(df['Income_Category'], errors='coerce')
    df['Income_Category'].fillna(df['Income_Category'].median(), inplace=True)
    INCOME_MIN = float(df['Income_Category'].min())
    INCOME_MAX = float(df['Income_Category'].max())

    # ADASYN: pregătim datele folosind pipeline-uri pentru categorice și numerice
    X = df[categorical_cols + numeric_cols]
    y = df['Behavior_Risk_Level']

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_pipeline, categorical_cols),
            ('num', num_pipeline, numeric_cols)
        ]
    )
    X_encoded = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, stratify=y, random_state=42
    )

    adasyn = ADASYN(random_state=42)
    X_adasyn_np, y_adasyn_np = adasyn.fit_resample(X_train, y_train)

    # Decodificăm datele ADASYN pentru categorice și numerice
    cat_transformer = preprocessor.named_transformers_['cat']
    onehot = cat_transformer.named_steps['onehot']
    n_cat_features = onehot.transform(X[categorical_cols]).shape[1]
    X_adasyn_cat = onehot.inverse_transform(X_adasyn_np[:, :n_cat_features])
    X_adasyn_cat = pd.DataFrame(X_adasyn_cat, columns=categorical_cols)

    num_transformer = preprocessor.named_transformers_['num']
    X_adasyn_num = num_transformer.named_steps['scaler'].inverse_transform(
        X_adasyn_np[:, n_cat_features:]
    )
    X_adasyn_num = pd.DataFrame(X_adasyn_num, columns=numeric_cols)

    X_adasyn = pd.concat([X_adasyn_cat, X_adasyn_num], axis=1)
    X_adasyn = apply_constraints(X_adasyn)

    data_combined = pd.concat([X_adasyn, pd.Series(y_adasyn_np, name='Behavior_Risk_Level')], axis=1)

    # Antrenăm CTGAN pe datele combinate (cu etichetă inclusă)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data_combined)
    for col in categorical_cols + ['Behavior_Risk_Level']:
        metadata.update_column(column_name=col, sdtype='categorical')
    ctgan = CTGANSynthesizer(metadata=metadata, epochs=200, batch_size=500, verbose=True, cuda=True)
    ctgan.fit(data_combined)

    num_to_generate = max(0, TOTAL_ROWS - data_combined.shape[0])
    if num_to_generate > 0:
        synthetic_samples = ctgan.sample(num_rows=num_to_generate).dropna()
        synthetic_samples = apply_constraints(synthetic_samples)
        final_dataset = pd.concat([data_combined, synthetic_samples], ignore_index=True)
    else:
        final_dataset = data_combined.copy()

    # Post-procesare: convertim coloanele numerice la întregi
    for col in ['Age', 'Income_Category', 'Essential_Needs_Percentage',
                'Expense_Distribution_Food', 'Expense_Distribution_Housing',
                'Expense_Distribution_Transport', 'Expense_Distribution_Entertainment',
                'Expense_Distribution_Health', 'Expense_Distribution_Personal_Care',
                'Expense_Distribution_Child_Education', 'Expense_Distribution_Other']:
        final_dataset[col] = pd.to_numeric(final_dataset[col], errors='coerce').fillna(0)
        final_dataset[col] = final_dataset[col].round(0).astype(int)

    # Aplicăm funcția de calcul al scorului de risc folosind pragul încărcat din fișier
    final_dataset, _ = calculate_risk_score(final_dataset, threshold=threshold_value)

    # Reordonăm coloanele conform listei FINAL_COL_ORDER
    final_dataset = final_dataset[FINAL_COL_ORDER]

    final_dataset = final_dataset.head(TOTAL_ROWS)
    final_dataset.to_csv('ADASYN_WCGAN_augmented.csv', index=False)
    print("Final dataset saved as 'ADASYN_WCGAN_augmented.csv'.")


if __name__ == "__main__":
    main()
