# DEPRECATED: This script is in the Old/ folder and should not be used.
# It causes experimental drift due to iterative augmentation practices.
# For reproducible experiments, use run_experiment.py with configs/*.yaml

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import os
from FirstProcessing.risk_calculation import calculate_risk_score
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

if os.path.exists("global_risk_threshold.txt"):
    with open("global_risk_threshold.txt", "r") as f:
        threshold_value = float(f.read().strip())
    print(f"Loaded global risk threshold: {threshold_value:.2f}")
else:
    print("Global risk threshold file not found. Using None.")
    threshold_value = None

INCOME_MIN = None
INCOME_MAX = None

# Final list of columns in the desired order
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

def preprocess_data(df):
    categorical_cols = [
        'Family_Status', 'Gender', 'Financial_Attitude', 'Budget_Planning', 'Save_Money',
        'Impulse_Buying_Category', 'Impulse_Buying_Reason', 'Debt_Level', 'Financial_Investments',
        'Bank_Account_Analysis_Frequency', 'Savings_Goal_Emergency_Fund',
        'Savings_Goal_Major_Purchases', 'Savings_Goal_Child_Education', 'Savings_Goal_Vacation',
        'Savings_Goal_Retirement', 'Savings_Goal_Other', 'Savings_Obstacle_Insufficient_Income',
        'Savings_Obstacle_Other_Expenses', 'Savings_Obstacle_Not_Priority', 'Savings_Obstacle_Other',
        'Credit_Essential_Needs', 'Credit_Major_Purchases', 'Credit_Unexpected_Expenses',
        'Credit_Personal_Needs', 'Credit_Never_Used', 'Impulse_Buying_Frequency',
        'Product_Lifetime_Clothing', 'Product_Lifetime_Tech', 'Product_Lifetime_Appliances',
        'Product_Lifetime_Cars'
    ]
    numeric_cols = [
        'Age', 'Income_Category', 'Essential_Needs_Percentage',
        'Expense_Distribution_Food', 'Expense_Distribution_Housing',
        'Expense_Distribution_Transport', 'Expense_Distribution_Entertainment',
        'Expense_Distribution_Health', 'Expense_Distribution_Personal_Care',
        'Expense_Distribution_Child_Education', 'Expense_Distribution_Other'
    ]
    df[numeric_cols] = df[numeric_cols].astype(str).replace({'months': '', 'years': ''}, regex=True)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    numeric_cols = [col for col in numeric_cols if not df[col].isna().all()]
    df[categorical_cols] = df[categorical_cols].fillna('missing').astype(str)
    return df, categorical_cols, numeric_cols

def safe_int(row, col):
    try:
        return int(float(row.get(col, 0)))
    except Exception:
        return 0

def enforce_constraints(row):
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

    credit_cols = [
        'Credit_Essential_Needs', 'Credit_Major_Purchases', 'Credit_Unexpected_Expenses',
        'Credit_Personal_Needs', 'Credit_Never_Used'
    ]
    if sum([safe_int(row, col) for col in credit_cols]) != 1:
        for col in credit_cols:
            row[col] = '0'
        row[np.random.choice(credit_cols)] = '1'

    try:
        age = int(float(row.get('Age', 18)))
    except:
        age = 18
    row['Age'] = str(max(18, min(age, 100)))

    try:
        enp = float(row.get('Essential_Needs_Percentage', 0))
    except:
        enp = 0.0
    row['Essential_Needs_Percentage'] = str(int(round(max(0, min(enp, 100)))))

    try:
        income_val = row.get('Income_Category', None)
        if income_val is not None and income_val != '':
            income = float(income_val)
        else:
            income = np.random.uniform(INCOME_MIN, INCOME_MAX)
    except Exception:
        income = np.random.uniform(INCOME_MIN, INCOME_MAX)
    row['Income_Category'] = str(int(round(income)))

    for col in ['Product_Lifetime_Clothing', 'Product_Lifetime_Tech', 'Product_Lifetime_Appliances', 'Product_Lifetime_Cars']:
        val = row.get(col, '')
        if isinstance(val, str) and 'months' in val and col == 'Product_Lifetime_Cars':
            try:
                months = int(val.split()[0])
                years = max(1, round(months / 12))
                row[col] = f"{years} years"
            except Exception:
                row[col] = "5 years"

    allowed_family_status = ["Single, no children", "In a relationship/married without children", "In a relationship/married with children"]
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

    if row.get('Impulse_Buying_Frequency', 'missing') == "Very rarely" and row.get('Impulse_Buying_Reason', 'missing') == "Self-reward":
        row['Impulse_Buying_Reason'] = "Discounts or promotions"

    return row

def apply_constraints(df):
    return df.apply(enforce_constraints, axis=1)

def main():
    global INCOME_MIN, INCOME_MAX
    df = pd.read_csv('../DatasetOriginal.csv')
    df, categorical_cols, numeric_cols = preprocess_data(df)

    # Set the minimum and maximum values for Income_Category
    INCOME_MIN = float(df['Income_Category'].min())
    INCOME_MAX = float(df['Income_Category'].max())

    X = df[categorical_cols + numeric_cols]
    y = df['Behavior_Risk_Level']
    original_data = pd.concat([X, y], axis=1)
    original_data = apply_constraints(original_data)

    # Train CTGAN on the original data
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(original_data)
    for col in categorical_cols + ['Behavior_Risk_Level']:
        metadata.update_column(column_name=col, sdtype='categorical')

    ctgan = CTGANSynthesizer(metadata=metadata, epochs=200, batch_size=500, verbose=True, cuda=True)
    ctgan.fit(original_data)

    # Total number of desired rows
    TOTAL_ROWS = 1000
    num_to_generate = max(0, TOTAL_ROWS - original_data.shape[0])

    if num_to_generate > 0:
        synthetic_samples = ctgan.sample(num_rows=num_to_generate).dropna()
        synthetic_samples = apply_constraints(synthetic_samples)

        # Convert numeric columns to integers
        for col in ['Age', 'Income_Category', 'Essential_Needs_Percentage',
                    'Expense_Distribution_Food', 'Expense_Distribution_Housing',
                    'Expense_Distribution_Transport', 'Expense_Distribution_Entertainment',
                    'Expense_Distribution_Health', 'Expense_Distribution_Personal_Care',
                    'Expense_Distribution_Child_Education', 'Expense_Distribution_Other']:
            synthetic_samples[col] = pd.to_numeric(synthetic_samples[col], errors='coerce').fillna(0)
            synthetic_samples[col] = synthetic_samples[col].round(0).astype(int)

        # Concatenate original data with synthetic samples
        final_dataset = pd.concat([original_data, synthetic_samples], ignore_index=True)
    else:
        final_dataset = original_data.copy()

    # 🔹 Apply the risk score function using the global threshold loaded from file
    final_dataset, _ = calculate_risk_score(final_dataset, threshold=threshold_value)

    # Ensure the correct column order
    final_dataset = final_dataset[FINAL_COL_ORDER]

    # Limit to TOTAL_ROWS
    final_dataset = final_dataset.head(TOTAL_ROWS)

    # Save the final dataset
    final_dataset.to_csv('WCGAN_augmented.csv', index=False)
    print("Final dataset saved as 'WCGAN_augmented.csv'.")

if __name__ == "__main__":
    main()