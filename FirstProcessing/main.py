import pandas as pd
import os
from tkinter import Tk, filedialog
from FirstProcessing.preprocessing import normalize_and_translate_data, postprocess_data, range_smoothing
from FirstProcessing.risk_calculation import calculate_risk_advanced, fit_and_save_scaler, apply_existing_scaler
from FirstProcessing.data_generation import random_product_lifetime
from FirstProcessing.file_operations import auto_adjust_column_width
from pathlib import Path

# Limit parallel processing CPU usage
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Risk scoring weights configuration
CONFIG = {
    'weights': {
        'Budget_Planning_Plan budget in detail': 0.097,
        'Budget_Planning_Plan only essentials': 0.0805,
        'Age': 0.0782,
        'Family_Status_Single, no children': 0.0717,
        'Financial_Investments_Yes, regularly': 0.0705,
        'Gender_Male': 0.0682,
        'Impulse_Buying_Category_Food': 0.0682,
        'Family_Status_Another': 0.0647,
        'Impulse_Buying_Reason_Social pressure': 0.0629,
        'Impulse_Buying_Category_Other': 0.0611,
        'Gender_Female': 0.0594,
        'Impulse_Buying_Category_Entertainment': 0.0594,
        'Savings_Goal': 0.0547,
        'Savings_Obstacle_0001': 0.0529,
        'Savings_Obstacle_0010': 0.0506
    }
}

multi_value_cols = ["Savings_Goal", "Savings_Obstacle", "Expense_Distribution", "Credit_Usage"]

# Display NaN summary and example rows
def check_nan_values(df):
    nan_info = df.isna().sum()
    nan_columns = nan_info[nan_info > 0]

    if len(nan_columns) == 0:
        print("\n>>> INFO: No NaN values in data.")
        return

    print("\n>>> WARNING: Detected NaN values in the following columns:")
    for col, count in nan_columns.items():
        print(f"- Column '{col}': {count} NaN values")

    for col in nan_columns.index:
        nan_rows = df[df[col].isna()]
        print(f"\nFirst 5 rows with NaN in column '{col}':")
        print(nan_rows.head(5))

def process_dataset(input_path: str, output_dir: str = "data/processed", output_prefix: str = "1") -> str | None:
    """
    Procesează dataset-ul programatic (fără dialoguri).

    Args:
        input_path: Calea către fișierul CSV brut
        output_dir: Directorul unde se salvează rezultatele
        output_prefix: Prefixul pentru fișierele de output

    Returns:
        Calea către fișierul CSV encodat sau None dacă a eșuat
    """
    from pathlib import Path

    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"Loading file: {input_path}")
        df_raw = pd.read_csv(input_path, sep=",", quotechar='"', engine="python")

        print("\n>>> Normalizing and translating data...")
        df_norm = normalize_and_translate_data(df_raw)

        # Lifetime columns to smooth
        lifetime_cols = [
            "Product_Lifetime_Clothing", "Product_Lifetime_Tech",
            "Product_Lifetime_Appliances", "Product_Lifetime_Cars"
        ]

        # Decoded version (text-based)
        df_decoded = df_norm.copy()
        for col in multi_value_cols:
            if col not in df_decoded.columns:
                if col in df_norm.columns:
                    df_decoded[col] = df_norm[col]
                else:
                    df_decoded[col] = pd.NA

        df_decoded = range_smoothing(
            df_decoded,
            age_column="Age",
            income_column="Income_Category",
            lifetime_columns=lifetime_cols,
            essential_needs_column="Essential_Needs_Percentage",
            lifetime_func=random_product_lifetime
        )

        # Reorder decoded columns
        desired_column_order = [
            'Age', 'Family_Status', 'Gender', 'Income_Category', 'Essential_Needs_Percentage',
            'Financial_Attitude', 'Budget_Planning', 'Save_Money', 'Savings_Goal', 'Savings_Obstacle',
            'Expense_Distribution', 'Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
            'Product_Lifetime_Appliances', 'Product_Lifetime_Cars', 'Impulse_Buying_Frequency',
            'Impulse_Buying_Category', 'Impulse_Buying_Reason', 'Credit_Usage', 'Debt_Level',
            'Financial_Investments', 'Bank_Account_Analysis_Frequency', 'Behavior_Risk_Level'
        ]
        df_decoded = df_decoded[[col for col in desired_column_order if col in df_decoded.columns]]

        # Encoded version (for scoring)
        df_encoded = df_norm.copy()
        df_encoded.drop(columns=multi_value_cols, inplace=True, errors='ignore')
        df_encoded.rename(columns=lambda x: x.replace('_encoded', ''), inplace=True)

        df_encoded = range_smoothing(
            df_encoded,
            age_column="Age",
            income_column="Income_Category",
            lifetime_columns=lifetime_cols,
            essential_needs_column="Essential_Needs_Percentage",
            lifetime_func=lambda x: random_product_lifetime(x, encoded=True)
        )

        print("\n>>> Post-processing data (encoded version)...")
        df_encoded = postprocess_data(df_encoded)
        if df_encoded is None:
            return None

        # Normalize numeric columns
        numeric_cols_to_scale = [
            'Age', 'Income_Category', 'Essential_Needs_Percentage',
            'Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
            'Product_Lifetime_Appliances', 'Product_Lifetime_Cars'
        ]
        scaler_path = Path("scaler/robust_scaler.pkl")
        scaler_path.parent.mkdir(parents=True, exist_ok=True)

        # Flag: True rewrite scaler
        RETRAIN_SCALER = True

        if RETRAIN_SCALER:
            df_encoded = fit_and_save_scaler(df_encoded, numeric_cols_to_scale, scaler_path)
        else:
            df_encoded = apply_existing_scaler(df_encoded, numeric_cols_to_scale, scaler_path)

        print("\n>>> Calculating risk score...")
        df_encoded = calculate_risk_advanced(df_encoded)
        if df_encoded is None:
            return None

        print("\nRisk distribution:")
        print(df_encoded['Behavior_Risk_Level'].value_counts(dropna=False))
        if len(df_encoded['Behavior_Risk_Level'].unique()) == 1:
            print("\nNot enough risk variation for analysis!")
            return None

        # Transfer risk label to decoded version
        df_decoded['Behavior_Risk_Level'] = df_encoded['Behavior_Risk_Level'].apply(
            lambda x: "Risky" if x == 1 else "Beneficial"
        )

        # Save files automatically
        auxiliary_cols = ['Confidence', 'Cluster', 'Auto_Label', 'Outlier']
        df_encoded_export = df_encoded.drop(columns=auxiliary_cols, errors='ignore')

        # Save decoded Excel
        excel_save_path = os.path.join(output_dir, f"{output_prefix}_decoded.xlsx")
        with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
            df_decoded.to_excel(writer, index=False, sheet_name='Decoded_Data')
            auto_adjust_column_width(writer, 'Decoded_Data')
        print(f"✓ Decoded Excel saved: {excel_save_path}")

        # Save encoded CSV
        csv_save_path = os.path.join(output_dir, f"{output_prefix}_encoded.csv")
        df_encoded_export.to_csv(csv_save_path, index=False, encoding='utf-8')
        print(f"✓ Encoded CSV saved: {csv_save_path}")

        print("\n✓ Processing complete!")
        return csv_save_path

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return None


# Full pipeline: load, clean, encode, score, export (versiunea interactivă cu dialog)
def main():
    Tk().withdraw()

    # Input CSV selection
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv")],
        title="Select a CSV file to process"
    )
    if not file_path:
        print("No file selected.")
        return

    try:
        print(f"Loading file: {file_path}")
        df_raw = pd.read_csv(file_path, sep=",", quotechar='"', engine="python")

        print("\n>>> Normalizing and translating data...")
        df_norm = normalize_and_translate_data(df_raw)

        # Lifetime columns to smooth
        lifetime_cols = [
            "Product_Lifetime_Clothing", "Product_Lifetime_Tech",
            "Product_Lifetime_Appliances", "Product_Lifetime_Cars"
        ]

        # Decoded version (text-based)
        df_decoded = df_norm.copy()
        for col in multi_value_cols:
            if col not in df_decoded.columns:
                if col in df_norm.columns:
                    df_decoded[col] = df_norm[col]
                else:
                    df_decoded[col] = pd.NA

        df_decoded = range_smoothing(
            df_decoded,
            age_column="Age",
            income_column="Income_Category",
            lifetime_columns=lifetime_cols,
            essential_needs_column="Essential_Needs_Percentage",
            lifetime_func=random_product_lifetime
        )

        # Reorder decoded columns
        desired_column_order = [
            'Age', 'Family_Status', 'Gender', 'Income_Category', 'Essential_Needs_Percentage',
            'Financial_Attitude', 'Budget_Planning', 'Save_Money', 'Savings_Goal', 'Savings_Obstacle',
            'Expense_Distribution', 'Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
            'Product_Lifetime_Appliances', 'Product_Lifetime_Cars', 'Impulse_Buying_Frequency',
            'Impulse_Buying_Category', 'Impulse_Buying_Reason', 'Credit_Usage', 'Debt_Level',
            'Financial_Investments', 'Bank_Account_Analysis_Frequency', 'Behavior_Risk_Level'
        ]
        df_decoded = df_decoded[[col for col in desired_column_order if col in df_decoded.columns]]

        # Encoded version (for scoring)
        df_encoded = df_norm.copy()
        df_encoded.drop(columns=multi_value_cols, inplace=True, errors='ignore')
        df_encoded.rename(columns=lambda x: x.replace('_encoded', ''), inplace=True)

        df_encoded = range_smoothing(
            df_encoded,
            age_column="Age",
            income_column="Income_Category",
            lifetime_columns=lifetime_cols,
            essential_needs_column="Essential_Needs_Percentage",
            lifetime_func=lambda x: random_product_lifetime(x, encoded=True)
        )

        print("\n>>> Post-processing data (encoded version)...")
        df_encoded = postprocess_data(df_encoded)
        if df_encoded is None:
            return

        # Normalize numeric columns
        numeric_cols_to_scale = [
            'Age', 'Income_Category', 'Essential_Needs_Percentage',
            'Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
            'Product_Lifetime_Appliances', 'Product_Lifetime_Cars'
        ]
        scaler_path = Path("scaler/robust_scaler.pkl")
        scaler_path.parent.mkdir(parents=True, exist_ok=True)

        # Flag: True rewrite scaller
        RETRAIN_SCALER = True

        if RETRAIN_SCALER:
            df_encoded = fit_and_save_scaler(df_encoded, numeric_cols_to_scale, scaler_path)
        else:
            df_encoded = apply_existing_scaler(df_encoded, numeric_cols_to_scale, scaler_path)

        print("\n>>> Calculating risk score...")
        df_encoded = calculate_risk_advanced(df_encoded) # Select desired risk calculation method
        if df_encoded is None:
            return

        print("\nRisk distribution:")
        print(df_encoded['Behavior_Risk_Level'].value_counts(dropna=False))
        if len(df_encoded['Behavior_Risk_Level'].unique()) == 1:
            print("\nNot enough risk variation for analysis!")
            print("Possible solutions:")
            print("- Adjust CONFIG['risk_weights']")
            print("- Review input data")
            return

        # Transfer risk label to decoded version
        df_decoded['Behavior_Risk_Level'] = df_encoded['Behavior_Risk_Level'].apply(
            lambda x: "Risky" if x == 1 else "Beneficial"
        )

        # Save decoded Excel
        Tk().withdraw()
        excel_save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Select save location for decoded Excel file"
        )
        if not excel_save_path:
            print("No save location selected for Excel.")
            return

        auxiliary_cols = ['Confidence', 'Cluster', 'Auto_Label', 'Outlier']
        df_encoded_export = df_encoded.drop(columns=auxiliary_cols, errors='ignore')

        with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
            df_decoded.to_excel(writer, index=False, sheet_name='Decoded_Data')
            auto_adjust_column_width(writer, 'Decoded_Data')
        print(f"Decoded Excel file saved at: {excel_save_path}")

        # Save encoded CSV
        base_name, _ = os.path.splitext(excel_save_path)
        csv_save_path = base_name + "_encoded.csv"
        df_encoded_export.to_csv(csv_save_path, index=False, encoding='utf-8')
        print(f"Encoded CSV file saved at: {csv_save_path}")

        print("\nProcessing complete!")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
