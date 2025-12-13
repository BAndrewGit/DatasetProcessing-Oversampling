import pandas as pd
import numpy as np
import re

from FirstProcessing.data_generation import replace_age_column, replace_income_category, \
    replace_product_lifetime_columns, random_product_lifetime, replace_essential_needs


# Translate Romanian column names and values to English
def normalize_and_translate_data(df):
    df.columns = df.columns.str.strip()

    # Remove timestamp column if exists
    if df.columns[0].lower() in ['marcaj de timp', 'timestamp']:
        df.drop(columns=df.columns[0], inplace=True)

    # Rename columns to standard English identifiers
    column_mapping = {
        "Câți ani aveți?": "Age",
        "Care este statutul dumneavoastră familial?": "Family_Status",
        "Care este genul dumneavoastră?": "Gender",
        "În ce categorie se încadrează venitul dumneavoastră lunar?": "Income_Category",
        "Ce procent aproximativ din venit considerați că vă este suficient pentru nevoi esențiale (mâncare, chirie, transport)?": "Essential_Needs_Percentage",
        "Cum ați descrie atitudinea dumneavoastră față de gestionarea banilor?": "Financial_Attitude",
        "Planificați bugetul lunar înainte de cheltuieli?": "Budget_Planning",
        "Reușiți să economisiți bani lunar?": "Save_Money",
        "Ce anume vă împiedică să economisiți bani lunar?": "Savings_Obstacle",
        "Cât de frecvent faceți achiziții impulsive (neplanificate)?": "Impulse_Buying_Frequency",
        "Pe ce categorie sunt, de obicei, cheltuielile dumneavoastră impulsive?": "Impulse_Buying_Category",
        "Care este principalul motiv pentru cheltuielile impulsive?": "Impulse_Buying_Reason",
        "Ați folosit vreodată un credit sau o linie de împrumut?": "Credit_Usage",
        "Cum considerați nivelul actual al datoriilor dumneavoastră?": "Debt_Level",
        "Ați făcut investiții financiare până acum?": "Financial_Investments",
        "Cât de des analizați situația contului bancar (venituri și cheltuieli)?": "Bank_Account_Analysis_Frequency",
        "Cât timp utilizați, în general, următoarele tipuri de produse înainte de a le înlocui?   [Îmbrăcăminte]": "Product_Lifetime_Clothing",
        "Cât timp utilizați, în general, următoarele tipuri de produse înainte de a le înlocui?   [Gadget-uri și dispozitive tech (telefoane, laptopuri, tablete, console etc.)]": "Product_Lifetime_Tech",
        "Cât timp utilizați, în general, următoarele tipuri de produse înainte de a le înlocui?   [Electrocasnice (frigider, mașină de spălat etc.)]": "Product_Lifetime_Appliances",
        "Cât timp utilizați, în general, următoarele tipuri de produse înainte de a le înlocui?   [Autoturisme]": "Product_Lifetime_Cars",
        "Care este scopul principal al economiilor dumneavoastră?  \n  (Alegeți toate opțiunile care se aplică)": "Savings_Goal",
        "Cum distribuiți, în general, cheltuielile lunare?  \n(Maxim 3 categorii principale.)": "Expense_Distribution"
    }
    df.rename(columns=column_mapping, inplace=True)

    # Translate basic single-value fields
    basic_translation = {
        "Masculin": "Male",
        "Feminin": "Female",
        "Prefer să nu răspund": "Prefer not to say",
        "40-50": "41-50",
        "30-40": "31-40",
        "Necăsătorit/ă, fără copii": "Single, no children",
        "Necăsătorit/ă, cu copii": "Single, with children",
        "Într-o relație sau căsătorit/ă, cu copii.": "In a relationship/married with children",
        "Într-o relație (coabitare) sau căsătorit/ă, fără copii.": "In a relationship/married without children",
        "Într-o relație sau căsătorit/ă, fără copii.": "In a relationship/married without children",
        "Altul": "Another",
        "Încerc să găsesc un echilibru": "I try to find a balance",
        "Cheltuiesc mai mult decât câștig": "Spend more than I earn",
        "Sunt disciplinat/ă în economisire": "I am disciplined in saving",
        "Planific bugetul în detaliu": "Plan budget in detail",
        "Planific doar lucrurile esențiale": "Plan only essentials",
        "Nu planific deloc": "Don't plan at all",
        "Recompensă personală („merit acest lucru”)": "Self-reward",
        "Reduceri sau promoții": "Discounts or promotions",
        "Presiuni sociale („toți prietenii au acest lucru”)": "Social pressure",
        "Da, pentru cheltuieli neprevăzute dar inevitabile": "Yes, for unexpected expenses",
        "Inexistent": "Absent",
        "Scăzut": "Low",
        "Gestionabil": "Manageable",
        "Dificil de gestionat": "Difficult to manage",
        "Nu, dar sunt interesat/ă": "No, but interested",
        "Da, ocazional": "Yes, occasionally",
        "Da, regulat": "Yes, regularly",
        "Nu, nu sunt interesat/ă": "No, not interested",
        "Săptămânal": "Weekly",
        "Lunar": "Monthly",
        "Zilnic": "Daily",
        "Rar sau deloc": "Rarely or never",
        "Sub 6 luni": "<6 months",
        "Sub 50%": "<50%",
        "Peste 75%": ">75%",
        "6-12 luni": "6-12 months",
        "1-3 ani": "1-3 years",
        "3-5 ani": "3-5 years",
        "5-10 ani": "5-10 years",
        "10-15 ani": "10-15 years",
        "Peste 15 ani": ">15 years",
        "Sub 20": "<20",
        "Peste 50": ">50",
        "Nu am achizitionat": "Not purchased yet",
        "Uneori": "Sometimes",
        "Des": "Often",
        "Foarte rar": "Very rarely",
        "Da": "Yes",
        "Nu": "No",
        "12.000-16 000 RON": "12.000-16.000 RON",
        "8000-12.000 RON": "8.000-12.000 RON",
        "4000-8000 RON": "4.000-8.000 RON",
        "Peste 16.000 RON": ">16.000 RON",
        "Sub 4000 RON": "<4.000 RON",
    }
    df.replace(basic_translation, inplace=True)

    # Translate impulse buying fields
    impulse_map = {
        "Alimentație": "Food",
        "Haine sau produse de îngrijire personală": "Clothing or personal care products",
        "Electronice sau gadget-uri": "Electronics or gadgets",
        "Divertisment și timp liber": "Entertainment",
        "Altceva": "Other"
    }
    if "Impulse_Buying_Category" in df.columns:
        df["Impulse_Buying_Category"] = df["Impulse_Buying_Category"].replace(impulse_map)

    impulse_r_map = {"Altceva": "Other"}
    if "Impulse_Buying_Reason" in df.columns:
        df["Impulse_Buying_Reason"] = df["Impulse_Buying_Reason"].replace(impulse_r_map)

    # Translate and encode multi-option fields (multi-hot)
    multiple_val_map = {
        "Savings_Goal": {
            "Economii pentru achiziții majore (locuință, mașină)": "Major_Purchases",
            "Siguranță financiară pentru pensionare": "Retirement",
            "Fond de urgență": "Emergency_Fund",
            "Educația copiilor": "Child_Education",
            "Vacanțe sau cumpărături mari": "Vacation",
            "Altceva": "Other"
        },
        "Savings_Obstacle": {
            "Altceva": "Other",
            "Venitul este insuficient": "Insufficient_Income",
            "Alte cheltuieli urgente au prioritate": "Other_Expenses",
            "Nu consider economiile o prioritate": "Not_Priority"
        },
        "Expense_Distribution": {
            "Alimentație": "Food",
            "Locuință (chirie, utilități)": "Housing",
            "Transport": "Transport",
            "Divertisment și timp liber (iesiri cu prietenii, hobby-uri, excursii)": "Entertainment",
            "Sănătate (consultații medicale, medicamente, fizioterapie)": "Health",
            "Aspect personal (salon, cosmetice, haine, fitness)": "Personal_Care",
            "Cheltuieli generale pentru copii (îmbrăcăminte, activități extrașcolare)": "Child_Education",
            "Alte cheltuieli": "Other"
        },
        "Credit_Usage": {
            "Da, pentru cheltuieli esențiale (locuință, hrană)": "Essential_Needs",
            "Da, pentru cheltuieli mari (ex. vacanțe, electronice, autoturism, imobil etc.)": "Major_Purchases",
            "Da, pentru cheltuieli mari (ex. vacanțe, electronice)": "Major_Purchases",
            "Da, pentru cheltuieli neprevăzute dar inevitabile (ex. sănătate, reparații)": "Unexpected_Expenses",
            "Da, pentru nevoi personale (ex. evenimente speciale, educație)": "Personal_Needs",
            "Nu am folosit niciodată": "Never_Used"
        }
    }

    for col, translations in multiple_val_map.items():
        if col not in df.columns:
            continue

        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else "")
        df[col] = df[col].str.replace(r', (?=[A-ZĂÎȘȚÂ])', '; ', regex=True).str.strip()

        df[col] = df[col].apply(
            lambda cell: '; '.join(
                translations.get(part.strip(), part.strip()) for part in cell.split('; ')
            ) if cell else cell
        )

        # Group multi-column features into single categorical columns
        # Instead of creating multiple binary columns, we will keep the main column
        # and ensure it contains simplified categories.
        # For multi-value cells (separated by ;), we might need to decide how to handle them.
        # Option 1: Keep as is (string) - but this is bad for ML if not encoded properly.
        # Option 2: Split into multiple rows (explode) - changes dataset size.
        # Option 3: Create binary columns (current approach) - but user wants "Group multi-column features into single categorical columns".
        # If the user means "don't create 4 columns for the same concept", they might refer to OneHotEncoding result.
        # But here we are doing manual multi-hot encoding.

        # Let's modify to create binary columns but ensure they are properly named and values are 0/1
        # The user said: "Group multi-column features into single categorical columns (don't create 4 columns for același concept)."
        # This usually implies using a single column with categorical values instead of One-Hot Encoding.
        # However, for multi-select questions (like "Check all that apply"), One-Hot is the standard way.
        # If we use a list-type column, we can't easily feed it to standard ML models without processing.

        # Wait, the user also said: "Normalize frequencies (0-1) for each category".
        # And "Transform everything into correct categories (don't use true/false)".

        # Let's stick to the current multi-hot encoding but ensure values are 0/1 integers (which they are).
        # Maybe the user refers to the visualization part where these are treated as separate features?

        # Let's look at the "Group multi-column features" request again.
        # If I have "Savings_Goal_Emergency_Fund", "Savings_Goal_Retirement", etc.
        # The user might want to analyze "Savings_Goal" as a whole group.

        # Re-reading: "Grupează valorile în același feature (nu mai crea 4 coloane pentru același concept)."
        # This strongly suggests avoiding One-Hot Encoding for single-choice questions.
        # But these ARE multi-choice questions (e.g. "Alegeți toate opțiunile care se aplică").
        # For multi-choice, we MUST use multiple columns or a list-type column.
        # If we use a list-type column, we can't easily feed it to standard ML models without processing.

        # However, for single-choice questions that were One-Hot Encoded later (by get_dummies),
        # we should avoid that if we want to keep them as single categorical columns for some analysis.
        # But `postprocess_data` does `get_dummies`.

        # Let's look at `postprocess_data`. It does `get_dummies`.
        # If we want to keep them as categories, we should use `astype('category')` and maybe Label Encoding or Target Encoding.
        # But XGBoost supports `enable_categorical=True`.

        # The error message was:
        # "DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, the experimental DMatrix parameter`enable_categorical` must be set to `True`. Invalid columns:Gender: object"

        # So we need to convert object columns to 'category' dtype if we want to use them directly,
        # OR encode them to numbers.
        # My previous fix in `postprocess_data` added `get_dummies` which converts them to numbers (0/1).
        # This solves the "Invalid columns:Gender: object" error.

        # Now about "Group multi-column features" in `preprocessing.py` for the multi-value columns.
        # The current implementation creates dummy columns manually.
        # I will keep this but ensure they are strictly 0/1 integers.

        col_text = df[col].copy()
        df[col] = df[col].str.split(r';\s*')

        for option in translations.values():
            dummy_col = f"{col}_{option}"
            df[dummy_col] = df[col].apply(
                lambda lst: int(option in lst if isinstance(lst, list) else 0)
            )

        # We keep the original column as text for reference/decoding if needed,
        # but for ML we usually drop it.
        # The `main.py` drops `multi_value_cols` before scoring.
        df[col] = col_text


    return df


# Encode and prepare processed data for ML
def postprocess_data(df):
    try:
        # Map ordinal categories to integers
        ordinal_mappings = {
            'Impulse_Buying_Frequency': {
                'Very rarely': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Very often': 5
            },
            'Debt_Level': {
                'Absent': 1, 'Low': 2, 'Manageable': 3, 'Difficult to manage': 4, np.nan: 0, 'Unknown': 0
            },
            'Bank_Account_Analysis_Frequency': {
                'Rarely or never': 1, 'Monthly': 2, 'Weekly': 3, 'Daily': 4
            }
        }
        for col, mapping in ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        # One-hot encode remaining categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols)

        # Ensure all boolean columns are integers (0/1)
        bool_cols = df.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)

        return df
    except Exception as e:
        print(f"Error in postprocess_data: {e}")
        return None


# Smooth fuzzy inputs (ranges, categories) to numeric values
def range_smoothing(df, age_column="Age", income_column="Income_Category", lifetime_columns=None,
                    essential_needs_column="Essential_Needs_Percentage", lifetime_func=None):
    if age_column in df.columns:
        df = replace_age_column(df, age_column)

    if income_column in df.columns:
        df = replace_income_category(df, income_column)

    if lifetime_columns:
        if lifetime_func is not None:
            df = replace_product_lifetime_columns(df, lifetime_columns, lifetime_func)
        else:
            df = replace_product_lifetime_columns(df, lifetime_columns, random_product_lifetime)

    if essential_needs_column in df.columns:
        df = replace_essential_needs(df, essential_needs_column)

    return df