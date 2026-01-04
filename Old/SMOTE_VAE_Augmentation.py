# DEPRECATED: This script is in the Old/ folder and should not be used.
# It causes experimental drift due to iterative augmentation practices.
# For reproducible experiments, use run_experiment.py with configs/*.yaml

import pandas as pd
import numpy as np
from keras.src.layers import Dropout, GaussianNoise
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


###############################
# Funcții de încărcare și preprocesare
###############################

def load_dataset(filename):
    df = pd.read_csv(filename)
    print("Coloane existente în dataset:")
    print(df.columns.tolist())
    print("\nValori lipsă pe coloane:")
    print(df.isna().sum()[df.isna().sum() > 0])
    return df


def convert_duration(s):
    if isinstance(s, str):
        if 'month' in s:
            return int(s.split()[0])
        elif 'year' in s:
            return int(s.split()[0]) * 12
    return np.nan


def preprocess_data(df):
    savings_goal_cols = ['Savings_Goal_Emergency_Fund', 'Savings_Goal_Major_Purchases',
                         'Savings_Goal_Child_Education', 'Savings_Goal_Vacation',
                         'Savings_Goal_Retirement', 'Savings_Goal_Other']

    savings_obstacle_cols = ['Savings_Obstacle_Insufficient_Income', 'Savings_Obstacle_Other_Expenses',
                             'Savings_Obstacle_Not_Priority', 'Savings_Obstacle_Other']

    expense_dist_cols = ['Expense_Distribution_Food', 'Expense_Distribution_Housing',
                         'Expense_Distribution_Transport', 'Expense_Distribution_Entertainment',
                         'Expense_Distribution_Health', 'Expense_Distribution_Personal_Care',
                         'Expense_Distribution_Child_Education', 'Expense_Distribution_Other']

    credit_cols = ['Credit_Essential_Needs', 'Credit_Major_Purchases',
                   'Credit_Unexpected_Expenses', 'Credit_Personal_Needs', 'Credit_Never_Used']

    passthrough_cols = savings_goal_cols + savings_obstacle_cols + expense_dist_cols + credit_cols

    # Coloanele pentru durate
    duration_cols = ['Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
                     'Product_Lifetime_Appliances', 'Product_Lifetime_Cars']
    for col in duration_cols:
        df[col] = df[col].apply(convert_duration)

    # Seturi de coloane
    numerical_cols = ['Age', 'Income_Category', 'Essential_Needs_Percentage'] + duration_cols
    ordinal_cols = ['Impulse_Buying_Frequency']
    nominal_cols = ['Family_Status', 'Gender', 'Financial_Attitude',
                    'Budget_Planning', 'Save_Money',
                    'Impulse_Buying_Category', 'Impulse_Buying_Reason',
                    'Debt_Level', 'Financial_Investments',
                    'Bank_Account_Analysis_Frequency']

    # Preprocesare pentru coloanele numerice
    for col in numerical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Preprocesare pentru coloanele categorice
    for col in ordinal_cols + nominal_cols:
        if col == 'Debt_Level':
            # Dacă lipsesc valori, le înlocuim cu "Absent"
            df[col] = df[col].fillna("Absent")
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        # Convertim în string
        df[col] = df[col].astype(str)

    # Definim preprocesorul cu transformări separate:
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('ord', OrdinalEncoder(categories=[['Very rarely', 'Rarely', 'Sometimes', 'Often', 'Very often']]), ordinal_cols),
        ('nom', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), nominal_cols),
        ('pass', 'passthrough', passthrough_cols)
    ])

    return preprocessor, numerical_cols, ordinal_cols, nominal_cols, passthrough_cols


###############################
# Funcții pentru SMOTE, VAE și augmentare
###############################

def apply_smote(X, y):
    if np.isnan(X).any():
        print("Atenție: S-au găsit NaN în datele preprocesate! Se înlocuiesc cu 0.")
        X = np.nan_to_num(X)
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)


class VAE:
    def __init__(self, input_dim, latent_dim=512): # [!] Crește latent_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder, self.decoder, self.vae = self._build_model()

    def _build_model(self):
        # STRATUL DE INTRARE
        inputs = Input(shape=(self.input_dim,))

        # ENCODER
        x = Dense(128, activation='relu')(inputs)
        x = GaussianNoise(0.9)(x) # Adăugăm zgomot
        x = Dropout(0.9)(x)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)

        # SAMPLING
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim))
            return z_mean + K.exp(2.6 * z_log_var) * epsilon  # [!] Crește coeficientul

        z = Lambda(sampling)([z_mean, z_log_var])

        # DECODER - CREAREA STRATURILOR DECODERULUI (le salvăm în variabile)
        dec_dense1 = Dense(256, activation='relu')
        dec_dense2 = Dense(128, activation='relu')
        dec_dense3 = Dense(64, activation='relu')
        dec_dense4 = Dense(32, activation='relu')# [!] Adaugă un strat suplimentar
        dec_out = Dense(self.input_dim)  # Fără activare finală sau cu activare adecvată pentru datele tale

        # Aplicăm straturile asupra lui z pentru modelul VAE:
        d = dec_dense1(z)
        d = dec_dense2(d)
        d = dec_dense3(d)
        d = dec_dense4(d)# [!] Adaugă stratul suplimentar
        outputs = dec_out(d)


        # MODELE
        vae = Model(inputs, outputs)
        reconstruction_loss = K.mean(K.square(inputs - outputs))
        kl_loss = -0.03 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) # [!] Scade KL loss
        vae.add_loss(K.mean(reconstruction_loss + 0.05 * kl_loss))
        vae.compile(optimizer=Adam(0.0001)) # [!] Scade rata de învățare

        # MODELUL DE ENCODER
        encoder = Model(inputs, z_mean)

        # MODELUL DE DECODER
        latent_inputs = Input(shape=(self.latent_dim,))
        h_decoded = dec_dense1(latent_inputs)
        h_decoded = dec_dense2(h_decoded)
        h_decoded = dec_dense3(h_decoded)
        h_decoded = dec_dense4(h_decoded)# [!] Adaugă stratul suplimentar
        _decoded = dec_out(h_decoded)
        decoder = Model(latent_inputs, _decoded)

        return encoder, decoder, vae

    def generate(self, n_samples, noise_level=0.09): # [!] Creste nivelul de zgomot
        z = np.random.normal(size=(n_samples, self.latent_dim))
        generated = self.decoder.predict(z)
        return generated + np.random.normal(0, noise_level, generated.shape)



def augment_class(X_class, n_samples):
    vae = VAE(X_class.shape[1])
    vae.vae.fit(X_class, epochs=1000, batch_size=8, verbose=0) # [!] Crește numărul de epoci
    return vae.generate(n_samples)


###############################
# Funcții pentru inverse transform (la formatul encoded final)
###############################

def full_inverse_transform(preprocessor, X_final, numerical_cols, ordinal_cols, nominal_cols, passthrough_cols):
    # Inversează transformările pentru fiecare segment
    n_num = len(numerical_cols)
    n_ord = len(ordinal_cols)
    n_nom = preprocessor.named_transformers_['nom'].get_feature_names_out(nominal_cols).shape[0]
    n_pass = len(passthrough_cols)

    # Extrage segmentele
    X_num = X_final[:, :n_num]
    X_ord = X_final[:, n_num:n_num + n_ord]
    X_nom = X_final[:, n_num + n_ord:n_num + n_ord + n_nom]
    X_pass = X_final[:, n_num + n_ord + n_nom:]  # Passthrough este ultimul segment

    # Inversează transformările
    X_num_inv = preprocessor.named_transformers_['num'].inverse_transform(X_num)
    X_ord_inv = preprocessor.named_transformers_['ord'].inverse_transform(X_ord)
    X_nom_inv = preprocessor.named_transformers_['nom'].inverse_transform(X_nom)  # Returnează coloanele nominale originale (string)

    # Combină toate segmentele
    return np.hstack([X_num_inv, X_ord_inv, X_nom_inv, X_pass])


###############################
# Funcția main – fluxul complet
###############################

def main():
    df = load_dataset('../DatasetOriginal.csv')
    preprocessor, numerical_cols, ordinal_cols, nominal_cols, passthrough_cols = preprocess_data(df)

    # Separați X de etichetă y
    X = df.drop('Behavior_Risk_Level', axis=1)
    y = df['Behavior_Risk_Level'].values  # Valorile binare (0/1)

    # Aplicăm SMOTE pentru echilibrare
    X_processed = preprocessor.fit_transform(X)
    X_smote, y_smote = apply_smote(X_processed, y)

    # Definire VAE cu dimensiunea corectă
    #input_dim = X_processed.shape[1]
    #vae = VAE(input_dim=input_dim)

    # Aplicăm VAE pe fiecare clasă pentru augmentare suplimentară
    X_class0 = X_smote[y_smote == 0]
    X_class1 = X_smote[y_smote == 1]

    # Generăm date sintetice: 50% din dimensiunea inițială a fiecărei clase
    synthetic_ratio = 5.5
    synthetic0 = augment_class(X_class0, int(len(X_class0) * synthetic_ratio))
    synthetic1 = augment_class(X_class1, int(len(X_class1) * synthetic_ratio))

    # Combinăm datele SMOTE + VAE
    X_final = np.vstack([X_smote, synthetic0, synthetic1])
    y_final = np.hstack([y_smote, np.zeros(len(synthetic0)), np.ones(len(synthetic1))])

    # Transformare inversă pentru a reveni la formatul original
    X_inversed = full_inverse_transform(
        preprocessor,
        X_final,
        numerical_cols,
        ordinal_cols,
        nominal_cols,
        passthrough_cols
    )

    # Construim coloanele finale
    nom_features = preprocessor.named_transformers_['nom'].get_feature_names_out(nominal_cols)
    final_columns = numerical_cols + ordinal_cols + nominal_cols + passthrough_cols

    # Verificare dimensiuni
    if X_inversed.shape[1] != len(final_columns):
        raise ValueError(f"Discrepanță coloane: {X_inversed.shape[1]} vs {len(final_columns)}")

    df_final = pd.DataFrame(X_inversed, columns=final_columns)

    # Post-procesare obligatorie
    # 1. Rotunjire coloane numerice
    df_final['Age'] = df_final['Age'].round().astype(int)
    df_final['Income_Category'] = (df_final['Income_Category']/100).round().astype(int)*100
    df_final['Essential_Needs_Percentage'] = (df_final['Essential_Needs_Percentage']/5).round().astype(int)*5

    # 2. Coloane binare (0/1)
    for col in passthrough_cols:
        df_final[col] = df_final[col].round().astype(int)

    # 3. Formatare durate
    duration_cols = ['Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
                    'Product_Lifetime_Appliances', 'Product_Lifetime_Cars']
    for col in duration_cols:
        df_final[col] = df_final[col].apply(lambda x: f"{int(x)} months" if x < 12 else f"{int(x//12)} years")

    # Adăugare etichete
    df_final['Behavior_Risk_Level'] = y_final
    print("\nDistribuția finală a claselor:")


    # Salvare
    df_final.to_csv('dataset_augmentat_SMOTE+VAE.csv', index=False)
    print("\nDatasetul augmentat a fost salvat cu succes!")

if __name__ == "__main__":
    main()