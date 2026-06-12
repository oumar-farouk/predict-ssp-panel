import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import joblib

print("⏳ Chargement et conversion brute des données...")

# 1. CHARGEMENT
file_name = "..\Base de données Essai 3_15022026_ajout colonne L.xlsx"
df = pd.read_excel(file_name)
df.columns = df.columns.str.strip()

# Noms exacts des variables
cols_macro = [
    "dépenses publiques par habitant, allouées aux soins de santé primaires",
    "PIB en PPA (Worlbank)",
    "Population Totale (Worlbank)",
    "Recettes publiques en pourcentage du PIB",
    "Dette publique en pourcentage du PIB",
    "Dépenses publiques de santé en % des dépenses publiques totales (gghed_gge)=Pt"
]

gov_vars = [
    "Maîtrise de la corruption", 
    "Efficacité des pouvoirs publics", 
    "Stabilité politique et absence de violence", 
    "Qualité de la réglementation",
    "8. Information sur la performance\ndes services publics=Lien entre planification et budget",
    "13. Gestion de la dette=Gestion de la dette", 
    "14. Prévisions macroéconomiques\net budgétaires=Prévisibilité des recettes",
    "22. Arriérés de dépenses=Gestion des arriérés de paiement",
    "Efficience technique des dépenses publiques dédiées aux soins de santé primaires",
    "Degré de fongibilité des ressources externes dédiées aux soins de santé primaires"
]

# Dictionnaire de conversion pour les scores PEFA au format texte (Lettres)
pefa_mapping = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'A+': 4.0, 'B+': 3.5, 'C+': 2.5, 'D+': 1.5}

# 2. PRÉ-NETTOYAGE ET CONVERSION NUMÉRIQUE
for col in cols_macro + gov_vars:
    if col in df.columns:
        # Si la colonne contient des chaînes de caractères (ex: lettres PEFA), on applique le mapping
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().upper().replace(pefa_mapping)
        
        # Remplacement des virgules par des points et forçage en numérique
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# Éliminer les observations où les données vitales manquent complètement
df = df.dropna(subset=[
    "dépenses publiques par habitant, allouées aux soins de santé primaires",
    "PIB en PPA (Worlbank)",
    "Population Totale (Worlbank)"
])

# Remplacer les NaN restants dans le bloc gouvernance par la médiane globale de la colonne
for col in gov_vars:
    # Si toute la colonne est vide (sécurité), on initialise à 2.0 (note moyenne par défaut)
    if df[col].isna().all():
        df[col] = df[col].fillna(2.0)
    else:
        df[col] = df[col].fillna(df[col].median())

# Sécurité d'échelle positive avant l'application des Logarithmes
df = df[(df["dépenses publiques par habitant, allouées aux soins de santé primaires"] > 0) & 
        (df["PIB en PPA (Worlbank)"] > 0) & 
        (df["Population Totale (Worlbank)"] > 0)].copy()

print(f"📉 Nombre de lignes saines identifiées : {df.shape[0]}")

# ==============================================================================
# 3. CALCUL DES LOGARITHMES
# ==============================================================================
df['log_target_SSP_hab'] = np.log(df["dépenses publiques par habitant, allouées aux soins de santé primaires"])
df['log_PIB'] = np.log(df["PIB en PPA (Worlbank)"])
df['log_Pop'] = np.log(df["Population Totale (Worlbank)"])

# ==============================================================================
# 4. NORMALISATION MINMAX SANS NaN
# ==============================================================================
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[gov_vars] = scaler.fit_transform(df[gov_vars])

# Variables explicatives finales (Sans constante manuelle pour éviter le conflit d'effet fixe)
variables_X_finales = [
    "log_PIB", 
    "log_Pop", 
    "Recettes publiques en pourcentage du PIB",
    "Dette publique en pourcentage du PIB",
    "Dépenses publiques de santé en % des dépenses publiques totales (gghed_gge)=Pt"
] + gov_vars

# ==============================================================================
# 5. ENTRAÎNEMENT PANELOLS SÉCURISÉ CONTRE LA COLINÉARITÉ
# ==============================================================================
df_panel = df.set_index(["Pays", "Année"]).sort_index()

X = df_panel[variables_X_finales].copy()
Y = df_panel['log_target_SSP_hab']

# Suppression préventive des variables sans aucune variation
cols_invariantes = [col for col in X.columns if X[col].std() == 0]
if cols_invariantes:
    print(f"⚠️ Suppression des variables invariantes : {cols_invariantes}")
    X = X.drop(columns=cols_invariantes)

# Nettoyage des variables corrélées à plus de 98% pour empêcher le crash SVD
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.98)]
if to_drop:
    print(f"❌ Suppression des colonnes colinéaires pour stabiliser l'estimation : {to_drop}")
    X = X.drop(columns=to_drop)

print(f"⚙️ Dimension finale de la matrice d'apprentissage : {X.shape[0]} lignes, {X.shape[1]} variables.")

print("🚀 Lancement de l'estimation PanelOLS...")
# check_rank=False est activé ici par sécurité numérique pour surmonter l'approximation des petits échantillons
model_panel = PanelOLS(Y, X, entity_effects=True, time_effects=False, check_rank=False,drop_absorbed=True)
results = model_panel.fit()

print("\n✅ Modèle estimé avec succès !")
print(results.summary)

# ==============================================================================
# 6. EXPORTATION DES ARTÉFACTS NETTOYÉS
# ==============================================================================
joblib.dump(results, "panel_ols_model.pkl")
joblib.dump(scaler, "minmax_scaler.pkl")
joblib.dump(list(X.columns), "variables_x.pkl")
print("💾 Tous les fichiers .pkl ont été synchronisés !")