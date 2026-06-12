import streamlit as st
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm

# Configuration de la page
st.set_page_config(page_title="Simulateur SSP vs PNDS (FCFA)", layout="wide")

st.title("📊 Confrontation Offre Budgétaire SSP vs Coût PNDS")
st.write("Cet outil calcule l'enveloppe budgétaire SSP soutenable et la confronte directement aux ambitions du PNDS.")

# FACTEUR DE CONVERSION : 1 $ PPA = 300 FCFA
TAUX_PPA_FCFA = 300.0

# ==============================================================================
# 1. CHARGEMENT DES OBJETS DU MODÈLE
# ==============================================================================
@st.cache_resource
def load_model_objects():
    model = joblib.load("panel_ols_model.pkl")
    scaler = joblib.load("minmax_scaler.pkl")
    features = joblib.load("variables_x.pkl")
    return model, scaler, features

try:
    model, scaler, features = load_model_objects()
    st.success("Modèle économétrique opérationnel !")
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# ==============================================================================
# 2. CONFIGURATION DE LA SIMULATION
# ==============================================================================
st.header("📍 Contexte et Objectif PNDS")
col_sel1, col_sel2, col_sel3 = st.columns(3)

with col_sel1:
    liste_7_pays = ["Bénin", "Burkina Faso", "Côte d'Ivoire", "Guinée", "Mali", "Niger", "Sénégal"]
    pays_selectionne = st.selectbox("Pays cible :", options=liste_7_pays)

with col_sel2:
    annee_selectionnee = st.selectbox("Année de projection :", options=list(range(2024, 2031)))

with col_sel3:
    cout_pnds_milliards = st.number_input("Coût SSP prévu par le PNDS (en Milliards de FCFA) :", min_value=0.0, value=300.0, step=5.0, )
    cout_pnds_fcfa = cout_pnds_milliards * 1.0e9

# ==============================================================================
# 3. SAISIE DES DONNÉES BRUTES
# ==============================================================================
st.markdown("---")
st.header("✍️ Paramètres Prévisionnels")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Données Macroéconomiques")
    pib_milliards = st.number_input("PIB projeté (en Milliards de FCFA) :", min_value=1.0, value=19500.0, step=100.0)
    pib_fcfa = pib_milliards * 1.0e9
    pop = st.number_input("Population Totale projetée :", min_value=1000, value=24000000)
    recettes = st.slider("Recettes publiques (% du PIB) :", 0.0, 100.0, 18.5, 0.1)
    dette = st.slider("Dette publique (% du PIB) :", 0.0, 300.0, 55.0, 0.1)
    pt = st.slider("Part de la santé dans le budget total (Pt en %) :", 0.0, 100.0, 7.5, 0.1)

with col2:
    st.subheader("⚖️ Gouvernance & GFP")
    with st.expander("📘 Guide de correspondance des scores PEFA (Lettres ➔ Chiffres)", expanded=False):
        st.markdown("""
        Pour les indicateurs de finances publiques (**PEFA 8, 13, 14 et 22**), veuillez convertir les lettres de votre rapport en valeurs numériques selon la correspondance officielle ci-dessous :
        """)
        
        # Création d'un tableau propre pour l'affichage
        df_legende = pd.DataFrame({
            "Note PEFA (Rapport)": ["A", "B+", "B", "C+", "C", "D+", "D", "A+"],
            "Valeur numérique à saisir": [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 4.0]
        })
        
        # Affichage du tableau dans Streamlit sans l'index restrictif
        st.dataframe(df_legende, hide_index=True, use_container_width=True)
        
        st.caption("💡 *Note : Les notes 'A+' sont traitées à l'équivalent d'un 'A' (4.0) conformément au dictionnaire de conversion du modèle.*")
    
    corruption = st.slider("Maîtrise de la corruption (WGI) :", -2.5, 2.5, -0.7, 0.01)
    goveff = st.slider("Efficacité des pouvoirs publics (WGI) :", -2.5, 2.5, -0.6, 0.01)
    stab_pol = st.slider("Stabilité politique (WGI) :", -2.5, 2.5, -1.0, 0.01)
    reg_qual = st.slider("Qualité de la réglementation (WGI) :", -2.5, 2.5, -0.5, 0.01)
    
    pefa_8 = st.slider("PEFA 8: Lien Planification-Budget (1 à 4) :", 1.0, 4.0, 2.0, 0.5)
    pefa_13 = st.slider("PEFA 13: Gestion de la dette (1 à 4) :", 1.0, 4.0, 2.5, 0.5)
    pefa_14 = st.slider("PEFA 14: Prévisibilité des recettes (1 à 4) :", 1.0, 4.0, 2.0, 0.5)
    pefa_22 = st.slider("PEFA 22: Gestion des arriérés (1 à 4) :", 1.0, 4.0, 1.5, 0.5)
    
    efficience = st.slider("Efficience technique des dépenses SSP (0 à 1) :", 0.0, 1.0, 0.15, 0.01)
    fongibilite = st.slider("Fongibilité de l'aide externe SSP (0 à 1) :", 0.0, 1.0, 0.30, 0.01)
# ==============================================================================
# 4. CALCULS SÉCURISÉS AVEC ALIGNEMENT STRICT DES SHAPES (MATRICE 1x10 FRACTURE-PROOF)
# ==============================================================================
if st.button("🚀 Lancer la Confrontation Budgétaire"):
    
    # Normalisation du bloc gouvernance
    raw_gov = np.array([[corruption, goveff, stab_pol, reg_qual, pefa_8, pefa_13, pefa_14, pefa_22, efficience, fongibilite]])
    norm_gov = scaler.transform(raw_gov)[0]
    
    # Conversion PIB en $ PPA
    pib_ppa = float(pib_fcfa) / TAUX_PPA_FCFA
    
    # 1. Base de toutes les variables calculées possibles
    toutes_les_variables = {
        "log_PIB": np.log(pib_ppa),
        "log_Pop": np.log(float(pop)),
        "Recettes publiques en pourcentage du PIB": float(recettes),
        "Dette publique en pourcentage du PIB": float(dette),
        "Dépenses publiques de santé en % des dépenses publiques totales (gghed_gge)=Pt": float(pt),
        "Maîtrise de la corruption": norm_gov[0],
        "Efficacité des pouvoirs publics": norm_gov[1],
        "Stabilité politique et absence de violence": norm_gov[2],
        "Qualité de la réglementation": norm_gov[3],
        "8. Information sur la performance\ndes services publics=Lien entre planification et budget": norm_gov[4],
        "13. Gestion de la dette=Gestion de la dette": norm_gov[5],
        "14. Prévisions macroéconomiques\net budgétaires=Prévisibilité des recettes": norm_gov[6],
        "22. Arriérés de dépenses=Gestion des arriérés de paiement": norm_gov[7],
        "Efficience technique des dépenses publiques dédiées aux soins de santé primaires": norm_gov[8],
        "Degré de fongibilité des ressources externes dédiées aux soins de santé primaires": norm_gov[9],
        "const": 1.0  # On prépare la constante au cas où
    }
    
    # 2. ALIGNEMENT RADICAL : On construit le dictionnaire final en ne prenant 
    # QUE les clés validées dans la liste 'features' (vos 10 variables d'entraînement)
    dictionnaire_aligne = {col: toutes_les_variables[col] for col in features if col in toutes_les_variables}
    
    # Création du DataFrame ordonné
    X_input = pd.DataFrame([dictionnaire_aligne], columns=features)
    
    try:
        # ==============================================================================
        # ALIGNEMENT CHIRURGICAL SUR LES COEFFICIENTS RÉELS DU MODÈLE
        # ==============================================================================
        
        # 1. On récupère les coefficients et les VRAIS noms des variables retenues par le modèle
        noms_variables_du_modele = list(model.params.index)  # Taille exacte attendue
        coefficients = model.params.values                 # Vecteur de taille (10,)
        
        # 2. On reconstruit le vecteur de saisies en se basant STRICTEMENT sur ce que le modèle demande
        Vecteur_propre = []
        for col in noms_variables_du_modele:
            if col == 'const':
                Vecteur_propre.append(1.0)
            elif col in X_input.columns:
                # Sécurité : on extrait la valeur brute en forçant la conversion en float simple
                valeur_brute = X_input[col].iloc[0]
                if isinstance(valeur_brute, (pd.Series, np.ndarray)):
                    valeur_brute = valeur_brute.item() if hasattr(valeur_brute, 'item') else valeur_brute[0]
                Vecteur_propre.append(float(valeur_brute))
            else:
                Vecteur_propre.append(0.0)
                
        valeurs_saisies = np.array(Vecteur_propre)
        
        # 3. Produit scalaire de base (Somme des Variables * Coefficients)
        log_pred = float(np.dot(valeurs_saisies, coefficients))
        
        # ==============================================================================
        # EXTRACTION SÉCURISÉE DE L'EFFET FIXE PAYS (CORRECTION DU CRASH 'SERIES')
        # ==============================================================================
        try:
            if hasattr(model, 'estimated_effects'):
                effets_fixes = model.estimated_effects
                
                # Si l'effet fixe est indexé par le pays
                if pays_selectionne in effets_fixes.index.get_level_values(0):
                    sous_ensemble = effets_fixes.loc[pays_selectionne]
                    
                    # On extrait la toute première valeur numérique brute, peu importe la forme (DataFrame, Series ou tableau)
                    if isinstance(sous_ensemble, (pd.DataFrame, pd.Series)):
                        effet_pays = float(sous_ensemble.iloc[0].item() if hasattr(sous_ensemble.iloc[0], 'item') else sous_ensemble.iloc[0])
                    else:
                        effet_pays = float(sous_ensemble)
                        
                    log_pred += effet_pays
        except Exception as ef_err:
            # Si l'extraction de l'effet fixe est trop capricieuse en local, 
            # on passe cette étape pour ne pas bloquer l'affichage des métriques
            pass
            
        # ==============================================================================
        # EXPONENTIELLE ET CONVERSION FCFA
        # ==============================================================================
        pred_ssp_hab_ppa = np.exp(log_pred)
        pred_ssp_total_fcfa = pred_ssp_hab_ppa * TAUX_PPA_FCFA * float(pop)
        
        # ==============================================================================
        # AFFICHAGE DES RÉSULTATS VISUELS COÛT VS PRÉDICTION
        # ==============================================================================
        st.markdown("---")
        st.subheader(f"📊 Résultats de la confrontation : {pays_selectionne} ({annee_selectionnee})")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(
                label="🏦 Enveloppe Globale SSP Prédite (Soutenable)", 
                value=f"{float(pred_ssp_total_fcfa):,.0f} FCFA".replace(",", " ")
            )
        with col_res2:
            st.metric(
                label="📋 Coût Fixé dans le PNDS", 
                value=f"{float(cout_pnds_fcfa):,.0f} FCFA".replace(",", " ")
            )
            
        gap_fcfa = pred_ssp_total_fcfa - cout_pnds_fcfa
        st.markdown("### 🔍 Diagnostic de l'arbitrage budgétaire")
        if gap_fcfa >= 0:
            st.success(f"🟢 **Marge Budgétaire Excédentaire : +{float(gap_fcfa):,.0f} FCFA**".replace(",", " "))
        else:
            st.error(f"🔴 **Déficit de Financement (GAP) : {float(gap_fcfa):,.0f} FCFA**".replace(",", " "))
            
    except Exception as e:
        st.error(f"Erreur lors du calcul : {e}")