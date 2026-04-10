# ============================================================
#  APP.PY — Interface Streamlit pour la détection du cancer
#  pulmonaire par Machine Learning + Deep Learning
#
#  Pour lancer l'application, ouvre un terminal et tape :
#      streamlit run app.py
#
#  L'application s'ouvre automatiquement sur :
#      http://localhost:8501
# ============================================================


# ── IMPORTS ─────────────────────────────────────────────────
import streamlit as st   # Streamlit = crée l'interface web
import numpy as np       # NumPy = calculs sur tableaux de nombres
import pandas as pd      # Pandas = manipulation de tableaux de données
import joblib            # Joblib = chargement des modèles .pkl
import os                # Os = vérification de l'existence des fichiers
from PIL import Image    # PIL = ouverture et traitement des images


# ── CONFIGURATION DE LA PAGE ─────────────────────────────────
# DOIT être la toute première commande Streamlit du fichier

st.set_page_config(
    page_title="Détection Cancer Pulmonaire",
    page_icon="🫁",
    layout="wide",
)


# ── STYLE CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1a2332; }
    [data-testid="stSidebar"] * { color: white !important; }

    .resultat-vert {
        background-color: #d4edda; color: #155724;
        border: 2px solid #28a745; padding: 20px;
        border-radius: 10px; font-size: 1.3rem;
        font-weight: bold; text-align: center; margin: 10px 0;
    }
    .resultat-orange {
        background-color: #fff3cd; color: #856404;
        border: 2px solid #ffc107; padding: 20px;
        border-radius: 10px; font-size: 1.3rem;
        font-weight: bold; text-align: center; margin: 10px 0;
    }
    .resultat-rouge {
        background-color: #f8d7da; color: #721c24;
        border: 2px solid #dc3545; padding: 20px;
        border-radius: 10px; font-size: 1.3rem;
        font-weight: bold; text-align: center; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── CHARGEMENT DES MODÈLES ───────────────────────────────────
# @st.cache_resource = charge UNE SEULE FOIS au démarrage
# et garde en mémoire pour ne pas recharger à chaque clic

@st.cache_resource
def charger_modeles():
    modeles = {
        "tabulaire": None,  # model_lr.pkl  → Modèle 1
        "cnn":       None,  # model_cnn.h5  → Modèle 2 version 1
        "fusion":    None,  # model_fusion.pkl → Modèle 2 version 2
    }

    # Modèle 1 : régression logistique sur données tabulaires
    if os.path.exists("model_lr.pkl"):
        modeles["tabulaire"] = joblib.load("model_lr.pkl")

   # Modèle 2 version 1 : CNN sur images
    if os.path.exists("model_cnn.h5"):
        try:
            import importlib
            tf_spec = importlib.util.find_spec("tensorflow")
            if tf_spec is not None:
                import tensorflow as tf
                modeles["cnn"] = tf.keras.models.load_model("model_cnn.h5")
        except Exception:
            pass  # CNN non disponible, simulation activée
    # Modèle 2 version 2 : fusion (LogisticRegression sur [pred_cnn, pred_reglog])
    if os.path.exists("model_fusion.pkl"):
        modeles["fusion"] = joblib.load("model_fusion.pkl")

    return modeles


modeles = charger_modeles()


# ── BARRE LATÉRALE ───────────────────────────────────────────
st.sidebar.title("🫁 Menu")
st.sidebar.markdown("---")

# Seulement 2 pages
page = st.sidebar.radio(
    "Choisir une page :",
    ["🏠 Accueil", "🔬 Faire une prédiction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### État des modèles")
st.sidebar.markdown("✅ Modèle 1 (tabulaire)" if modeles["tabulaire"] else "❌ model_lr.pkl manquant")
st.sidebar.markdown("✅ Modèle 2 (CNN)"        if modeles["cnn"]       else "❌ model_cnn.h5 manquant")
st.sidebar.markdown("✅ Modèle fusion"          if modeles["fusion"]    else "❌ model_fusion.pkl manquant")
st.sidebar.markdown("---")
st.sidebar.caption("M2 ESIC — IA & ML · 2025-2026")


# ════════════════════════════════════════════════════════════
#  PAGE 1 : ACCUEIL
# ════════════════════════════════════════════════════════════

if page == "🏠 Accueil":

    st.title("🫁 Détection du Cancer Pulmonaire")
    st.subheader("Système d'aide à la décision médicale par Intelligence Artificielle")
    st.markdown("---")

    col_gauche, col_droite = st.columns(2)

    with col_gauche:
        st.markdown("### Comment fonctionne ce système ?")
        st.markdown("""
        Ce système utilise **deux modèles d'IA** travaillant ensemble :

        **🔵 Modèle 1 — Données cliniques**
        Analyse les informations médicales du patient (âge, tabagisme,
        taille du nodule, symptômes...) et calcule un **score de risque**
        de malignité sur 3 niveaux : faible / intermédiaire / élevé.

        **🟣 Modèle 2 — Radio + Fusion**
        Analyse la radio thoracique avec un réseau de neurones (CNN),
        puis combine ce résultat avec le Modèle 1 pour donner
        une **décision finale** : cancer probable ou non.
        """)

    with col_droite:
        st.markdown("### Pipeline de décision")
        st.code("""
Données cliniques (15 variables)
         ↓
   Modèle 1 — Régression Logistique
         ↓
   Prédiction risque : 0, 1 ou 2
         ↓
         + ←——— Radio thoracique → CNN
         ↓
   Modèle Fusion (LogisticRegression)
   Entrées : [pred_cnn, pred_reglog]
         ↓
   Décision finale :
   0 = Non probable / 1 = Probable
        """, language=None)

    st.markdown("---")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.info("**📋 Étape 1**\n\nRemplis les données cliniques du patient dans le formulaire.")
    with e2:
        st.info("**🖼️ Étape 2**\n\nCharge la radio thoracique (JPG ou PNG).")
    with e3:
        st.info("**🔬 Étape 3**\n\nClique sur **'Lancer l'analyse'** et consulte les résultats.")


# ════════════════════════════════════════════════════════════
#  PAGE 2 : FAIRE UNE PRÉDICTION
# ════════════════════════════════════════════════════════════

elif page == "🔬 Faire une prédiction":

    st.title("🔬 Analyse d'un patient")
    st.markdown("---")

    # ── Formulaire données cliniques ─────────────────────────
    with st.expander("📋 Étape 1 — Données cliniques", expanded=True):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Général**")
            age            = st.number_input("Âge", min_value=18, max_value=100, value=55)
            sexe           = st.selectbox("Sexe", ["Masculin", "Féminin"])
            sexe_val       = 1 if sexe == "Masculin" else 0
            tabagisme      = st.number_input("Tabagisme (paquets/an)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
            antecedent     = st.selectbox("Antécédent familial", ["Non", "Oui"])
            antecedent_val = 1 if antecedent == "Oui" else 0

        with col2:
            st.markdown("**Nodule**")
            presence     = st.selectbox("Présence d'un nodule", ["Oui", "Non"])
            presence_val = 1 if presence == "Oui" else 0
            subtilite    = st.slider("Subtilité (1=subtil, 5=évident)", 1, 5, 3)
            taille       = st.number_input("Taille du nodule (px)", min_value=0, max_value=50, value=10)
            x_pos        = st.number_input("Position X normalisée (0 à 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            y_pos        = st.number_input("Position Y normalisée (0 à 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        with col3:
            st.markdown("**Symptômes**")
            toux        = st.selectbox("Toux chronique", ["Non", "Oui"])
            toux_val    = 1 if toux == "Oui" else 0
            dyspnee     = st.selectbox("Dyspnée (essoufflement)", ["Non", "Oui"])
            dyspnee_val = 1 if dyspnee == "Oui" else 0
            douleur     = st.selectbox("Douleur thoracique", ["Non", "Oui"])
            douleur_val = 1 if douleur == "Oui" else 0
            perte_poids = st.selectbox("Perte de poids", ["Non", "Oui"])
            perte_val   = 1 if perte_poids == "Oui" else 0
            spo2        = st.number_input("SpO2 (%)", min_value=70, max_value=100, value=95)

    # ── Upload radio ─────────────────────────────────────────
    with st.expander("🖼️ Étape 2 — Radio thoracique", expanded=True):
        fichier_image = st.file_uploader("Charger la radio (JPG ou PNG)", type=["jpg", "jpeg", "png"])
        if fichier_image is not None:
            image = Image.open(fichier_image)
            st.image(image, caption="Radio chargée ✅", width=300)

    # ── Bouton prédiction ─────────────────────────────────────
    st.markdown("---")
    bouton = st.button("🔬 Lancer l'analyse", use_container_width=True, type="primary")

    if bouton:
        st.markdown("---")
        st.markdown("## 📊 Résultats")

        # Colonnes utilisées lors de l'entraînement du Modèle 1
        # (correspond exactement au X = data.drop(...) du notebook)
        # cancer_image est incluse car elle n'a pas été retirée lors de l'entraînement
        noms_colonnes = [
            "age", "sexe_masculin", "presence_nodule", "subtilite_nodule",
            "taille_nodule_px", "x_nodule_norm", "y_nodule_norm",
            "tabagisme_paquets_annee", "toux_chronique", "dyspnee",
            "douleur_thoracique", "perte_poids", "spo2", "antecedent_familial",
            "cancer_image",  # valeur 0 par défaut car inconnue en production
        ]

        valeurs = np.array([[
            age, sexe_val, presence_val, subtilite, taille,
            x_pos, y_pos, tabagisme, toux_val, dyspnee_val,
            douleur_val, perte_val, spo2, antecedent_val,
            0  # cancer_image = 0
        ]])

        X_patient = pd.DataFrame(valeurs, columns=noms_colonnes)

        col_res1, col_res2 = st.columns(2)

        # ── Résultat Modèle 1 ────────────────────────────────
        with col_res1:
            st.markdown("### 🔵 Modèle 1 — Risque de malignité")
            st.caption("Analyse des données cliniques")

            if modeles["tabulaire"] is not None:
                risque_predit = int(modeles["tabulaire"].predict(X_patient)[0])
                probas_risque = modeles["tabulaire"].predict_proba(X_patient)[0]
            else:
                st.warning("⚠️ model_lr.pkl non trouvé → résultat simulé")
                nb_sym = toux_val + dyspnee_val + douleur_val + perte_val
                if tabagisme > 40 or (nb_sym >= 3 and antecedent_val == 1):
                    risque_predit, probas_risque = 2, np.array([0.05, 0.20, 0.75])
                elif tabagisme > 15 or nb_sym >= 2:
                    risque_predit, probas_risque = 1, np.array([0.20, 0.60, 0.20])
                else:
                    risque_predit, probas_risque = 0, np.array([0.75, 0.20, 0.05])

            infos = {
                0: ("🟢 Risque FAIBLE",        "resultat-vert"),
                1: ("🟡 Risque INTERMÉDIAIRE", "resultat-orange"),
                2: ("🔴 Risque ÉLEVÉ",         "resultat-rouge"),
            }
            label, css = infos[risque_predit]
            st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)

            st.markdown("**Probabilités par classe :**")
            for i, nom in enumerate(["Classe 0 — Faible", "Classe 1 — Intermédiaire", "Classe 2 — Élevé"]):
                if i < len(probas_risque):
                    st.progress(float(probas_risque[i]), text=f"{nom} : {probas_risque[i]*100:.1f}%")

        # ── Résultat Modèle 2 (CNN + Fusion) ─────────────────
        with col_res2:
            st.markdown("### 🟣 Modèle 2 — Prédiction finale")
            st.caption("Fusion image + données cliniques")

            if fichier_image is None:
                st.info("📌 Charge une radio thoracique pour obtenir la prédiction finale.")

            else:
                # Prédiction CNN sur l'image
                if modeles["cnn"] is not None:
                    img_gris  = image.convert("L").resize((128, 128))
                    img_array = np.array(img_gris) / 255.0
                    img_array = img_array.reshape(1, 128, 128, 1)
                    prob_cnn  = float(modeles["cnn"].predict(img_array)[0][0])
                    pred_cnn  = 1 if prob_cnn > 0.5 else 0
                else:
                    st.warning("⚠️ model_cnn.h5 non trouvé → CNN simulé")
                    prob_cnn = {0: 0.15, 1: 0.45, 2: 0.78}.get(risque_predit, 0.3)
                    pred_cnn = 1 if prob_cnn > 0.5 else 0

                # Fusion CNN + Modèle 1
                # Le modèle fusion attend exactement 2 colonnes : 'cnn' et 'reglog'
                # pred_reglog est ramené à 0 ou 1 (2 → 1) comme dans le notebook
                pred_reglog_harmonise = min(risque_predit, 1)

                X_fusion = pd.DataFrame(
                    [[pred_cnn, pred_reglog_harmonise]],
                    columns=["cnn", "reglog"]
                )

                if modeles["fusion"] is not None:
                    pred_finale = int(modeles["fusion"].predict(X_fusion)[0])
                    prob_finale = float(modeles["fusion"].predict_proba(X_fusion)[0][1])
                else:
                    st.warning("⚠️ model_fusion.pkl non trouvé → fusion simulée")
                    prob_finale = prob_cnn * 0.6 + (pred_reglog_harmonise * 0.4)
                    pred_finale = 1 if prob_finale > 0.5 else 0

                if pred_finale == 1:
                    st.markdown(
                        '<div class="resultat-rouge">🔴 Cancer pulmonaire : PROBABLE</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="resultat-vert">🟢 Cancer pulmonaire : NON PROBABLE</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("**Probabilités :**")
                st.progress(float(prob_finale),
                            text=f"Probabilité cancer (fusion) : {prob_finale*100:.1f}%")
                st.progress(float(prob_cnn),
                            text=f"Score CNN image seule : {prob_cnn*100:.1f}%")

        # ── Résumé patient ────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Résumé du patient")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Âge",       f"{age} ans")
        m2.metric("Tabagisme", f"{tabagisme} paq/an")
        m3.metric("SpO2",      f"{spo2}%")
        m4.metric("Symptômes", f"{toux_val+dyspnee_val+douleur_val+perte_val}/4")

        st.warning(
            "⚠️ **Avertissement** : Outil d'aide à la décision uniquement. "
            "Ne remplace pas l'avis d'un médecin qualifié."
        )
