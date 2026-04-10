# Détection du Cancer Pulmonaire — TP Noté M2 ESIC

Ce projet a été réalisé dans le cadre du TP noté de M2 ESIC en Intelligence Artificielle.
L'objectif est de détecter le cancer pulmonaire à partir de données cliniques et de radios thoraciques,
en combinant un modèle de Machine Learning et un réseau de neurones convolutif (CNN).

---

## Ce que fait l'application

L'application Streamlit permet de :
- Saisir les données cliniques d'un patient (âge, tabagisme, symptômes, etc.)
- Charger une radio thoracique
- Obtenir une prédiction du risque de malignité (Modèle 1)
- Obtenir une décision finale sur la présence probable d'un cancer (Modèle 2 — fusion)

---

## Comment lancer l'application en local

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement sur : http://localhost:8501

---

## Application déployée en ligne

L'application est accessible publiquement ici :

🔗 https://tp-note-ml-j6cghatkehbfl3aebq4mco.streamlit.app

---

## Fichiers du projet

```
├── app.py                   → Interface Streamlit
├── requirements.txt         → Dépendances Python
├── model_lr.pkl             → Modèle 1 : régression logistique
├── model_cnn.h5             → Modèle 2 : réseau de neurones CNN
├── model_fusion.pkl         → Modèle fusion (décision finale)
├── patients_cancer_poumon.csv → Données tabulaires
├── TP_Noté.ipynb            → Notebook complet
├── rapport_tp.docx          → Rapport du projet
└── README.md                → Ce fichier
```

---

## Comment générer les modèles

Si les fichiers .pkl et .h5 sont absents, il faut exécuter le notebook `TP_Noté.ipynb` en entier.
Les cellules de sauvegarde à la fin du notebook génèrent automatiquement les fichiers :

```python
import joblib

# Modèle 1
joblib.dump(model_lr, "model_lr.pkl")

# CNN
model_cnn.save("model_cnn.h5")

# Modèle fusion
joblib.dump(model, "model_fusion.pkl")
```

---

## Technologies utilisées

- Python 3.11
- Streamlit — interface utilisateur
- Scikit-learn — modèles tabulaires (régression logistique, Random Forest, Gradient Boosting)
- TensorFlow / Keras — réseau de neurones CNN
- Joblib — sauvegarde des modèles
- Pandas / NumPy — traitement des données
- Pillow — traitement des images

---

## Déploiement

L'application a été déployée sur **Streamlit Cloud** en connectant ce dépôt GitHub.
Le déploiement se met à jour automatiquement à chaque modification du code.

> Note : TensorFlow n'étant pas compatible avec Python 3.14 (Streamlit Cloud),
> le CNN est simulé en production. Les modèles tabulaire et fusion fonctionnent
> pleinement en ligne. En local avec Python 3.11, tout fonctionne intégralement.

---

