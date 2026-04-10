# 🚀 Guide de Déploiement — Détection Cancer Pulmonaire

## Prérequis

```bash
pip install -r requirements.txt
```

## Lancement en local

```bash
streamlit run app.py
```
Accès sur : http://localhost:8501

---

## Étape 1 — Sauvegarder les modèles (dans le notebook)

Ajouter ces cellules à la fin du notebook avant de déployer :

```python
import joblib

# Modèle 1 — tabulaire
joblib.dump(model_lr, "model_lr.pkl")
joblib.dump(scaler, "scaler.pkl")      # si vous avez normalisé

# Modèle 2 — CNN
model.save("model_cnn.h5")

# Modèle fusion (version 2)
joblib.dump(fusion_model, "model_fusion.pkl")
```

Placer ces fichiers dans le **même dossier** que `app.py`.

---

## Étape 2 — Déploiement sur Streamlit Cloud (recommandé)

1. Créer un dépôt GitHub avec :
   ```
   app.py
   requirements.txt
   model_lr.pkl
   scaler.pkl
   model_cnn.h5
   model_fusion.pkl
   ```
2. Aller sur https://share.streamlit.io
3. "New app" → sélectionner le dépôt → `app.py` → Deploy
4. URL publique générée automatiquement (ex: `https://votre-app.streamlit.app`)

---

## Étape 2 (alternative) — Déploiement sur Render

1. Créer un `Procfile` :
   ```
   web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
2. Pousser sur GitHub
3. Créer un "Web Service" sur https://render.com
4. Connecter le dépôt → Render détecte automatiquement Python

---

## Étape 2 (optionnel) — Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

```bash
docker build -t cancer-detection .
docker run -p 8501:8501 cancer-detection
```

---

## Structure des fichiers

```
projet/
├── app.py                   ← Interface Streamlit
├── requirements.txt         ← Dépendances Python
├── model_lr.pkl             ← Modèle 1 sauvegardé
├── scaler.pkl               ← Normaliseur (optionnel)
├── model_cnn.h5             ← CNN sauvegardé
├── model_fusion.pkl         ← Modèle fusion (optionnel)
├── README.md                ← Ce fichier
└── Dockerfile               ← (optionnel)
```
