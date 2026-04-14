# 💎 Prédiction du Prix des Diamants

## Auteur
**Hadj Mbarek Arwa** — Projet Machine Learning End-to-End · Avril 2026

---

## 📌 Description du Projet

Ce projet vise à construire un modèle de Machine Learning capable de **prédire le prix d'un diamant** à partir de ses caractéristiques physiques (carat, taille, couleur, clarté, dimensions). Il s'agit d'un problème de **régression supervisée**, accompagné d'une application web interactive déployée avec Streamlit.

---

## 📂 Source du Dataset

- **Plateforme** : Kaggle
- **Lien** : [Diamond Dataset — Lovish Bansal](https://www.kaggle.com/datasets/lovishbansal123/diamond-dataset)
- **Taille brute** : 53 940 observations × 10 colonnes
- **Taille après nettoyage** : 53 772 observations × 12 colonnes (après suppression de 20 lignes avec dimensions nulles, 3 outliers extrêmes et 145 doublons)
- **Variable cible** : `price` (prix en dollars US)

### Variables du dataset

| Variable | Type | Description |
|----------|------|-------------|
| `carat` | Numérique | Poids du diamant |
| `cut` | Catégorielle Ordinale | Qualité de la taille (Fair → Ideal) |
| `color` | Catégorielle Ordinale | Couleur (J → D) |
| `clarity` | Catégorielle Ordinale | Clarté (I1 → IF) |
| `depth` | Numérique | Profondeur totale (%) |
| `table` | Numérique | Largeur du plateau (%) |
| `x` | Numérique | Longueur (mm) |
| `y` | Numérique | Largeur (mm) |
| `z` | Numérique | Profondeur (mm) |
| `price` | Numérique | **Prix en $ (cible)** |

**Features créées (Feature Engineering) :**

| Feature | Formule | Interprétation |
|---------|---------|----------------|
| `volume` | x × y × z | Volume du diamant en mm³ |
| `carat_per_volume` | carat / (volume + ε) | Densité du diamant |

---

## ❓ Problématique

> *"Quels sont les facteurs les plus influents sur le prix d'un diamant, et peut-on construire un modèle fiable pour l'estimer ?"*

---

## 🗂️ Structure du Repository

```
diamond_project/
├── data/                          # Dataset (à télécharger via Kaggle)
├── notebooks/
│   ├── 01_EDA.ipynb               # Analyse Exploratoire des Données
│   └── 02_Modeling_log_vs_nolog.ipynb  # Modélisation & comparaison log vs no-log
├── streamlit_dashboard/
│   ├── app.py                     # Application Streamlit interactive
│   │── model/
│   │   ├── best_model_optimise.pkl    # Random Forest optimisé sauvegardé
│   │   └── model_info.pkl             # Métadonnées du modèle
│   ├── src/
│   │   └── preprocessing.py       # Pipeline de préprocessing réutilisable
│  
├── requirements.txt               # Librairies nécessaires
└── README.md                      # Ce fichier
```

---

## ⚙️ Installation et Lancement

### 1. Cloner le repository
```bash
git clone https://github.com/arwa774/diamond_project.git
cd diamond_project
```

### 2. Créer l'environnement et installer les dépendances
```bash
conda create -n diamond_ml python=3.11
conda activate diamond_ml
pip install -r requirements.txt
```

### 3. Télécharger le dataset
Télécharger `diamonds.csv` depuis [Kaggle](https://www.kaggle.com/datasets/lovishbansal123/diamond-dataset) et le placer dans le dossier `data/`.

### 4. Lancer les notebooks
```bash
jupyter notebook
```
Ouvrir dans l'ordre : `01_EDA.ipynb` → `02_Modeling_log_vs_nolog.ipynb`

### 5. Lancer l'application Streamlit
```bash
cd streamlit_dashboard
python train_model.py   # entraîner et sauvegarder le modèle (une seule fois)
streamlit run app.py    # lancer le dashboard
```

---

## 🔬 Expérience : Log vs Sans Log

Le notebook `02_Modeling_log_vs_nolog.ipynb` compare deux approches pour chaque modèle :

- **Approche 1** : entraînement directement sur `price` en dollars
- **Approche 2** : entraînement sur `log(price)`, métriques reconverties en dollars via `exp()` pour une comparaison équitable

### Tableau de comparaison complet (métriques toutes en $)

| Modèle | Approche | RMSE Test ($) | MAE Test ($) | R² Test |
|--------|----------|:---:|:---:|:---:|
| **Random Forest** | **price [$]** | **$525** | **$263** | **0.9822** |
| XGBoost | price [$] | $525 | $271 | 0.9822 |
| Random Forest | log(price) | $526 | $263 | 0.9822 |
| XGBoost | log(price) | $528 | $268 | 0.9821 |
| Arbre de Décision | price [$] | $629 | $336 | 0.9745 |
| Arbre de Décision | log(price) | $631 | $338 | 0.9744 |
| Régression Linéaire | log(price) | $827 | $437 | 0.9560 |
| Régression Linéaire | price [$] | $1 199 | $781 | 0.9075 |

**Conclusion de l'expérience :** La transformation logarithmique n'apporte pas d'amélioration pour les modèles à base d'arbres (Random Forest, XGBoost). Ces algorithmes n'étant pas sensibles à l'échelle ni à la distribution de la cible, travailler directement en dollars est suffisant et plus simple à interpréter. La transformation log est utile principalement pour la **Régression Linéaire** (amélioration de R² de 0.907 → 0.956).

---

## 📊 Résultats Finaux

### Meilleur modèle : Random Forest (sans log, optimisé par GridSearchCV)

| Métrique | Valeur |
|----------|--------|
| **RMSE Test** | **$522.59** |
| **MAE Test** | **$262.36** |
| **R² Test** | **0.9824 (98.2%)** |
| R² Cross-Validation (5-fold) | 0.9812 ± 0.0006 |

**Meilleurs hyperparamètres trouvés :**
```python
{'max_depth': 20, 'min_samples_leaf': 2, 'n_estimators': 300}
```

### Top 3 features les plus importantes

| Rang | Feature | Importance |
|------|---------|:---:|
| 🥇 | `volume` (x × y × z) | 67.1% |
| 🥈 | `y` (largeur) | 19.3% |
| 🥉 | `clarity_encoded` | 6.3% |

> Le volume du diamant (feature créée manuellement) est de loin la variable la plus prédictive, ce qui confirme l'importance du Feature Engineering.

---

## 🌐 Application Streamlit

Le projet inclut un dashboard interactif développé avec Streamlit et Plotly. Le code source est disponible dans le dossier `streamlit_dashboard/`. Pour le lancer en local :

```bash
cd streamlit_dashboard
python train_model.py   # entraîner et sauvegarder le modèle (une seule fois)
streamlit run app.py    # lancer le dashboard sur localhost:8501
```

Le dashboard permet de :

- **Prédire** le prix d'un diamant en temps réel via des sliders et menus déroulants
- **Explorer** les données avec 11 graphiques Plotly interactifs (distributions, corrélations, feature importance, réel vs prédit, résidus)
- **Visualiser** les performances du modèle avec des KPI clairs (R², RMSE, MAE)

---

## 📚 Technologies Utilisées

| Catégorie | Librairies |
|-----------|-----------|
| Manipulation des données | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn, xgboost |
| Sauvegarde modèle | joblib |
| Application web | streamlit |
| Environnement | Python 3.11, Miniconda |
