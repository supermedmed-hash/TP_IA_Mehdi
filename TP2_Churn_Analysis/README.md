# 📡 TP2 — Mini-Projet IA : Prédiction du Churn Client (Cas A)
> **Auteur :** Mehdi | **Matière :** Les fondamentaux de l'IA (B3)

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg) ![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg) ![SHAP](https://img.shields.io/badge/SHAP-Explicabilité-brightgreen.svg) ![AI Act](https://img.shields.io/badge/AI%20Act-Conformité-orange.svg)

---

## 🎯 Contexte du Projet
Ce TP consiste à construire un pipeline ML complet sur des **données réelles** (IBM Telco Customer Churn : 7 043 clients, 21 colonnes). L'objectif métier est de **prédire si un client va résilier son abonnement télécom** avant qu'il ne parte, pour pouvoir intervenir.

Le projet couvre :
- 📋 Le cadrage métier et la définition des KPI
- 🧹 Le nettoyage et la préparation des données (encodage, normalisation)
- 🤖 La comparaison de 3 modèles : Régression Logistique, Random Forest, XGBoost
- 📊 L'évaluation approfondie (matrice de confusion, F1-score)
- 🔍 L'explicabilité avec SHAP
- ⚖️ L'analyse de conformité réglementaire (AI Act & RGPD)

---

## 📂 Structure du Répertoire

```text
📁 TP2 (branche)
├── 📁 src
│   └── main.py                     # Pipeline complet (chargement, modèles, SHAP)
├── 📁 outputs                      # Graphiques générés automatiquement
│   ├── comparaison_modeles.png
│   ├── confusion_matrix.png
│   ├── rf_evolution.png
│   └── shap_summary.png
├── 📁 docs
│   ├── cadrage_metier.md           # Cadrage projet (Étape 1)
│   ├── analyse_critique.md         # Analyse approfondie (Étape 4)
│   └── conformite_aiact.md         # Conformité AI Act & RGPD (Étape 6)
├── 📁 EXPLICATION
│   └── Explication_Code_TP2.md     # Guide ligne par ligne du code
├── 📄 Livrable_Final_TP2.html      # Rapport HTML complet
└── 📄 requirements.txt             # Dépendances Python
```

---

## 🚀 Lancer le projet

```bash
git clone https://github.com/supermedmed-hash/TP_IA_Mehdi.git
cd TP_IA_Mehdi
git checkout TP2
pip install -r requirements.txt
python src/main.py
```

Les graphiques finaux apparaîtront dans `outputs/`. Ouvrez `Livrable_Final_TP2.html` dans votre navigateur pour le rapport complet.
