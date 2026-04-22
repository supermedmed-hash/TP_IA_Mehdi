# 🌸 TP1 : Analyse d'un algorithme de Machine Learning (Iris Dataset)
> **Auteur :** Mehdi (B3) | **Matière :** Les fondamentaux de l'IA

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg) ![SHAP](https://img.shields.io/badge/SHAP-Explicabilité-brightgreen.svg)

---

## 🎯 Présentation du Projet
Ce dépôt contient mon travail complet sur le TP1 d'Intelligence Artificielle. L'objectif de ce laboratoire était de mettre en place un pipeline de Machine Learning de A à Z sur le célèbre jeu de données biométriques **Iris**.

Dans ce projet, j'ai notamment eu l'opportunité de :
- 📈 **Comparer la performance** d'une approche de base (Arbre de Décision) face à un modèle complexe (Random Forest).
- 🛑 **Étudier les dangers de l'Overfitting** (Sur-apprentissage) en observant la scission des performances entre l'entraînement et nos jeux de tests.
- 🔍 **Ouvrir la "Boîte Noire" avec SHAP** (Shapley Additive exPlanations) pour comprendre *exactement* sur quelles mensurations l'I.A a dirigé ses propres prédictions.

---

## 📂 Structure de mon Répertoire

```text
📁 TP1_ML_Analysis_Mehdi
├── 📁 src
│   └── main.py                     # Script Python final gérant l'I.A, l'entraînement et exportant les graphiques
├── 📁 outputs                      # Graphiques (.png) générés par le script (Matrice de confusion, Overfitting, etc.)
├── 📁 docs
│   ├── explications_techniques.md  # Réponses détaillées aux questions du TP
│   └── insights_debrief.md         # Synthèse de mes 3 insights majeurs (Performance, Overfitting, SHAP)
├── 📄 Livrable_Final_TP1.html      # Mon rapport HTML qui compile mes explications finales
├── 📄 Cours_Interactif_Mehdi.html  # Cours explicatif interactif
└── 📄 requirements.txt             # Dépendances requises par le projet
```

---

## 🚀 Comment lancer le projet

Le code a été travaillé pour de la reproductibilité automatique. Son mode "headless" empêche votre machine d'être inondée par les fenêtres de graphiques : le script compile et range lui-même ses livrables directement dans `outputs/` !

**1. Cloner et installer les modules essentiels :**
```bash
git clone https://github.com/supermedmed-hash/TP1_ML_Analysis_Mehdi.git
cd TP1_ML_Analysis_Mehdi
pip install -r requirements.txt
```

**2. Lancer l'entraînement et l'analyse complète :**
```bash
python src/main.py
```

Pensez à ouvrir le fichier **`Livrable_Final_TP1.html`** dans votre navigateur pour une lecture visuelle des réponses et de la méthodologie retenue ! ✨
