# 🧠 TP3 — Deep Learning : Classification Automatique de Produits E-commerce
> **Auteur :** Mehdi | **Matière :** Les fondamentaux de l'IA (B3)

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg) ![CNN](https://img.shields.io/badge/CNN-Convolutif-brightgreen.svg) ![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red.svg)

---

## 🎯 Contexte du Projet
Ce TP consiste à développer un système de **classification automatique d'images produits** pour la plateforme e-commerce Zalando. L'objectif métier est de réduire le taux d'erreur de catégorisation de **12 % à moins de 5 %**.

Le projet couvre :
- 📦 Le chargement et l'exploration du dataset Fashion-MNIST (70 000 images Zalando)
- 🔄 Le prétraitement et la normalisation des images
- 🌲 Une baseline ML classique avec Random Forest
- 🧠 Un réseau de neurones dense (MLP)
- 🔬 Un réseau convolutif (CNN) exploitant la structure spatiale
- 📈 L'analyse des courbes d'apprentissage (overfitting/underfitting)
- 🎯 La comparaison des 3 approches et l'analyse des erreurs
- 👁️ La visualisation des filtres et activations du CNN (interprétabilité)

---

## 📂 Structure du Répertoire

```text
📁 TP3 (branche)
├── 📁 src
│   └── main.py                        # Pipeline complet (Random Forest, MLP, CNN)
├── 📁 outputs                         # Graphiques générés automatiquement
│   ├── catalogue_samples.png
│   ├── learning_curves.png
│   ├── confusion_matrix_cnn.png
│   ├── erreurs_cnn.png
│   ├── filtres_conv.png
│   └── activations_conv.png
├── 📁 docs
│   ├── cadrage_metier.md              # Cadrage projet (contexte Zalando)
│   ├── analyse_critique.md            # Analyse approfondie des résultats
│   └── debrief_technique.md           # 3 insights pour le comité technique
├── 📁 EXPLICATION
│   └── Explication_Code_TP3.md        # Guide ligne par ligne du code
├── 📄 Livrable_Final_TP3.html         # Rapport HTML complet (livrable)
├── 📄 Cours_Interactif_Mehdi_TP3.html # Cours explicatif interactif
└── 📄 requirements.txt               # Dépendances Python
```

---

## 🚀 Lancer le projet

```bash
git clone https://github.com/supermedmed-hash/TP_IA_Mehdi.git
cd TP_IA_Mehdi
git checkout TP3
pip install -r requirements.txt
python src/main.py
```

Les graphiques finaux apparaîtront dans `outputs/`. Ouvrez `Livrable_Final_TP3.html` dans votre navigateur pour le rapport complet.
