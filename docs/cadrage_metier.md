# 📋 Cadrage Métier — TP3 Deep Learning (Zalando)

## Contexte
- **Entreprise :** Zalando — plus grande plateforme e-commerce de mode en Europe
- **Chiffres clés :** 50 millions de clients actifs, 45 millions d'articles au catalogue
- **Équipe :** Catalog Intelligence (classification automatique des produits)

## Problème métier
Sur la marketplace, les vendeurs tiers saisissent manuellement la catégorie de leurs produits.
- **12 %** des articles sont mal catégorisés
- **Coût annuel :** 2,3 M€/an en retours liés à des erreurs de catalogue
- **Impact UX :** dégradation de l'expérience de recherche client

## Objectif business
Passer le taux d'erreur de catégorisation de **12 % à moins de 5 %** grâce à un système de classification automatique par images.

## Dataset
- **Source :** Fashion-MNIST (Zalando Research, 2017)
- **Taille :** 70 000 images réelles du catalogue Zalando
- **Format :** Niveaux de gris, 28×28 pixels
- **Classes :** 10 catégories (T-shirt, Pantalon, Pull, Robe, Manteau, Sandale, Chemise, Sneaker, Sac, Bottine)
- **Split :** 60 000 train / 10 000 test

## KPI technique
- **Métrique principale :** Accuracy (taux de bonnes classifications)
- **Objectif :** Accuracy ≥ 95 % (⟺ taux d'erreur < 5 %)
- **Métriques secondaires :** Precision/Recall par catégorie (certaines confusions sont plus coûteuses)

## Approche
Comparer 3 architectures de complexité croissante :
1. **Random Forest** (ML classique) — baseline
2. **Réseau Dense / MLP** (Deep Learning simple)
3. **CNN** (Deep Learning convolutif) — exploitation de la structure spatiale

## Risques identifiés
- Résolution 28×28 très inférieure à la production réelle (≥ 224×224)
- Certaines catégories visuellement proches (Pull ↔ Chemise, T-shirt ↔ Chemise)
- Overfitting possible sur les réseaux profonds
