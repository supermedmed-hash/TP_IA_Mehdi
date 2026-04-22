# 📊 Analyse Critique — TP3 Deep Learning

## Comparaison des 3 approches

| Modèle | Accuracy estimée | Taux d'erreur | Exploite la spatialité ? |
|--------|:---:|:---:|:---:|
| Random Forest | ~87–88 % | ~12–13 % | ❌ Non |
| Réseau Dense (MLP) | ~88–89 % | ~11–12 % | ❌ Non |
| CNN | ~91–92 % | ~8–9 % | ✅ Oui |

## Pourquoi le CNN surpasse les autres ?
1. **Structure spatiale :** Les convolutions exploitent la proximité des pixels. Un filtre 3×3 détecte des contours, textures et formes que ni le Random Forest ni le MLP ne peuvent capturer.
2. **Hiérarchie de features :** La 1re couche Conv2D détecte des bords simples. La 2e combine ces bords en formes complexes (manches, col, semelle). C'est analogue à la vision humaine.
3. **Partage de poids :** Un même filtre est réutilisé sur toute l'image → moins de paramètres qu'un Dense pour une meilleure généralisation.

## Diagnostic des courbes d'apprentissage
- **MLP :** Écart train/validation modéré, léger overfitting après l'époque 10. L'accuracy de validation stagne vers 88 %.
- **CNN :** Écart train/validation plus contrôlé grâce au Dropout(0.3). Convergence plus rapide (5-6 époques suffisent).

## Catégories les plus confondues
1. **Chemise ↔ T-shirt/top :** Silhouettes très similaires à 28×28
2. **Chemise ↔ Pull :** Mêmes proportions, seule la texture diffère
3. **Manteau ↔ Pull :** Contours proches, longueur similaire
4. **Sneaker ↔ Bottine :** Formes de chaussures proches

## Limites identifiées
- **Résolution insuffisante :** 28×28 pixels empêche de distinguer les textures fines
- **Niveaux de gris :** La couleur est un critère de distinction important perdu ici
- **Objectif non atteint :** Même le CNN n'atteint pas < 5 % d'erreur avec cette configuration
- **Pas de data augmentation :** Rotation, zoom, flip augmenteraient la généralisation

## Recommandations pour la production
1. Utiliser des images **224×224 RGB** (résolution réelle Zalando)
2. Appliquer du **Transfer Learning** (ResNet50, EfficientNet pré-entraîné sur ImageNet)
3. Ajouter de la **data augmentation** (rotation, flip horizontal, zoom)
4. Implémenter un **seuil de confiance** : si P(prédiction) < 0.85, soumettre à validation humaine
5. Fusionner les catégories les plus confondues pour un premier déploiement
