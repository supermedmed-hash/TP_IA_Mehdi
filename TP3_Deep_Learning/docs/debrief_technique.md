# 🎯 Debrief Technique — 3 Insights pour le Comité Zalando

## Insight 1 — ML classique vs Deep Learning

**Observation :** Le CNN atteint ~91 % d'accuracy contre ~87 % pour le Random Forest, soit un gain de ~4 points de pourcentage. Le taux d'erreur passe de ~13 % à ~9 %.

**Explication :** Le Random Forest traite chaque pixel comme une feature indépendante (784 features). Il ne comprend pas qu'un pixel à la position (10, 14) est voisin du pixel (10, 15). Le CNN, lui, applique des filtres locaux (3×3) qui détectent des motifs spatiaux : contours, textures, puis formes de plus en plus complexes. C'est cette compréhension de la géométrie de l'image qui fait la différence.

**Recommandation :** Le surcoût GPU du CNN est justifié par le gain de performance. Sur 45 millions d'articles, les 4 points gagnés représentent ~1,8 million d'articles mieux classifiés. L'inférence CNN sur GPU reste sous 5 ms/image — compatible avec le temps réel.

---

## Insight 2 — Dense vs CNN

**Observation :** Le réseau dense (MLP) atteint ~88–89 % tandis que le CNN atteint ~91 %. L'écart est de ~2–3 points.

**Explication :** Le réseau dense, malgré ses couches profondes, aplatit l'image en un vecteur de 784 valeurs et perd toute structure spatiale. La couche `Flatten()` détruit l'information de voisinage. Les convolutions du CNN préservent cette structure et exploitent l'invariance locale (un col de chemise reste un col de chemise, qu'il soit en haut à gauche ou au centre de l'image).

**Recommandation :** Un réseau dense suffirait pour des données tabulaires (comme le churn du TP2). Pour des données structurées en grille (images, séries temporelles), le CNN est systématiquement supérieur. La complexité supplémentaire est négligeable en production.

---

## Insight 3 — Fiabilité en production

**Observation :** Les confusions les plus fréquentes du CNN concernent les paires Chemise/T-shirt, Chemise/Pull, et Manteau/Pull. Certaines erreurs sont commises avec une confiance > 80 %, ce qui est un risque.

**Explication :** À 28×28 pixels en niveaux de gris, la texture (maille d'un pull vs tissu d'une chemise) est indiscernable. Seule la silhouette est exploitable. Or, ces catégories ont des silhouettes quasi-identiques à cette résolution. En production avec des images haute résolution et en couleur, ces confusions seraient fortement réduites.

**Recommandation :**
1. **Seuil de confiance :** Ne pas publier automatiquement si P(meilleure classe) < 85 %. Soumettre à un opérateur humain (human-in-the-loop).
2. **Catégories critiques :** Surveiller spécifiquement les paires Chemise/T-shirt et Chemise/Pull avec des métriques dédiées.
3. **Évolution :** Migrer vers un modèle Transfer Learning (EfficientNet) sur images 224×224 RGB pour atteindre l'objectif < 5 % d'erreur.
