# Synthèse de fin de TP : Les 3 Insights Majeurs

- **Insight 1 - La Performance VS La Simplicité**
  - Observation : Sur notre jeu de Test final, le modèle de base (Arbre de Décision) donne des scores quasiment aussi bons que le fameux Random Forest.
  - Explication : Vu que nos données sont assez simples et peu nombreuses au début (150 individus), utiliser un modèle beaucoup plus complexe (Random Forest) ne ramène presque pas de pourcents de précision supplémentaire.
  - Implication pratique : Dans un vrai projet de développement avec un client, il vaut parfois mieux conserver un algorithme très basique mais ultra rapide, plutôt qu'un système très lourd à faire tourner sur des serveurs si la qualité de sortie est finalement tout aussi bonne.

- **Insight 2 - Le problème du Sur-apprentissage (Overfitting)**
  - Observation : Dès qu'on donne un grand arbre à l'I.A. (`max_depth > 4`), la courbe de précision de l'entraînement grimpe à 100%, mais à côté le score du vrai test n'avance plus.
  - Explication : C'est ce qu'on appelle "l'overfitting". Le modèle a juste recopié tout le tableau qu'on lui a donné. Il a eu 20/20 à l'examen mais il en est ressorti en étant devenu incapable de généraliser son raisonnement sur la classification de nouvelles fleurs dans la nature.
  - Implication pratique : Il faut impérativement encadrer nos algorithmes (les brider), et utiliser continuellement une base de Test et Train pour surveiller le moment exact où on doit arrêter l'ordinateur de mémoriser.

- **Insight 3 - Pouvoir justifier ses choix grâce à SHAP**
  - Observation : Avec le graphique SHAP, l'outil nous prouve que c'est la taille des longueurs (`petal_length`) qui dirige purement la décision pour identifier l'espèce finale. C'est elle qui impacte le résultat.
  - Explication : Au lieu de rester avec une "Boîte Noire" dont on ne comprend rien au raisonnement mathématique en sortie, SHAP nous montre visuellement là où l'I.A. a mis son "poids" pour pencher dans telle direction.
  - Implication pratique : En B2B (Médical, Banques), il est impossible d'exploiter un algorithme sans pouvoir prouver à un être humain (le spécialiste, le médecin, le juge) pourquoi l'algorithme a refusé une demande. Intégrer de l'explicabilité comme SHAP est devenu obligatoire.
