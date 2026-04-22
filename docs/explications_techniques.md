# Réponses aux questions du TP1

## Étape 1 : Exploration des données
- **Le dataset est-il équilibré ?** Oui. Quand on regarde la répartition, on a exactement 50 fleurs pour chacune des 3 espèces. Le jeu de données est donc parfaitement équilibré.
- **Y a-t-il des valeurs manquantes ?** Non (0 valeurs manquantes), les données sont propres et prêtes à être utilisées.
- **Quelles variables semblent a priori les plus discriminantes ?** En regardant les graphiques, on voit très bien que la taille des pétales (`petal_length` et `petal_width`) permet de séparer très facilement l'espèce Setosa des deux autres espèces.

## Étape 2 : Modélisation
- **Pourquoi commencer par un modèle simple (Baseline) ?** C'est essentiel pour avoir un "score de base" à battre. Ça permet de voir si nos données ont du potentiel (pour ne pas s'embêter à coder un gros modèle lourd si un arbre simple marche déjà très bien dès le début).

## Étape 3 : Évaluation
- **Le Random Forest surpasse-t-il le Decision Tree ?** Sur nos données "Iris" oui et non. Les deux modèles ont des scores très proches (autour de 93%). Étant donné que nos données séparent assez bien les fleurs à la base, un simple arbre de décision suffit en général à avoir de très bons résultats !
- **Y a-t-il des espèces plus difficiles à classifier que d’autres ?** Les espèces `versicolor` et `virginica` se font parfois confondre par l'algorithme, car leurs caractéristiques physiques se chevauchent un peu plus que l'espèce Setosa.
- **Quelle est la différence entre accuracy et F1-score ?** L'Accuracy calcule juste le pourcentage de bonnes réponses (c'est très bien quand on a des classes à 50/50/50 comme on l'a ici). Le F1-score est beaucoup plus pertinent quand on a des données déséquilibrées (comme la détection de fraudes bancaires ou médicales).

## Étape 4 : Tuning
- **À partir de quelle profondeur observe-t-on de l'overfitting ?** On voit le sur-apprentissage (overfitting) souvent à partir de `max_depth = 4`. La précision d'entraînement monte vers les 100%, alors que celle du vrai jeu de Test stagne (autour de 93%). En clair, l'algo apprend "par cœur" la réponse au lieu de comprendre la logique.
- **Quel est le meilleur compromis nb_arbres/calcul ?** On gagne très vite en précision autour de 50 arbres. Dire au PC de calculer 500 arbres ne sert à rien sur de si petites bases, ça fait juste travailler la machine dans le vide pour 0.5% hypothétiques.
- **Qu'apporte la validation croisée ?** La Validation Croisée (Cross Validation) permet de lisser nos résultats et de tuer le hasard. Au lieu de couper notre jeu de données une seule fois, le code va faire par exemple 5 découpes. La note finale sera la moyenne des 5. Le résultat sera forcément plus fiable.

## Étape 5 : SHAP
- **Quelle variable domine selon SHAP ?** L'outil nous le montre visuellement : ce sont les variables liées aux pétales (`petal_length` par exemple) qui dirigent les choix de l'I.A. (et non les sépales).
- **Quelle différence avec sklearn (MDI) et SHAP ?** L'importance générée par Sklearn calcule de manière un peu bête comment l'arbre a coupé le plus souvent ses feuilles (mathématique de base). SHAP est une méthode supérieure car il calcule comment la présence de cette variable précise a pu faire basculer la décision finale.
- **Pourquoi l'explicabilité est cruciale ?** Dans un cas dans la vraie vie (ex: pour accorder un crédit, refuser un dossier médical...), on n'a plus le droit juridique d'utiliser des algorithmes "boîte noire" qui se cachent derrière un pourcentage. Il faut absolument pouvoir prouver ce qui a influencé l'IA pour ne pas créer de discrimination ou d'injustice. SHAP sert précisément à ça.
