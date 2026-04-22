# 🐍 Décryptage du Code Python : `src/main.py`

Ce document t'explique comment comprendre le script étape par étape, afin que tu puisses l'expliquer sereinement si ton professeur ou examinateur te pose des questions sur ton code.

---

## 1. La partie "Préparation & Outils" (Les Importations)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import shap
```
**Explication pour toi :** 
Ici, on remplit ta boîte à outils. 
- **Pandas (`pd`)** : C'est le super "Excel" de Python. Il te permet de lire le tableau de données (les différentes plantes).
- **Matplotlib (`plt`) & Seaborn (`sns`)** : Ce sont tes pinceaux virtuels. Ils servent exclusivement à programmer et dessiner tes jolis graphiques.
- **Sklearn (Scikit-Learn)** : C'est la bibliothèque reine du Machine Learning. C'est elle qui contient tous les modèles d'IA, et t'évite de recoder l'Arbre de Décision mathématiquement.
- **SHAP** : Le fameux outil puissant utilisé pour "lire dans les pensées" du modèle à la fin.

---

## 2. Chargement et découverte du Dataset (PANDAS)

```python
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
```
**Explication :** 
On indique au programme l'URL publique de la base de données. Ensuite, on ordonne à Pandas de télécharger ce fichier CSV et de le métamorphoser en un tableau virtuel appelé `df` (pour **D**ata**F**rame).

```python
print(df.describe())
sns.pairplot(df, hue='species')
```
**Explication :** 
La fonction `.describe()` génère en une ligne les statistiques du tableau (la moyenne, les valeurs minimum, l'écart-type...). 
Le `.pairplot()` est une fonction magique : elle gère automatiquement un panneau de multiples nuages de points qui fait s'affronter tes variables, en les repérant par couleur (`hue='species'`).

---

## 3. Préparer le terrain pour l'IA (Train / Test)

```python
le = LabelEncoder()
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = le.fit_transform(df['species'])
```
**Explication :** 
L'informatique ne sait techniquement pas manger du texte pour ses calculs neuronaux.
- **Tu as ton `X` (tes "features")** : Les chiffres, tailles des tiges etc...
- **Le `LabelEncoder`** va traduire ton texte résultat (`y`) en chiffres, par exemple : `0` pour Setosa, `1` pour Versicolor, `2` pour Virginica. L'ordinateur peut maintenant les traiter mathématiquement.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
**Explication :** 
L'une des étapes les plus importantes : la fameuse fente à 80/20. 
`test_size=0.2` veut dire qu'on va cacher précieusement 20% des fleurs d'Iris dans un coffre-fort (`X_test`). L'algorithme ne les verra jamais lors de ses révisions. Les 80% restants (`X_train`) serviront pour son entraînement. C'est ce qui permet de tester qu'il ne "triche" pas !
*Note : `random_state=42` bloque simplement le hasard. C'est pour être sûr que peu importe quand tu vas lancer le bot, "tes 20%" seront fixés. C'est utile pour partager ton test avec un collègue sans avoir une répartition aléatoire.*

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
```
**Explication :** 
Le **StandardScaler** est capital dans de très nombreux cas d'I.A. Il met toutes les unités "sur le même piédestal". Si l'une de t'es données est en km et une autre en centimètre, l'I.A pourrait faussement croire en favorisant le gros chiffre. On harmonise le tout pour stabiliser (`fit_transform`).

---

## 4. L'Entraînement 

```python
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train_sc, y_train)

rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf.fit(X_train_sc, y_train)
```
**Explication :** 
- **`.fit()`** : Retiens bien que `Fit` = `apprendre`. Tu livres tes données élèves (`X`) pour trouver les attentes exactes (`y`). 
- On met à l'épreuve l'Arbre simple (`dt`) face à la Forêt Aléatoire (`rf`). La forêt est plus douée car elle est composée de 100 mini-arbres indépendants (`n_estimators=100`) : et quand on fait la moyenne du vote de 100 individus... l'IA gagne en précision globale !
- **`max_depth=5`** : Le paramètre pour empêcher qu'un arbre aille creuser trop de branches jusqu'à tout résoudre avec son cœur (Overfitting). Ton arbre s'arrête à 5 étages max.

---

## 5. Overfitting avec "Tuning" intelligent
```python
for n in [10, 25, 50, 100]:
    m = RandomForestClassifier(n_estimators=n)
    cv = cross_val_score(m, X_train_sc, y_train, cv=5)
```
**Explication :** 
C'est ta boucle de Fine-Tuning. Au lieu d'essayer au hasard, ton ordi teste plein de forêts différentes (10 arbres, puis 25...).
Tu utilises la validation croisée `cross_val_score` avec `cv=5`. Cela signifie que pour tester les 25 arbres, l'ordinateur va segmenter tes données d'entraînements 5 fois avec un nouvel apprentissage, puis pondérer la moyenne. Tu abolis l'effet de chance, tes chiffres d'Accuracy ne mentent plus.

---

## 6. L'Interrogatoire final : Interprétabilité (SHAP)

```python
explainer = shap.TreeExplainer(rf_final)
shap_values = explainer.shap_values(X_test_sc)
```
**Explication :** 
Notre Random Forest est un modèle dit "Boite Noire", donc mathématiquement impossible à justifier à un client sans outils...
C'est pour cela qu'on jette notre modèle validé `rf_final` dans les bras de l'outil `shap.TreeExplainer`. Ce module exceptionnel va regarder dans ton I.A. les `shap_values` : "Ah, pour cette fleur-ci, l'I.A. a mis un poids +15 à cause du `petal_width`".

```python
importances_sklearn = rf_final.feature_importances_
```
**Explication (la différence majeure) :** 
`feature_importances_` c'est le calcul "Natif Sklearn" de l'I.A. Cet indicateur est moins bon, parce qu'il regarde surtout "combien de fois cette tige a réussi à séparer des catégories en cours de route dans la construction des branches" (ce qu'on appelle "l'impureté de nœud"). C'est purement mathématique, il ne s'intéresse jamais au résultat absolu de fin, **contrairement à SHAP**. Il est donc vital d'expliciter avec ce dernier.
