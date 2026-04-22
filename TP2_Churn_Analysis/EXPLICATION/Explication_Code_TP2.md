# 🐍 Décryptage du Code Python : `src/main.py` (TP2 - Churn)

Ce document explique le code du TP2 ligne par ligne, pour que tu puisses comprendre chaque détail et le défendre devant ton professeur.

---

## 1. Les Importations (la boîte à outils)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
```
**Explication :**
On charge 3 algorithmes de Machine Learning différents :
- **LogisticRegression** : Le modèle le plus simple. Il trace une "ligne" dans les données pour séparer les clients qui partent de ceux qui restent. Malgré son nom, c'est bien un modèle de classification (pas de régression).
- **RandomForestClassifier** : Une forêt d'arbres de décision. Chaque arbre vote, et la majorité l'emporte.
- **XGBClassifier** : Le petit prodige. XGBoost (eXtreme Gradient Boosting) construit des arbres les uns après les autres, chaque nouvel arbre corrigeant les erreurs du précédent. C'est souvent le meilleur sur les données tabulaires.

---

## 2. Le Chargement et Nettoyage des Données

```python
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
```
**Explication :**
On charge un vrai dataset d'IBM contenant 7 043 clients télécom avec 21 colonnes (type de contrat, facture mensuelle, services souscrits, etc.). C'est du VRAI data, pas un jouet comme Iris.

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
```
**Explication :**
Ici on nettoie les données sales. La colonne `TotalCharges` contient des espaces vides au lieu de chiffres (pour les nouveaux clients qui n'ont pas encore payé). `pd.to_numeric(..., errors='coerce')` transforme ces espaces en `NaN` (valeur manquante), puis `dropna()` les supprime. On passe de 7 043 à 7 032 lignes. Pas de panique, on perd que 11 lignes.

---

## 3. L'Encodage des Données Textuelles

```python
df_enc = pd.get_dummies(df.drop('customerID', axis=1), drop_first=True)
```
**Explication :**
C'est le moment clé. Nos données contiennent plein de texte ("Yes", "No", "Month-to-month"...) mais l'ordinateur ne sait calculer qu'avec des chiffres.

`get_dummies()` transforme chaque catégorie en colonne binaire (0 ou 1). Par exemple, la colonne `Contract` qui contenait "Month-to-month", "One year", "Two year" devient :
- `Contract_One year` → 0 ou 1
- `Contract_Two year` → 0 ou 1
(Si les deux valent 0, ça veut dire "Month-to-month" → c'est le `drop_first`)

On passe de 21 colonnes à **30 features** ! C'est normal, c'est l'effet de l'encodage.

On enlève aussi `customerID` car c'est juste un identifiant unique, ça n'a aucun sens pour l'IA.

---

## 4. Le Split Stratifié (80/20)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
**Explication :**
On coupe le dataset en deux : 80% pour l'entraînement, 20% pour le test final.

Le mot-clé important ici c'est **`stratify=y`**. Vu que notre dataset est déséquilibré (~26% de churn), si on coupait au hasard on pourrait se retrouver avec 35% de churn dans le test et 20% dans le train. Le `stratify` garantit que la proportion reste la même dans les deux morceaux. C'est super important pour que notre évaluation soit juste !

---

## 5. L'Entraînement des 3 Modèles

```python
modeles = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
}

for nom, modele in modeles.items():
    modele.fit(X_train_sc, y_train)
    pred = modele.predict(X_test_sc)
```
**Explication :**
On a un dictionnaire avec nos 3 challengers. La boucle `for` entraîne chaque modèle un par un avec `.fit()` ("apprendre") puis fait ses prédictions avec `.predict()` ("deviner").

- `max_iter=1000` : On laisse la Régression Logistique faire 1000 itérations maximum pour converger (trouver sa meilleure "ligne").
- `n_estimators=100` : On fait voter 100 arbres (pour Random Forest et XGBoost).
- `eval_metric='logloss'` : La fonction de perte de XGBoost (comment il mesure ses erreurs internes).
- `verbosity=0` : On dit à XGBoost de se taire dans la console.

---

## 6. Les Métriques (Accuracy vs F1)

```python
acc = accuracy_score(y_test, pred)
f1_mac = f1_score(y_test, pred, average='macro')
f1_wei = f1_score(y_test, pred, average='weighted')
```
**Explication :**
- **Accuracy** : Le % de bonnes réponses brut. Simple mais trompeur sur un dataset déséquilibré. Exemple : si 74% des clients restent, un modèle qui dit toujours "il reste" aurait 74% d'accuracy sans rien comprendre !
- **F1-macro** : La moyenne des F1-scores de chaque classe, sans tenir compte du nombre d'individus. C'est plus équitable car ça met la classe minoritaire (Churn=Yes) sur le même plan que la majorité.
- **F1-weighted** : Comme le macro, mais en tenant compte du nombre d'individus par classe.

Pour ce TP, on se fie au **F1-macro** car on veut être aussi bon sur les churners que sur les non-churners.

---

## 7. L'Explicabilité SHAP

```python
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)
```
**Explication :**
On utilise SHAP sur le Random Forest (et pas XGBoost) car `TreeExplainer` est optimisé pour les modèles à base d'arbres.

SHAP calcule pour chaque client et chaque variable : "Est-ce que cette variable a poussé la prédiction vers Churn=Yes ou vers Churn=No, et de combien ?"

Le graphique final (`shap_summary.png`) te montre les 15 variables qui influencent le plus la décision. Les points rouges à droite = cette variable pousse vers le churn. Les points bleus à gauche = cette variable protège du churn.

Par exemple, un contrat "Month-to-month" (mensuel sans engagement) va fortement pousser vers le churn. Un contrat "Two year" va protéger.
