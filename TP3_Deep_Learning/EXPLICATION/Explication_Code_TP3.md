# 📖 Explication du Code TP3 — Deep Learning (ligne par ligne)

Ce document explique chaque section du fichier `src/main.py` pour que tu comprennes exactement ce que fait le code.

---

## 1. Imports et configuration

```python
import tensorflow as tf
from tensorflow import keras
```
- **TensorFlow** est la librairie de Google pour le Deep Learning. **Keras** est l'API haut-niveau intégrée à TensorFlow qui simplifie la création de réseaux de neurones.
- `matplotlib.use('Agg')` → On configure matplotlib pour sauvegarder les graphiques sans ouvrir de fenêtre (utile sur serveur/Colab).

---

## 2. Chargement des données (Étape 1)

```python
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
```
- Fashion-MNIST est intégré directement dans Keras. Le téléchargement est automatique (~30 Mo).
- `X_train` = 60 000 images de 28×28 pixels (valeurs entières 0–255)
- `y_train` = 60 000 labels (entiers de 0 à 9, correspondant aux 10 catégories)
- Même chose pour test (10 000 images)

---

## 3. Normalisation (Étape 2)

```python
X_train_norm = X_train.astype('float32') / 255.0
```
- On divise par 255 pour ramener les pixels de [0, 255] à [0.0, 1.0].
- **Pourquoi ?** Les réseaux de neurones utilisent des gradients pour apprendre. Si les valeurs d'entrée sont trop grandes (0–255), les gradients explosent ou deviennent instables. Avec des valeurs entre 0 et 1, l'optimiseur `adam` converge beaucoup plus vite et de manière plus stable.

```python
X_train_flat = X_train_norm.reshape(-1, 784)
```
- Pour le Random Forest et le MLP, on « aplatit » chaque image 28×28 en un vecteur de 784 valeurs.
- Le `-1` signifie « calcule automatiquement cette dimension » (ici 60 000).

---

## 4. Random Forest — Baseline (Étape 3)

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_flat, y_train)
```
- 100 arbres de décision qui votent ensemble.
- `n_jobs=-1` → utilise tous les cœurs CPU en parallèle.
- Le Random Forest traite chaque pixel comme une feature indépendante : il ne sait PAS que le pixel (10,14) est à côté du pixel (10,15).

---

## 5. Réseau Dense / MLP (Étape 4)

```python
model_dense = keras.Sequential([...])
```
- `Sequential` = on empile les couches les unes après les autres (comme un sandwich).

### Détail des couches :
- **`Flatten()`** : Transforme l'image 28×28 en vecteur de 784. Zéro paramètre appris.
- **`Dense(128, activation='relu')`** : 128 neurones, chacun connecté aux 784 entrées. ReLU = « si la valeur est négative, on met 0 ; si positive, on garde ». C'est la non-linéarité qui permet au réseau d'apprendre des patterns complexes.
- **`Dense(64, activation='relu')`** : 2e couche cachée, 64 neurones.
- **`Dense(10, activation='softmax')`** : Couche de sortie. 10 neurones = 10 catégories. Softmax transforme les valeurs brutes en probabilités (somme = 1). Ex : [0.02, 0.01, 0.85, 0.03, ...] → « 85% de chance que ce soit un Pull ».

### Compilation :
```python
model_dense.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
- **`adam`** : Optimiseur adaptatif qui ajuste le learning rate automatiquement. Le plus utilisé en pratique.
- **`sparse_categorical_crossentropy`** : Fonction de perte pour classification multi-classe quand les labels sont des entiers (0, 1, 2, ..., 9). Si les labels étaient en one-hot ([0,0,1,...,0]), on utiliserait `categorical_crossentropy`.
- **`metrics=['accuracy']`** : On affiche l'accuracy pendant l'entraînement.

### Entraînement :
```python
history_dense = model_dense.fit(X_train_norm, y_train, epochs=15, batch_size=64, validation_split=0.15)
```
- **`epochs=15`** : Le réseau voit les 60 000 images 15 fois.
- **`batch_size=64`** : On traite 64 images à la fois avant de mettre à jour les poids.
- **`validation_split=0.15`** : On réserve 15 % du train (9 000 images) pour évaluer la généralisation à chaque époque, SANS jamais entraîner dessus.

---

## 6. CNN — Réseau Convolutif (Étape 5)

```python
X_train_cnn = X_train_norm.reshape(-1, 28, 28, 1)
```
- On ajoute une dimension « canal » : le `1` signifie niveaux de gris (1 seul canal). En couleur RGB ce serait `3`.

### Architecture du CNN :

```python
keras.layers.Conv2D(32, (3, 3), activation='relu')
```
- **32 filtres** de taille 3×3 balaient l'image. Chaque filtre détecte un motif local différent (contour horizontal, contour vertical, coin, etc.).
- La sortie est de taille 26×26 (et non 28×28) car le filtre 3×3 ne peut pas déborder du bord : il perd 1 pixel de chaque côté (28 - 3 + 1 = 26).

```python
keras.layers.MaxPooling2D((2, 2))
```
- Prend le maximum dans chaque bloc de 2×2 pixels. Réduit la taille de moitié (26×26 → 13×13).
- **Intérêt :** Réduit le nombre de calculs, rend le modèle un peu invariant aux petites translations, et force le réseau à capturer des motifs de plus en plus globaux.

```python
keras.layers.Dropout(0.3)
```
- À chaque batch d'entraînement, 30 % des neurones sont « éteints » aléatoirement.
- **Pourquoi ça réduit l'overfitting ?** Le réseau ne peut pas « mémoriser » les données en se reposant sur quelques neurones dominants. Il est forcé de distribuer l'information sur tous les neurones, ce qui améliore la généralisation.
- **Important :** Le Dropout est automatiquement désactivé en mode test/prédiction.

---

## 7. Courbes d'apprentissage (Étape 6)

```python
axes[0].plot(history_dense.history['accuracy'], 'b-', label='Train')
axes[0].plot(history_dense.history['val_accuracy'], 'r-', label='Validation')
```
- On trace l'accuracy sur le train (bleu) et la validation (rouge) à chaque époque.
- **Diagnostic :**
  - Si les deux courbes montent ensemble → le modèle apprend bien ✅
  - Si la courbe train monte mais la validation stagne/descend → **overfitting** ⚠️
  - Si les deux courbes sont basses → **underfitting** (modèle trop simple)

---

## 8. Matrice de confusion (Étape 7)

```python
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ...)
```
- La matrice de confusion montre, pour chaque catégorie réelle (ligne), combien d'articles ont été classés dans chaque catégorie prédite (colonne).
- La diagonale = prédictions correctes. Hors-diagonale = erreurs.
- Les zones « chaudes » hors-diagonale identifient les paires de catégories les plus confondues.

---

## 9. Visualisation des filtres et activations (Étape 9)

```python
filters, biases = first_conv_layer.get_weights()
```
- On extrait les 32 filtres 3×3 appris par la première couche convolutive.
- Ces filtres sont les « yeux » du CNN : ils montrent exactement quels motifs le modèle a appris à détecter.

```python
activation_model = keras.Model(inputs=model_cnn.input, outputs=model_cnn.layers[0].output)
activations = activation_model.predict(sample)
```
- On crée un « sous-modèle » qui renvoie la sortie intermédiaire de la 1re couche Conv2D.
- Les **feature maps** montrent quelles zones de l'image « allument » chaque filtre. C'est l'interprétabilité visuelle du CNN.
