import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configurer matplotlib pour un rendu en arrière-plan sans GUI
matplotlib.use('Agg')

def main():
    sys.stdout.write("=== DEBUT DU PIPELINE TP3 (DEEP LEARNING — ZALANDO) ===\n")

    # 0. Setup directories
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # =============================================
    # ÉTAPE 1 — CHARGER ET EXPLORER LES DONNÉES
    # =============================================
    sys.stdout.write("\n[1] Chargement du dataset Zalando Fashion-MNIST...\n")
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Les 10 catégories du catalogue Zalando
    class_names = [
        'T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
        'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine'
    ]

    sys.stdout.write(f"Train : {X_train.shape[0]} images de {X_train.shape[1]}x{X_train.shape[2]} pixels\n")
    sys.stdout.write(f"Test  : {X_test.shape[0]} images\n")
    sys.stdout.write(f"Pixels : min={X_train.min()}, max={X_train.max()} (niveaux de gris 0-255)\n")
    sys.stdout.write(f"Categories : {len(class_names)}\n")

    # Distribution des catégories
    sys.stdout.write("\nDistribution des categories (train) :\n")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        sys.stdout.write(f"  {class_names[u]:12s} : {c:>5d} articles ({c/len(y_train)*100:.1f}%)\n")

    # Visualisation : 1 exemple par catégorie
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        idx = np.where(y_train == i)[0][0]
        ax.imshow(X_train[idx], cmap='gray')
        ax.set_title(class_names[i], fontsize=10)
        ax.axis('off')
    plt.suptitle('Catalogue Zalando — 1 exemple par catégorie', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'catalogue_samples.png'))
    plt.close()
    sys.stdout.write(">> catalogue_samples.png sauvegarde.\n")

    # =============================================
    # ÉTAPE 2 — PRÉTRAITEMENT DES IMAGES
    # =============================================
    sys.stdout.write("\n[2] Pretraitement des images...\n")

    # Normalisation : [0, 255] → [0.0, 1.0]
    X_train_norm = X_train.astype('float32') / 255.0
    X_test_norm = X_test.astype('float32') / 255.0
    sys.stdout.write(f"Apres normalisation : min={X_train_norm.min():.1f}, max={X_train_norm.max():.1f}\n")

    # Pour scikit-learn : aplatir chaque image 28x28 en vecteur de 784 valeurs
    X_train_flat = X_train_norm.reshape(-1, 784)
    X_test_flat = X_test_norm.reshape(-1, 784)
    sys.stdout.write(f"Forme pour ML classique (aplatie) : {X_train_flat.shape}\n")
    sys.stdout.write(f"Forme pour reseau dense (grille)  : {X_train_norm.shape}\n")

    # =============================================
    # ÉTAPE 3 — BASELINE : RANDOM FOREST
    # =============================================
    sys.stdout.write("\n[3] Entrainement du Random Forest (baseline)...\n")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, y_train)
    y_pred_rf = rf.predict(X_test_flat)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    sys.stdout.write(f"Random Forest — Accuracy : {acc_rf*100:.1f}%\n")

    # =============================================
    # ÉTAPE 4 — RÉSEAU DENSE (MLP)
    # =============================================
    sys.stdout.write("\n[4] Construction et entrainement du reseau dense (MLP)...\n")

    model_dense = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),                         # 28x28 → 784
        keras.layers.Dense(128, activation='relu'),     # Couche cachée 1
        keras.layers.Dense(64, activation='relu'),      # Couche cachée 2
        keras.layers.Dense(10, activation='softmax')    # 10 catégories → probabilités
    ])

    model_dense.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model_dense.summary(print_fn=lambda x: sys.stdout.write(x + '\n'))

    history_dense = model_dense.fit(
        X_train_norm, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.15,
        verbose=1
    )

    # =============================================
    # ÉTAPE 5 — RÉSEAU CONVOLUTIF (CNN)
    # =============================================
    sys.stdout.write("\n[5] Construction et entrainement du CNN...\n")

    # Ajouter la dimension canal
    X_train_cnn = X_train_norm.reshape(-1, 28, 28, 1)
    X_test_cnn = X_test_norm.reshape(-1, 28, 28, 1)
    sys.stdout.write(f"Forme pour CNN : {X_train_cnn.shape}\n")

    model_cnn = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        # Bloc 1 : détection de motifs simples (contours, bords)
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        # Bloc 2 : détection de motifs complexes (formes, structures)
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        # Classification finale
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])

    model_cnn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model_cnn.summary(print_fn=lambda x: sys.stdout.write(x + '\n'))

    history_cnn = model_cnn.fit(
        X_train_cnn, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.15,
        verbose=1
    )

    # =============================================
    # ÉTAPE 6 — COURBES D'APPRENTISSAGE
    # =============================================
    sys.stdout.write("\n[6] Generation des courbes d'apprentissage...\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Réseau dense
    axes[0].plot(history_dense.history['accuracy'], 'b-', label='Train')
    axes[0].plot(history_dense.history['val_accuracy'], 'r-', label='Validation')
    axes[0].set_title('Réseau Dense (MLP)')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CNN
    axes[1].plot(history_cnn.history['accuracy'], 'b-', label='Train')
    axes[1].plot(history_cnn.history['val_accuracy'], 'r-', label='Validation')
    axes[1].set_title('CNN')
    axes[1].set_xlabel('Époque')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Courbes d'apprentissage — Diagnostic overfitting", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()
    sys.stdout.write(">> learning_curves.png sauvegarde.\n")

    # =============================================
    # ÉTAPE 7 — COMPARAISON DES 3 APPROCHES
    # =============================================
    sys.stdout.write("\n[7] Comparaison des 3 approches...\n")

    loss_dense, acc_dense = model_dense.evaluate(X_test_norm, y_test, verbose=0)
    loss_cnn, acc_cnn = model_cnn.evaluate(X_test_cnn, y_test, verbose=0)

    sys.stdout.write("=" * 55 + "\n")
    sys.stdout.write(" COMPARAISON DES 3 APPROCHES — COMITE TECHNIQUE\n")
    sys.stdout.write("=" * 55 + "\n")
    sys.stdout.write(f" {'Modele':<20s} {'Accuracy':>10s} {'Taux erreur':>12s}\n")
    sys.stdout.write("-" * 55 + "\n")
    sys.stdout.write(f" {'Random Forest':<20s} {acc_rf*100:>9.1f}% {(1-acc_rf)*100:>11.1f}%\n")
    sys.stdout.write(f" {'Reseau Dense':<20s} {acc_dense*100:>9.1f}% {(1-acc_dense)*100:>11.1f}%\n")
    sys.stdout.write(f" {'CNN':<20s} {acc_cnn*100:>9.1f}% {(1-acc_cnn)*100:>11.1f}%\n")
    sys.stdout.write("-" * 55 + "\n")
    sys.stdout.write(f" Objectif business : taux d'erreur < 5.0%\n")
    sys.stdout.write("=" * 55 + "\n")

    # Prédictions CNN
    y_pred_cnn = model_cnn.predict(X_test_cnn, verbose=0)
    y_pred_classes = np.argmax(y_pred_cnn, axis=1)

    # Matrice de confusion CNN
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de confusion — CNN (articles Zalando)')
    plt.ylabel('Catégorie réelle')
    plt.xlabel('Catégorie prédite')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_cnn.png'))
    plt.close()
    sys.stdout.write(">> confusion_matrix_cnn.png sauvegarde.\n")

    # Rapport par catégorie
    sys.stdout.write("\n=== RAPPORT PAR CATEGORIE — CNN ===\n")
    sys.stdout.write(classification_report(y_test, y_pred_classes, target_names=class_names))

    # =============================================
    # ÉTAPE 8 — ANALYSE DES ERREURS
    # =============================================
    sys.stdout.write("\n[8] Analyse des erreurs du CNN...\n")

    errors = np.where(y_pred_classes != y_test)[0]
    sys.stdout.write(f"Erreurs : {len(errors)} / {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)\n")

    # Afficher 10 erreurs types
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, ax in enumerate(axes.flat):
        idx = errors[i]
        ax.imshow(X_test[idx], cmap='gray')
        confiance = y_pred_cnn[idx, y_pred_classes[idx]] * 100
        ax.set_title(
            f"Prédit : {class_names[y_pred_classes[idx]]} ({confiance:.0f}%)\n"
            f"Réel : {class_names[y_test[idx]]}",
            fontsize=8,
            color='red'
        )
        ax.axis('off')
    plt.suptitle('CNN — Articles mal classifiés (analyse qualité)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'erreurs_cnn.png'))
    plt.close()
    sys.stdout.write(">> erreurs_cnn.png sauvegarde.\n")

    # Top 5 confusions
    sys.stdout.write("\n=== TOP 5 CONFUSIONS LES PLUS FREQUENTES ===\n")
    confusions = {}
    for real, pred in zip(y_test[errors], y_pred_classes[errors]):
        pair = (class_names[real], class_names[pred])
        confusions[pair] = confusions.get(pair, 0) + 1

    top_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:5]
    for (real, pred), count in top_confusions:
        sys.stdout.write(f"  {real:12s} -> classe comme {pred:12s} : {count} erreurs\n")

    # =============================================
    # ÉTAPE 9 — VISUALISER CE QUE LE CNN A APPRIS
    # =============================================
    sys.stdout.write("\n[9] Visualisation des filtres et activations du CNN...\n")

    # 9a — Filtres appris par la première couche
    first_conv_layer = model_cnn.layers[0]
    filters, biases = first_conv_layer.get_weights()
    sys.stdout.write(f"Filtres : {filters.shape}\n")

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filters[:, :, 0, i], cmap='gray')
        ax.set_title(f'F{i+1}', fontsize=7)
        ax.axis('off')
    plt.suptitle('Filtres appris par le CNN — 1re couche Conv2D (3x3)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'filtres_conv.png'))
    plt.close()
    sys.stdout.write(">> filtres_conv.png sauvegarde.\n")

    # 9b — Ce que « voit » le CNN sur un article (Sneaker)
    activation_model = keras.Model(
        inputs=model_cnn.input,
        outputs=model_cnn.layers[0].output
    )

    sample_idx = np.where(y_test == 7)[0][0]
    sample = X_test_cnn[sample_idx:sample_idx + 1]
    activations = activation_model.predict(sample, verbose=0)
    sys.stdout.write(f"Activations : {activations.shape}\n")

    fig, axes = plt.subplots(1, 9, figsize=(16, 2.5))
    axes[0].imshow(X_test[sample_idx], cmap='gray')
    axes[0].set_title('Original', fontsize=9)
    axes[0].axis('off')
    for i in range(8):
        axes[i + 1].imshow(activations[0, :, :, i], cmap='viridis')
        axes[i + 1].set_title(f'Filtre {i+1}', fontsize=9)
        axes[i + 1].axis('off')
    plt.suptitle(f'Activations du CNN — {class_names[y_test[sample_idx]]}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activations_conv.png'))
    plt.close()
    sys.stdout.write(">> activations_conv.png sauvegarde.\n")

    # =============================================
    # RÉSUMÉ FINAL
    # =============================================
    sys.stdout.write("\n" + "=" * 60 + "\n")
    sys.stdout.write("  RESUME FINAL — PERFORMANCES\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write(f"  Random Forest  : {acc_rf*100:.1f}% accuracy ({(1-acc_rf)*100:.1f}% erreur)\n")
    sys.stdout.write(f"  Reseau Dense   : {acc_dense*100:.1f}% accuracy ({(1-acc_dense)*100:.1f}% erreur)\n")
    sys.stdout.write(f"  CNN            : {acc_cnn*100:.1f}% accuracy ({(1-acc_cnn)*100:.1f}% erreur)\n")
    sys.stdout.write(f"  Objectif       : < 5.0% erreur\n")

    objectif_atteint = (1 - acc_cnn) * 100 < 5.0
    if objectif_atteint:
        sys.stdout.write("  >>> OBJECTIF BUSINESS ATTEINT par le CNN ! <<<\n")
    else:
        sys.stdout.write("  >>> Objectif non atteint — ameliorations necessaires <<<\n")

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("\n>>> Fin de l'execution TP3 ! Livrables sauvegardes dans outputs/\n")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
