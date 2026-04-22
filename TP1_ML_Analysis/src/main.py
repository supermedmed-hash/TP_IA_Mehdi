import os
import sys
import matplotlib
# Configuration pour une exécution sans affichage graphique bloquant (génération background)
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import shap

def main():
    # Définition du chemin pour le dossier d'outputs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================
    # Étape 1 : Charger et explorer les données
    # ==========================================
    sys.stdout.write("=== EXPLORATION DES DONNÉES ===\n")
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    
    sys.stdout.write(f"Forme du dataset : {df.shape}\n")
    sys.stdout.write(f"\nDistribution des classes :\n{df['species'].value_counts()}\n")
    sys.stdout.write(f"\nValeurs manquantes : {df.isnull().sum().sum()}\n")
    sys.stdout.write(f"\nAperçu :\n{df.head(3)}\n")
    sys.stdout.write(f"\nStatistiques descriptives :\n{df.describe()}\n")
    
    # Sauvegarde du pairplot
    sns.pairplot(df, hue='species', markers=["o", "s", "D"], plot_kws=dict(alpha=0.7), diag_kind='hist')
    plt.suptitle("Dataset Iris — Séparabilité des classes", y=1.02)
    plt.savefig(os.path.join(output_dir, 'iris_pairplot.png'), bbox_inches='tight')
    plt.close('all')
    sys.stdout.write("[i] iris_pairplot.png sauvegardé.\n")
    
    # ==========================================
    # Étape 2 : Entraînement Baseline & Modèle
    # ==========================================
    le = LabelEncoder()
    # Variables explicatives
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y = le.fit_transform(df['species'])
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Découpage du dataset (Train 80% / Test 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sys.stdout.write(f"\nEntraînement : {len(X_train)} échantillons | Test : {len(X_test)} échantillons\n")
    
    # Standardisation
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Baseline Simple (Decision Tree avec max_depth restreint)
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train_sc, y_train)
    
    # Modèle principal Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train_sc, y_train)
    sys.stdout.write("Modèles entraînés.\n")
    
    # ==========================================
    # Étape 3 : Évaluation
    # ==========================================
    y_pred_dt = dt.predict(X_test_sc)
    y_pred_rf = rf.predict(X_test_sc)
    
    sys.stdout.write("\n=== COMPARAISON BASELINE vs RANDOM FOREST ===\n")
    for nom, pred in [("Decision Tree (baseline)", y_pred_dt), ("Random Forest", y_pred_rf)]:
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted')
        sys.stdout.write(f"{nom} -> Accuracy : {acc*100:.1f}% | F1-score (weighted) : {f1:.3f}\n")
        
    sys.stdout.write("\n=== RAPPORT DÉTAILLÉ — RANDOM FOREST ===\n")
    sys.stdout.write(classification_report(y_test, y_pred_rf, target_names=le.classes_))
    sys.stdout.write("\n")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matrice de confusion — Random Forest')
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close('all')
    sys.stdout.write("[i] confusion_matrix.png sauvegardé.\n")
    
    # ==========================================
    # Étape 4 : Tuning et Overfitting
    # ==========================================
    sys.stdout.write("\n=== TUNING DES HYPERPARAMÈTRES ===\n")
    n_estimators_range = [10, 25, 50, 100, 200, 500]
    for n in n_estimators_range:
        m = RandomForestClassifier(n_estimators=n, random_state=42)
        cv = cross_val_score(m, X_train_sc, y_train, cv=5, scoring='accuracy')
        sys.stdout.write(f"n_estimators={n:4d} -> CV accuracy : {cv.mean()*100:.1f}% (+/-{cv.std()*100:.1f}%)\n")
        
    profondeurs = range(1, 20)
    scores_train = []
    scores_test = []
    for d in profondeurs:
        m = RandomForestClassifier(n_estimators=50, max_depth=d, random_state=42)
        m.fit(X_train_sc, y_train)
        scores_train.append(m.score(X_train_sc, y_train))
        scores_test.append(m.score(X_test_sc, y_test))
        
    plt.figure(figsize=(9, 4))
    plt.plot(profondeurs, scores_train, 'b-o', label='Score entraînement')
    plt.plot(profondeurs, scores_test, 'r-o', label='Score test')
    plt.xlabel('Profondeur maximale (max_depth)')
    plt.ylabel('Accuracy')
    plt.title('Underfitting vs Overfitting — Random Forest')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overfitting.png'))
    plt.close('all')
    sys.stdout.write("[i] overfitting.png sauvegardé.\n")
    
    # ==========================================
    # Étape 5 : Explicabilité avec SHAP
    # ==========================================
    sys.stdout.write("\n=== ANALYSE SHAP ===\n")
    rf_final = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_final.fit(X_train_sc, y_train)
    
    explainer = shap.TreeExplainer(rf_final)
    shap_values = explainer.shap_values(X_test_sc)
    
    plt.figure()
    # Le paramètre show=False empêche l'ouverture d'une fenêtre et permet la suite du code
    shap.summary_plot(shap_values, X_test_sc, feature_names=feature_names, 
                      class_names=list(le.classes_), plot_type='bar', show=False)
    plt.title("SHAP — Importance globale des variables (3 classes)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
    plt.close('all')
    sys.stdout.write("[i] shap_summary.png sauvegardé.\n")
    
    # Comparaison sklearn MDI vs SHAP
    importances_sklearn = rf_final.feature_importances_
    # Compatibilité entre les versions historiques (liste vs array 3D)
    if isinstance(shap_values, list):
        importances_shap = np.abs(shap_values[2]).mean(axis=0) # virginica = index 2
    else:
        importances_shap = np.abs(shap_values[:, :, 2]).mean(axis=0)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.barh(feature_names, importances_sklearn, color='steelblue')
    ax1.set_title('Feature Importance — sklearn (MDI)')
    ax1.set_xlabel('Importance')
    
    ax2.barh(feature_names, importances_shap, color='darkorange')
    ax2.set_title('Feature Importance — SHAP (classe virginica)')
    ax2.set_xlabel('|SHAP value| moyen')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close('all')
    sys.stdout.write("[i] feature_importance.png sauvegardé.\n")
    
    sys.stdout.write("\n>>> Fin de l'exécution: tous les livrables PNG ont été générés dans le sous-dossier outputs/\n")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
