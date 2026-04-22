import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import shap
import scipy.sparse as sp

# Configurer matplotlib pour un rendu en arrière-plan sans GUI
matplotlib.use('Agg')

def main():
    sys.stdout.write("=== DEBUT DU PIPELINE TP2 (CAS A - CHURN) ===\n")
    
    # 0. Setup directories
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # ÉTAPE 2 - PRÉPARATION DES DONNÉES
    sys.stdout.write("\n[1] Chargement des donnees IBM Telco Churn...\n")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    sys.stdout.write(f"Forme initiale : {df.shape}\n")
    
    # Nettoyage
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    sys.stdout.write(f"Apres nettoyage des espaces : {df.shape}\n")
    
    # Encodage (Dummies)
    df_enc = pd.get_dummies(df.drop('customerID', axis=1), drop_first=True)
    X = df_enc.drop('Churn_Yes', axis=1)
    y = df_enc['Churn_Yes'].astype(int)
    feature_names = list(X.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    sys.stdout.write(f"Train : {len(X_train)} | Test : {len(X_test)}\n")
    sys.stdout.write(f"Nombre de features : {len(feature_names)}\n")
    
    # ÉTAPE 3 - MODÉLISATION (3 MODÈLES)
    sys.stdout.write("\n[2] Entrainement des modeles...\n")
    modeles = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
    }
    
    resultats = {}
    for nom, modele in modeles.items():
        modele.fit(X_train_sc, y_train)
        pred = modele.predict(X_test_sc)
        acc = accuracy_score(y_test, pred)
        f1_mac = f1_score(y_test, pred, average='macro')
        f1_wei = f1_score(y_test, pred, average='weighted')
        resultats[nom] = {'accuracy': acc, 'f1_macro': f1_mac, 'f1_weighted': f1_wei}
        sys.stdout.write(f"{nom:22s} -> Accuracy : {acc*100:.1f}% | F1-macro : {f1_mac:.3f} | F1-weighted : {f1_wei:.3f}\n")
        
    df_resultats = pd.DataFrame(resultats).T
    
    # Graphique de comparaison
    df_resultats[['accuracy', 'f1_macro']].plot(kind='bar', figsize=(9, 4))
    plt.title('Comparaison des modeles')
    plt.ylabel('Score')
    plt.xticks(rotation=20)
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparaison_modeles.png'))
    plt.close()
    
    # ÉTAPE 4 - ÉVALUATION XGBOOST ET EVOLUTION RF
    sys.stdout.write("\n[3] Evaluation detaillee de XGBoost...\n")
    NOM_MEILLEUR = "XGBoost"
    meilleur = modeles[NOM_MEILLEUR]
    y_pred = meilleur.predict(X_test_sc)
    
    sys.stdout.write(f"=== RAPPORT XGBoost ===\n")
    sys.stdout.write(classification_report(y_test, y_pred))
    sys.stdout.write("\n")
    
    # Matrice
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de confusion - {NOM_MEILLEUR}')
    plt.ylabel('Realite')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Evolution RF
    sys.stdout.write("\n[4] Trace evolution Random Forest...\n")
    n_estimators_range = [10, 25, 50, 100, 200]
    scores_rf = []
    for n in n_estimators_range:
        rf_temp = RandomForestClassifier(n_estimators=n, random_state=42)
        rf_temp.fit(X_train_sc, y_train)
        pred_temp = rf_temp.predict(X_test_sc)
        scores_rf.append(f1_score(y_test, pred_temp, average='macro'))
        
    plt.figure(figsize=(8, 4))
    plt.plot(n_estimators_range, scores_rf, 'g-o')
    plt.xlabel("Nombre d'arbres (n_estimators)")
    plt.ylabel("F1-score macro")
    plt.title("Evolution des performances - Random Forest")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rf_evolution.png'))
    plt.close()
    
    # ÉTAPE 5 - SHAP
    sys.stdout.write("\n[5] Analyse d'Explicabilite SHAP...\n")
    rf_model = modeles["Random Forest"]
    X_sample = X_test_sc[:200]
    if sp.issparse(X_sample):
        X_sample = X_sample.toarray()
    else:
        X_sample = np.array(X_sample)
        
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
        
    plt.figure()
    shap.summary_plot(
        sv, X_sample,
        feature_names=feature_names,
        max_display=15,
        show=False
    )
    plt.title("SHAP - Top 15 variables (Churn = OUI)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), bbox_inches='tight')
    plt.close('all')
    
    sys.stdout.write("\n>>> Fin de l'execution TP2 ! Livrables sauvegardes dans outputs/\n")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
