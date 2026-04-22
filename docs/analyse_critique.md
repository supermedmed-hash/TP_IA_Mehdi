# Étape 4 : Analyse Critique Approfondie

## Lecture de la Matrice de Confusion (XGBoost)

Sur nos 1 407 clients de test :
- **Vrais Négatifs (en haut à gauche)** : Le modèle a correctement prédit que ces clients allaient rester. C'est la majorité.
- **Vrais Positifs (en bas à droite)** : Le modèle a correctement identifié les clients qui allaient partir. C'est ce qu'on cherche !
- **Faux Positifs (en haut à droite)** : Le modèle dit "il va partir" alors que non. Pas dramatique, on lui envoie juste une promo pour rien.
- **Faux Négatifs (en bas à gauche)** : Le modèle dit "il va rester" alors qu'il part. C'est le pire cas ! On perd le client sans avoir tenté de le retenir.

## Quel type d'erreur est le plus coûteux pour nous ?

Dans le secteur télécom, **le faux négatif est beaucoup plus coûteux**. Acquérir un nouveau client coûte en moyenne 5 à 7 fois plus cher que d'en garder un existant. Si on rate un client à risque (faux négatif), c'est une perte sèche de revenus mensuels récurrents. Tandis qu'un faux positif ne coûte qu'une petite action commerciale (appel, email, réduction).

## Résultats des 3 modèles

| Modèle | Accuracy | F1-macro | F1-weighted |
|---|---|---|---|
| Logistic Regression | 80.4% | 0.739 | 0.800 |
| Random Forest | 78.8% | 0.712 | 0.781 |
| XGBoost | 77.8% | 0.709 | 0.776 |

### Analyse
Surprise ! La Régression Logistique simple bat les deux modèles "avancés" sur ce dataset. C'est un résultat qu'on rencontre souvent en entreprise : quand les données sont bien préparées et que les relations entre variables sont assez linéaires, un modèle simple peut largement suffire.

## Le modèle est-il suffisant pour un déploiement réel ?

Honnêtement, un F1-macro de 0.739 c'est un bon départ mais ça ne suffirait pas pour une mise en production directe. On pourrait améliorer les choses en :
- Ajoutant des données supplémentaires (historique d'appels au support, données de navigation sur le site...)
- Testant d'autres techniques de rééquilibrage (SMOTE, class_weight)
- Faisant un vrai GridSearch sur les hyperparamètres
- Mettant en place une supervision humaine sur les cas limites (clients avec un score de churn entre 40% et 60%)
