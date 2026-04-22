# Étape 1 : Cadrage du Projet (Cas A — Churn Clients)

| Dimension | Réponse |
|---|---|
| **Problème métier** | On veut anticiper quels clients sont sur le point de résilier leur abonnement télécom, pour que l'entreprise puisse les contacter et leur proposer une offre de fidélisation avant qu'ils ne partent. |
| **KPI métier** | Taux de rétention client (on veut le faire monter). Concrètement, chaque client qu'on arrive à retenir, c'est du chiffre d'affaires mensuel sauvé. |
| **Variable cible (y)** | La colonne `Churn` qui vaut soit "Yes" (le client est parti), soit "No" (il est resté). On l'encode en 1 et 0. |
| **Type de tâche ML** | Classification binaire (deux résultats possibles : il part ou il reste). |
| **Métrique ML principale** | Le **F1-macro**. L'Accuracy seule serait trompeuse ici car le dataset est déséquilibré (~26% de churn seulement). Le F1-macro donne un poids égal aux deux classes, ce qui est plus juste. |
| **Risques identifiés** | La variable `SeniorCitizen` (personne âgée) et le `gender` pourraient créer un biais. Il ne faudrait pas que le modèle considère qu'être une femme ou un senior augmente le risque de churn juste par corrélation. |
| **Niveau AI Act** | **Risque limité**. Ce n'est pas une décision qui touche directement aux droits des personnes (comme un crédit refusé), mais ça reste du profilage commercial. Il faut quand même informer le client si on utilise un algorithme pour le cibler. |

---

## Questions guidées

### Un faux négatif ou un faux positif est-il plus coûteux ?
- **Faux Négatif** (le modèle dit "il va rester" mais il part) : C'est le plus cher ! On perd un client sans avoir rien tenté. Ça coûte directement du revenu mensuel.
- **Faux Positif** (le modèle dit "il va partir" mais il serait resté) : Moins grave. On lui envoie une promo de fidélisation pour rien, ça coûte un petit budget marketing mais le client reste content.

Conclusion : dans notre cas, on préfère maximiser le **Recall** sur la classe "Churn = Oui" pour ne rater aucun client à risque.

### Quelles populations pourraient être discriminées ?
Si le modèle est biaisé, les personnes âgées (`SeniorCitizen`) ou les personnes seules (`Partner = No`) pourraient être systématiquement ciblées par des offres de rétention. Ça pourrait devenir du profilage abusif si on ne fait pas attention.
