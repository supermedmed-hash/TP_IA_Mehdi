# Étape 6 : Conformité AI Act & RGPD

## Tableau de Conformité

| Critère | Analyse |
|---|---|
| **Niveau de risque AI Act** | **Risque limité**. Le système fait du profilage commercial (identifier les clients à risque de départ), mais ne prend pas de décisions ayant un impact juridique direct sur les personnes. |
| **Justification du niveau** | On est dans un cas de marketing ciblé et de CRM. Le modèle ne refuse pas un service, il identifie des clients pour leur proposer une offre de rétention. Ce n'est pas comparable à un refus de crédit ou une décision de santé. |
| **Base légale RGPD** | **Intérêt légitime** (Article 6.1.f). L'opérateur télécom a un intérêt commercial légitime à fidéliser ses clients. Alternativement, le **contrat** (Article 6.1.b) peut aussi s'appliquer si le client a accepté d'être contacté dans les CGV de son abonnement. |
| **DPIA requis ?** | **Oui, recommandé**. Même si le risque est "limité", on fait du profilage automatisé à grande échelle (7 000+ clients). La CNIL recommande une Analyse d'Impact (DPIA) dès qu'il y a du profilage systématique. |
| **Explicabilité requise** | **Oui**. Le client a le droit de savoir pourquoi il a été ciblé par une campagne de rétention (article 22 RGPD). SHAP permet de fournir cette explication : "Votre contrat mensuel et votre ancienneté faible indiquent un risque de départ." |
| **Supervision humaine** | **Recommandée**. L'idéal est qu'un conseiller commercial valide la liste des clients ciblés avant d'envoyer des offres. On ne laisse pas l'algorithme envoyer des promos tout seul sans contrôle humain. |
| **Audit et traçabilité** | Il faudrait versionner les modèles (ex: avec MLflow), logger chaque prédiction avec un horodatage, et garder une trace du dataset d'entraînement utilisé. En cas de contrôle CNIL, on doit pouvoir tout retracer. |
| **Droits des personnes** | Les clients ont un droit d'accès (savoir qu'un algorithme les a profilés), un droit de rectification (corriger leurs données), et un droit d'opposition (refuser le profilage automatisé). |

---

## Questions complémentaires

### Le système nécessite-t-il un audit de conformité avant déploiement ?
Oui, un audit interne est fortement recommandé. Le DPO (Délégué à la Protection des Données) de l'entreprise devrait valider le modèle avant mise en production. Pour les opérateurs télécoms en France, c'est la **CNIL** qui contrôle ces pratiques, et l'**ARCEP** qui surveille le secteur des télécoms.

### Les personnes affectées ont-elles un droit de recours ?
Oui. Selon l'article 22 du RGPD, toute personne a le droit de ne pas être soumise à une décision fondée exclusivement sur un traitement automatisé. Le client peut contester le fait d'avoir été "étiqueté" comme étant à risque de churn et demander une intervention humaine.

### Quel organisme de contrôle surveille ce type de système ?
- **CNIL** : Pour tout ce qui concerne les données personnelles et le profilage (RGPD).
- **ARCEP** : Pour la régulation du secteur des télécommunications.
- À terme, avec l'entrée en vigueur totale de l'**AI Act**, un organisme européen dédié pourra aussi intervenir sur la transparence des systèmes d'IA.
