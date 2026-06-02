# Diabetes Health Indicators Dataset

## Description
Dataset issu de l'enquête CDC BRFSS 2015. Contient 253 680 réponses
de patients américains avec des indicateurs de santé liés au diabète.

## Fichiers
| Fichier | Lignes | Description |
|---------|--------|-------------|
| diabetes_binary_health_indicators_BRFSS2015.csv | 253 680 | variable cible binaire |
| diabetes_binary_5050split_health_indicators_BRFSS2015.csv | 70 692 | dataset équilibré 50/50 |
| diabetes_012_health_indicators_BRFSS2015.csv | 253 680 | 3 classes (0/1/2) |

## Colonnes principales
| Colonne | Type | Description |
|---------|------|-------------|
| Diabetes_binary | int | 0 = non diabétique, 1 = diabétique |
| HighBP | int | Hypertension artérielle |
| HighChol | int | Cholestérol élevé |
| BMI | float | Indice de masse corporelle |
| Smoker | int | Fumeur (1 = oui) |
| PhysActivity | int | Activité physique (1 = oui) |
| Age | int | Tranche d'âge (1-13) |
| Income | int | Niveau de revenu (1-8) |
| Education | int | Niveau d'éducation (1-6) |

## Source
CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

## Licence
CC0 1.0 Public Domain