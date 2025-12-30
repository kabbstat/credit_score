# Credit Score Prediction - MLOps Project

[![CI/CD](https://github.com/username/credit-score/workflows/ML%20Pipeline%20CI/CD/badge.svg)](https://github.com/username/credit-score/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## 📋 Description

Ce projet implémente un pipeline MLOps complet pour la prédiction de score de crédit, suivant les meilleures pratiques de l'industrie :

- ✅ **Versioning des données** avec DVC
- ✅ **Tracking des expériences** avec MLflow
- ✅ **Configuration centralisée** avec YAML
- ✅ **Tests unitaires** avec pytest
- ✅ **Validation des données** avec schémas
- ✅ **CI/CD** avec GitHub Actions
- ✅ **Containerisation** avec Docker
- ✅ **Pre-commit hooks** pour la qualité du code

## 🏗️ Structure du Projet

```
credit_score/
├── .github/
│   └── workflows/          # CI/CD pipelines
│       ├── ci.yaml
│       └── model-report.yaml
├── configs/
│   └── config.yaml         # Configuration centralisée
├── src/
│   ├── __init__.py
│   ├── config.py           # Chargement configuration
│   ├── logger.py           # Logging structuré
│   ├── data_collection.py  # Collecte des données
│   ├── data_prepro.py      # Prétraitement
│   ├── train.py            # Entraînement avec MLflow
│   ├── evaluate.py         # Évaluation
│   ├── tracking.py         # MLflow utilities
│   └── validation.py       # Validation des données
├── tests/
│   ├── conftest.py
│   ├── test_data_prepro.py
│   └── test_model_training.py
├── data/
│   ├── raw/                # Données brutes
│   └── processed/          # Données prétraitées
├── models/                 # Modèles entraînés
├── metrics/                # Métriques JSON
├── plots/                  # Visualisations
├── logs/                   # Fichiers de log
├── Dockerfile
├── docker-compose.yaml
├── dvc.yaml               # Pipeline DVC
├── Makefile               # Commandes automatisées
├── pyproject.toml         # Configuration outils Python
├── requirements.txt       # Dépendances
├── .pre-commit-config.yaml
└── README.md
```

## 🚀 Installation

### Prérequis
- Python 3.10+
- Git
- Docker (optionnel)

### Installation rapide

```bash
# Cloner le repository
git clone <repository-url>
cd credit_score

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installer les hooks pre-commit
pip install pre-commit
pre-commit install
```

### Installation avec Make (recommandé)

```bash
make setup-env      # Créer l'environnement
make install-dev    # Installer avec outils de dev
```

## 📊 Dataset

Le projet utilise le dataset de credit scoring d'OpenML (ID: 46441) contenant des informations sur les clients bancaires pour prédire leur score de crédit.

### Features principales :
- **Critiques** : `total_month`, `outstanding_debt`, `num_of_delayed_payment`, `payment_behaviour`, `credit_utilization_ratio`
- **Importantes** : `annual_income`, `total_emi_per_month`, `monthly_inhand_salary`, `delay_from_due_date`, `credit_mix`
- **Engineered** : `debt_to_income`, `emi_to_income_ratio`, `loan_to_income_ratio`, `credit_efficiency`, etc.

## 🔄 Pipeline DVC

Le pipeline est composé de 5 étapes :

```bash
# Exécuter le pipeline complet
dvc repro

# Voir le statut
dvc status

# Voir le DAG
dvc dag

# Voir les métriques
dvc metrics show
```

### Étapes du pipeline :
1. **collection** : Téléchargement des données depuis OpenML
2. **processing** : Nettoyage et feature engineering
3. **validation** : Validation des données avec schémas
4. **model_training** : Entraînement du RandomForest avec MLflow
5. **evaluate** : Calcul des métriques et visualisations

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│ collection │────▶│ processing │────▶│ validation │
└────────────┘     └────────────┘     └─────┬──────┘
                                            │
                   ┌────────────┐     ┌─────▼──────┐
                   │  evaluate  │◀────│  training  │
                   └────────────┘     └────────────┘
```

## 📈 Utilisation

### Avec Make (recommandé)

```bash
make help           # Voir toutes les commandes
make train          # Entraîner le modèle
make evaluate       # Évaluer le modèle
make test           # Lancer les tests
make lint           # Vérifier le code
make format         # Formatter le code
make mlflow         # Lancer MLflow UI
make streamlit      # Lancer le dashboard
make docker-build   # Construire l'image Docker
```

### Sans Make

```bash
# Entraîner le modèle
python src/train.py

# Évaluer le modèle
python src/evaluate.py

# Lancer les tests
pytest tests/ -v

# Lancer MLflow UI
mlflow ui --port 5000

# Lancer Streamlit
streamlit run eda_stream.py
```

## 🐳 Docker

```bash
# Construire l'image
docker build -t credit-score:latest .

# Lancer avec docker-compose
docker-compose up -d

# Services disponibles:
# - MLflow UI: http://localhost:5000
# - Streamlit: http://localhost:8501
```

## 📊 Métriques

Les métriques sont sauvegardées dans `metrics/metrics.json` et incluent :
- Accuracy
- Precision (macro & weighted)
- Recall (macro & weighted)
- F1-Score (macro & weighted)
- ROC AUC
- Cross-validation scores

### Tracking avec MLflow

```bash
# Lancer MLflow UI
make mlflow
# ou
mlflow ui --port 5000
```

Accédez à http://localhost:5000 pour voir :
- Historique des expériences
- Comparaison des métriques
- Paramètres des modèles
- Artifacts (modèles, plots)

## 🧪 Tests

```bash
# Lancer tous les tests
make test

# Avec couverture
make coverage

# Tests en parallèle
make test-fast
```

## 🔒 Qualité du Code

Le projet utilise plusieurs outils pour garantir la qualité :

- **Ruff** : Linting rapide
- **Black** : Formatage du code
- **isort** : Tri des imports
- **pre-commit** : Hooks automatiques
- **Bandit** : Analyse de sécurité

```bash
# Vérifier le code
make check

# Formatter automatiquement
make format

# Lancer pre-commit
make pre-commit
```

## 🛠️ Technologies

| Catégorie | Technologies |
|-----------|-------------|
| **ML** | Scikit-learn, Pandas, NumPy |
| **MLOps** | DVC, MLflow |
| **Web** | Streamlit |
| **Tests** | Pytest, Coverage |
| **CI/CD** | GitHub Actions |
| **Container** | Docker, Docker Compose |
| **Quality** | Ruff, Black, Pre-commit |

## 📝 Auteur

Projet MLOps - Credit Scoring

## 📄 License

MIT License
