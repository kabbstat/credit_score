"""
API FastAPI de scoring crédit.
Validation Pydantic, gestion d'erreurs, health check, batch scoring.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config, load_config
from src.logger import get_logger

logger = get_logger("api")


# ─── Global State ──────────────────────────────────────────────

_model = None
_model_metadata: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, libère les ressources à l'arrêt."""
    global _model, _model_metadata

    try:
        load_config()
        config = get_config()
        model_path = config.model.model_path

        if os.path.exists(model_path):
            _model = joblib.load(model_path)
            _model_metadata = {
                "model_path": model_path,
                "loaded_at": datetime.now().isoformat(),
                "model_type": type(_model).__name__,
                "classes": list(_model.classes_) if hasattr(_model, 'classes_') else [],
            }
            logger.info(f"Model loaded: {_model_metadata['model_type']} from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. Run training pipeline first.")
    except Exception as e:
        logger.error(f"Startup error: {e}")

    yield

    logger.info("API shutdown complete")


# ─── Application ───────────────────────────────────────────────

app = FastAPI(
    title="Credit Score Prediction API",
    description=(
        "API de scoring crédit — Pipeline MLOps\n\n"
        "**Auteur**: Mohamed Kabbaj | Data Scientist – Scoring & Détection de Fraude\n\n"
        "**Classes**: Good (FAIBLE risque) | Standard (MOYEN risque) | Poor (ÉLEVÉ risque)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Schémas Pydantic ──────────────────────────────────────────

class CreditFeatures(BaseModel):
    """Features d'un dossier de crédit."""

    # Features critiques
    total_month: Optional[float] = Field(None, description="Historique crédit (mois)", ge=0)
    outstanding_debt: Optional[float] = Field(None, description="Dette en cours (USD)", ge=0)
    num_of_delayed_payment: Optional[float] = Field(None, description="Paiements retardés", ge=0)
    credit_utilization_ratio: Optional[float] = Field(None, description="Taux utilisation crédit (%)", ge=0, le=100)

    # Features importantes
    annual_income: Optional[float] = Field(None, description="Revenu annuel (USD)", ge=0)
    total_emi_per_month: Optional[float] = Field(None, description="EMI mensuel total (USD)", ge=0)
    monthly_inhand_salary: Optional[float] = Field(None, description="Salaire mensuel net (USD)", ge=0)
    delay_from_due_date: Optional[float] = Field(None, description="Retard moyen sur échéances (jours)", ge=0)

    # Features catégorielles
    payment_behaviour: Optional[str] = Field(None, description="Comportement de paiement")
    occupation: Optional[str] = Field(None, description="Profession")
    payment_of_min_amount: Optional[str] = Field(None, description="Paiement du minimum")
    credit_mix: Optional[str] = Field(None, description="Mix de crédit (Standard/Good/Bad)")

    # Features ingéniérées
    debt_to_income: Optional[float] = Field(None, description="Ratio dette/revenu")
    emi_to_income_ratio: Optional[float] = Field(None, description="Ratio EMI/revenu")
    loan_to_income_ratio: Optional[float] = Field(None, description="Ratio prêt/revenu")
    credit_efficiency: Optional[float] = Field(None, description="Score d'efficacité crédit")
    disposable_income_factor: Optional[float] = Field(None, description="Facteur revenu disponible")
    payment_discipline_score: Optional[float] = Field(None, description="Score discipline paiement")

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_month": 36,
                "outstanding_debt": 5000.0,
                "num_of_delayed_payment": 2,
                "credit_utilization_ratio": 30.0,
                "annual_income": 60000.0,
                "monthly_inhand_salary": 4500.0,
                "total_emi_per_month": 800.0,
                "delay_from_due_date": 5.0,
                "payment_behaviour": "High_spent_Small_value_payments",
                "occupation": "Engineer",
                "payment_of_min_amount": "No",
                "credit_mix": "Standard",
                "debt_to_income": 0.083,
                "emi_to_income_ratio": 0.178,
                "loan_to_income_ratio": 1.11,
                "credit_efficiency": 28.5,
                "disposable_income_factor": 0.82,
                "payment_discipline_score": 0.67,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Réponse de scoring crédit."""
    credit_score: str = Field(..., description="Classe prédite: Good | Standard | Poor")
    probabilities: dict = Field(..., description="Probabilités par classe (0-1)")
    confidence: float = Field(..., description="Confiance de la prédiction (max proba)")
    risk_level: str = Field(..., description="Niveau de risque: FAIBLE | MOYEN | ÉLEVÉ")
    timestamp: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Requête de scoring par lot."""
    records: List[CreditFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Réponse de scoring par lot."""
    predictions: List[PredictionResponse]
    n_records: int
    n_good: int
    n_standard: int
    n_poor: int
    timestamp: str


class HealthResponse(BaseModel):
    """Statut de santé de l'API."""
    status: str
    model_loaded: bool
    model_type: str
    timestamp: str


# ─── Helpers ───────────────────────────────────────────────────

_RISK_MAP = {'Good': 'FAIBLE', 'Standard': 'MOYEN', 'Poor': 'ÉLEVÉ'}


def get_model():
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Lancez d'abord: dvc repro model_training",
        )
    return _model


def _predict_one(mdl, features: CreditFeatures) -> PredictionResponse:
    """Effectue une prédiction individuelle."""
    X = pd.DataFrame([features.model_dump()]).fillna(0)

    pred = str(mdl.predict(X)[0])

    try:
        proba = mdl.predict_proba(X)[0]
        classes = list(mdl.classes_)
        probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        confidence = round(float(proba.max()), 4)
    except Exception:
        probabilities = {pred: 1.0}
        confidence = 1.0

    return PredictionResponse(
        credit_score=pred,
        probabilities=probabilities,
        confidence=confidence,
        risk_level=_RISK_MAP.get(pred, 'INCONNU'),
        timestamp=datetime.now().isoformat(),
        model_version=_model_metadata.get("loaded_at", "unknown"),
    )


# ─── Routes ────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "api": "Credit Score Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "POST /predict",
            "batch": "POST /predict/batch",
            "model_info": "/model/info",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Vérification de l'état de l'API et du modèle en mémoire."""
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_type=_model_metadata.get("model_type", "N/A"),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
async def predict(features: CreditFeatures, mdl=Depends(get_model)):
    """
    Scoring crédit d'un dossier individuel.

    - **credit_score**: Classe prédite (Good / Standard / Poor)
    - **probabilities**: Probabilités par classe (interprétabilité)
    - **confidence**: Niveau de confiance du modèle (0-1)
    - **risk_level**: Niveau de risque réglementaire (FAIBLE / MOYEN / ÉLEVÉ)
    """
    try:
        result = _predict_one(mdl, features)
        logger.info(f"Prediction: {result.credit_score} | confidence={result.confidence:.3f}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Scoring"])
async def predict_batch(request: BatchPredictionRequest, mdl=Depends(get_model)):
    """
    Scoring crédit par lot (max 1000 dossiers).
    Optimisé pour le traitement de portefeuilles clients en masse.
    """
    try:
        predictions = [_predict_one(mdl, record) for record in request.records]
        scores = [p.credit_score for p in predictions]

        return BatchPredictionResponse(
            predictions=predictions,
            n_records=len(predictions),
            n_good=scores.count('Good'),
            n_standard=scores.count('Standard'),
            n_poor=scores.count('Poor'),
            timestamp=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction batch: {str(e)}")


@app.get("/model/info", tags=["Monitoring"])
async def model_info(mdl=Depends(get_model)):
    """Informations sur le modèle en production (version, classes, type)."""
    return {
        "model_type": type(mdl).__name__,
        "classes": list(mdl.classes_) if hasattr(mdl, 'classes_') else [],
        "metadata": _model_metadata,
    }


# ─── Lancement standalone ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
