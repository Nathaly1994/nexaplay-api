from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pickle
import numpy as np

# ── Cargar modelo y encoders ──────────────────────────────────────────────────
with open("modelo.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    enc = pickle.load(f)

le_foot = enc["le_foot"]
le_work = enc["le_work"]
le_pos  = enc["le_pos"]

# ── App FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="NexaPlay Predictor API",
    description="Predice el **valor de mercado** de un jugador de FIFA 21 en euros, "
                "basado en sus atributos físicos, técnicos y contractuales.",
    version="0.1.0",
)

# ── Esquema de entrada ────────────────────────────────────────────────────────
class PlayerInput(BaseModel):
    age: int = Field(..., ge=15, le=45, example=24, description="Edad del jugador")
    overall: int = Field(..., ge=40, le=99, example=85, description="Overall rating (40-99)")
    potential: int = Field(..., ge=40, le=99, example=90, description="Potencial máximo (40-99)")
    wage_eur: float = Field(..., ge=0, example=50000, description="Salario semanal en euros")
    international_reputation: int = Field(..., ge=1, le=5, example=3, description="Reputación internacional (1-5)")
    weak_foot: int = Field(..., ge=1, le=5, example=3, description="Pierna débil (1-5)")
    skill_moves: int = Field(..., ge=1, le=5, example=4, description="Habilidades (1-5)")
    pace: float = Field(0, ge=0, le=99, example=80, description="Velocidad (0 para porteros)")
    shooting: float = Field(0, ge=0, le=99, example=82, description="Disparo (0 para porteros)")
    passing: float = Field(0, ge=0, le=99, example=75, description="Pase (0 para porteros)")
    dribbling: float = Field(0, ge=0, le=99, example=86, description="Regate (0 para porteros)")
    defending: float = Field(0, ge=0, le=99, example=40, description="Defensa (0 para porteros)")
    physic: float = Field(0, ge=0, le=99, example=74, description="Físico (0 para porteros)")
    preferred_foot: Literal["Left", "Right"] = Field("Right", example="Right", description="Pie dominante")
    work_rate: str = Field("Medium/ Medium", example="High/ Medium",
                           description="Ritmo de trabajo (ej: High/ Medium, Low/ High)")
    main_position: str = Field(..., example="ST",
                               description="Posición principal (ej: ST, CB, GK, CAM, LW...)")

class PredictionOutput(BaseModel):
    predicted_value_eur: float
    predicted_value_formatted: str
    input_summary: dict

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Status"])
def health():
    """Verifica que la API esté activa."""
    return {"status": "ok", "model": "Random Forest FIFA 21", "version": "0.1.0"}


@app.post("/predict", response_model=PredictionOutput, tags=["Predicción"])
def predict(player: PlayerInput):
    """
    Predice el **valor de mercado en euros** de un jugador FIFA 21.

    - Envía los atributos del jugador en el body.
    - Recibe el valor estimado en euros.
    """
    # Encodear categóricas
    try:
        foot_enc = le_foot.transform([player.preferred_foot])[0]
    except ValueError:
        raise HTTPException(400, f"preferred_foot inválido. Usa: {list(le_foot.classes_)}")

    try:
        work_enc = le_work.transform([player.work_rate])[0]
    except ValueError:
        raise HTTPException(400, f"work_rate inválido. Ejemplos: High/ Medium, Medium/ Medium, Low/ High")

    try:
        pos_enc = le_pos.transform([player.main_position])[0]
    except ValueError:
        raise HTTPException(400, f"Posición inválida. Usa: {list(le_pos.classes_)}")

    # Construir vector de features en el mismo orden del entrenamiento
    X = np.array([[
        player.age,
        player.overall,
        player.potential,
        player.wage_eur,
        player.international_reputation,
        player.weak_foot,
        player.skill_moves,
        player.pace,
        player.shooting,
        player.passing,
        player.dribbling,
        player.defending,
        player.physic,
        foot_enc,
        work_enc,
        pos_enc,
    ]])

    valor = float(model.predict(X)[0])

    # Formateo legible
    if valor >= 1_000_000:
        fmt = f"€{valor/1_000_000:.2f}M"
    else:
        fmt = f"€{valor/1_000:.0f}K"

    return PredictionOutput(
        predicted_value_eur=round(valor, 2),
        predicted_value_formatted=fmt,
        input_summary={
            "name_placeholder": f"Jugador ({player.main_position})",
            "overall": player.overall,
            "potential": player.potential,
            "age": player.age,
        }
    )
