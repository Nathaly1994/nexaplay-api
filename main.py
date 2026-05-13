from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pickle
import numpy as np

with open("modelo_v2.pkl", "rb") as f: model_valor = pickle.load(f)
with open("encoders_v2.pkl", "rb") as f: enc_valor = pickle.load(f)
with open("modelo_potencial_v2.pkl", "rb") as f: model_potencial = pickle.load(f)
with open("encoders_potencial_v2.pkl", "rb") as f: enc_potencial = pickle.load(f)
with open("modelo_salario_v2.pkl", "rb") as f: model_salario = pickle.load(f)
with open("encoders_salario_v2.pkl", "rb") as f: enc_salario = pickle.load(f)

app = FastAPI(
    title="NexaPlay Predictor API",
    description="Predice valor de mercado, potencial y salario de jugadores FIFA 21.",
    version="0.3.0",
)

class PlayerInput(BaseModel):
    age: int = Field(..., ge=15, le=45, example=22)
    overall: int = Field(..., ge=40, le=99, example=85)
    potential: int = Field(0, ge=0, le=99, example=90)
    wage_eur: float = Field(0, ge=0, example=80000)
    value_eur: float = Field(0, ge=0, example=50000000)
    international_reputation: int = Field(..., ge=1, le=5, example=3)
    weak_foot: int = Field(..., ge=1, le=5, example=4)
    skill_moves: int = Field(..., ge=1, le=5, example=4)
    pace: float = Field(0, ge=0, le=99, example=90)
    shooting: float = Field(0, ge=0, le=99, example=82)
    passing: float = Field(0, ge=0, le=99, example=78)
    dribbling: float = Field(0, ge=0, le=99, example=88)
    defending: float = Field(0, ge=0, le=99, example=35)
    physic: float = Field(0, ge=0, le=99, example=72)
    preferred_foot: Literal["Left", "Right"] = Field("Right", example="Right")
    work_rate: str = Field("Medium/Medium", example="High/Medium")
    main_position: str = Field(..., example="ST")
    height_cm: float = Field(181, ge=150, le=210, example=181)
    weight_kg: float = Field(75, ge=50, le=120, example=75)
    league_rank: int = Field(1, ge=1, le=5, example=1)
    contract_valid_until: int = Field(2023, ge=2015, le=2030, example=2023)

def encode(data, enc):
    try: foot = enc['le_foot'].transform([data.preferred_foot])[0]
    except: raise HTTPException(400, "preferred_foot invalido. Usa: Left o Right")
    try: work = enc['le_work'].transform([data.work_rate])[0]
    except: raise HTTPException(400, "work_rate invalido. Ej: High/Medium, Medium/Medium")
    try: pos = enc['le_pos'].transform([data.main_position])[0]
    except: raise HTTPException(400, "Posicion invalida. Ej: ST, CB, GK, CAM, LW")
    return foot, work, pos

@app.get("/health", tags=["Status"])
def health():
    return {"status": "ok", "version": "0.3.0", "modelos": ["valor", "potencial", "salario"]}

@app.post("/predict/valor", tags=["Predicciones"])
def predict_valor(data: PlayerInput):
    """Predice el valor de mercado en euros del jugador."""
    foot, work, pos = encode(data, enc_valor)
    X = np.array([[data.age, data.overall, data.potential, data.wage_eur,
        data.international_reputation, data.weak_foot, data.skill_moves,
        data.pace, data.shooting, data.passing, data.dribbling,
        data.defending, data.physic, foot, work,
        data.height_cm, data.weight_kg, data.league_rank,
        data.contract_valid_until, pos]])
    valor = float(model_valor.predict(X)[0])
    fmt = f"EUR{valor/1_000_000:.2f}M" if valor >= 1_000_000 else f"EUR{valor/1_000:.0f}K"
    return {"predicted_value_eur": round(valor, 2), "predicted_value_formatted": fmt}

@app.post("/predict/potencial", tags=["Predicciones"])
def predict_potencial(data: PlayerInput):
    """Predice el potencial maximo del jugador (escala 1-99)."""
    foot, work, pos = encode(data, enc_potencial)
    X = np.array([[data.age, data.overall, data.value_eur, data.wage_eur,
        data.international_reputation, data.weak_foot, data.skill_moves,
        data.pace, data.shooting, data.passing, data.dribbling,
        data.defending, data.physic, foot, work,
        data.height_cm, data.weight_kg, data.league_rank,
        data.contract_valid_until, pos]])
    potencial = float(model_potencial.predict(X)[0])
    return {"predicted_potential": round(potencial, 1), "predicted_potential_formatted": f"{potencial:.1f}/99"}

@app.post("/predict/salario", tags=["Predicciones"])
def predict_salario(data: PlayerInput):
    """Predice el salario semanal en euros del jugador."""
    foot, work, pos = encode(data, enc_salario)
    X = np.array([[data.age, data.overall, data.potential, data.value_eur,
        data.international_reputation, data.weak_foot, data.skill_moves,
        data.pace, data.shooting, data.passing, data.dribbling,
        data.defending, data.physic, foot, work,
        data.height_cm, data.weight_kg, data.league_rank,
        data.contract_valid_until, pos]])
    salario = float(model_salario.predict(X)[0])
    fmt = f"EUR{salario:,.0f}/semana"
    return {"predicted_wage_eur": round(salario, 2), "predicted_wage_formatted": fmt}
