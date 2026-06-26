from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="💳 Fraud Detection API",
    description="""
## Two ways to test:

### 1️⃣ /predict/simple — For normal users
Fill basic transaction details — get instant fraud prediction.
⚠️ Note: Rule-based approximate result.

### 2️⃣ /predict/technical — For accurate ML results
Provide V1-V28 PCA features — 100% ML model result.
""",
    version="3.0.0"
)

model = joblib.load("fraud_model_tuned.pkl")
amount_scaler = joblib.load("amount_scaler.pkl")
time_scaler = joblib.load("time_scaler.pkl")
feature_names = joblib.load("feature_names.pkl")


# ── Schema 1: Simple ──
class SimpleTransaction(BaseModel):
    amount: float = Field(..., example=1500.00,
        description="Transaction amount in rupees")
    transaction_hour: int = Field(..., example=14,
        description="Hour of transaction (0-23)")
    merchant_category: Literal["grocery","restaurant","online","atm","travel","other"] = Field(
        ..., example="online",
        description="Where did transaction happen?")
    transaction_type: Literal["purchase","withdrawal","transfer"] = Field(
        ..., example="purchase",
        description="Type of transaction")
    location_match: bool = Field(..., example=True,
        description="Did transaction happen at registered location?")
    is_foreign_transaction: bool = Field(..., example=False,
        description="Was this from a foreign country?")
    previous_fraud_count: int = Field(..., example=0,
        description="How many times fraud occurred before?")
    avg_monthly_spend: float = Field(..., example=5000.00,
        description="Your average monthly spending in rupees")
    card_age_days: int = Field(..., example=365,
        description="How many days old is your card?")
    is_round_amount: Optional[bool] = Field(None, example=False,
        description="Is amount a round figure? Leave empty to auto-detect")


# ── Schema 2: Technical ──
class TechnicalTransaction(BaseModel):
    Time: float = Field(..., example=0.0)
    V1: float = Field(..., example=-1.35)
    V2: float = Field(..., example=-0.07)
    V3: float = Field(..., example=2.53)
    V4: float = Field(..., example=1.37)
    V5: float = Field(..., example=-0.33)
    V6: float = Field(..., example=0.46)
    V7: float = Field(..., example=0.23)
    V8: float = Field(..., example=0.09)
    V9: float = Field(..., example=0.36)
    V10: float = Field(..., example=0.09)
    V11: float = Field(..., example=-0.55)
    V12: float = Field(..., example=-0.61)
    V13: float = Field(..., example=-0.99)
    V14: float = Field(..., example=-0.31)
    V15: float = Field(..., example=1.46)
    V16: float = Field(..., example=-0.47)
    V17: float = Field(..., example=0.20)
    V18: float = Field(..., example=0.02)
    V19: float = Field(..., example=0.40)
    V20: float = Field(..., example=0.25)
    V21: float = Field(..., example=-0.01)
    V22: float = Field(..., example=0.27)
    V23: float = Field(..., example=-0.11)
    V24: float = Field(..., example=0.06)
    V25: float = Field(..., example=0.12)
    V26: float = Field(..., example=-0.18)
    V27: float = Field(..., example=0.13)
    V28: float = Field(..., example=-0.02)
    Amount: float = Field(..., example=149.62)


# ── Home ──
@app.get("/", tags=["Home"])
def home():
    return {
        "message": "💳 Fraud Detection API is running!",
        "version": "3.0.0",
        "endpoints": {
            "simple": "/predict/simple — For normal users",
            "technical": "/predict/technical — For accurate ML results"
        }
    }

@app.get("/app", tags=["Frontend"])
def frontend():
    return FileResponse("index.html")

# ── Endpoint 1: Simple ──
@app.post("/predict/simple", tags=["Normal User"])
def predict_simple(data: SimpleTransaction):

    # Basic validations
    if data.amount <= 0:
        raise HTTPException(status_code=400,
            detail="❌ Amount must be greater than 0!")
    if data.amount > 50000:
        raise HTTPException(status_code=400,
            detail="❌ Amount cannot exceed ₹50,000!")
    if data.transaction_hour < 0 or data.transaction_hour > 23:
        raise HTTPException(status_code=400,
            detail="❌ Transaction hour must be between 0-23!")
    if data.card_age_days < 0:
        raise HTTPException(status_code=400,
            detail="❌ Card age cannot be negative!")
    if data.avg_monthly_spend <= 0:
        raise HTTPException(status_code=400,
            detail="❌ Average monthly spend must be greater than 0!")

    # ── Risk Score Calculation ──
    risk_score = 0
    risk_reasons = []

    # 1. Location risk
    if not data.location_match:
        risk_score += 30
        risk_reasons.append("⚠️ Transaction location mismatch")

    # 2. Foreign transaction risk
    if data.is_foreign_transaction:
        risk_score += 20
        risk_reasons.append("⚠️ Foreign transaction detected")

    # 3. Unusual hour risk
    if data.transaction_hour >= 23 or data.transaction_hour <= 4:
        risk_score += 20
        risk_reasons.append("⚠️ Transaction at unusual hour (late night)")

    # 4. Merchant risk
    if data.merchant_category == "atm":
        risk_score += 10
        risk_reasons.append("⚠️ ATM withdrawal")
    elif data.merchant_category == "online":
        risk_score += 5

    # 5. Transaction type risk
    if data.transaction_type == "withdrawal":
        risk_score += 10
        risk_reasons.append("⚠️ Cash withdrawal")

    # 6. Previous fraud history
    if data.previous_fraud_count >= 1:
        risk_score += data.previous_fraud_count * 15
        risk_reasons.append(f"⚠️ {data.previous_fraud_count} previous fraud(s) on this card")

    # 7. HIGH AMOUNT risk
    if data.amount > 30000:
        risk_score += 15
        risk_reasons.append("⚠️ High value transaction")
    elif data.amount > 10000:
        risk_score += 5

    # ── NEW FEATURE 1: Amount Spike Detection ──
    if data.avg_monthly_spend > 0:
        spend_ratio = data.amount / (data.avg_monthly_spend / 30)
        if spend_ratio > 10:
            risk_score += 25
            risk_reasons.append(
                f"🚨 Amount is {round(spend_ratio)}x higher than your daily average!")
        elif spend_ratio > 5:
            risk_score += 15
            risk_reasons.append(
                f"⚠️ Amount is {round(spend_ratio)}x higher than your daily average")

    # ── NEW FEATURE 2: Round Amount Detection ──
    round_amount = data.is_round_amount
    if round_amount is None:
        round_amount = data.amount % 1000 == 0 and data.amount >= 5000
    if round_amount:
        risk_score += 10
        risk_reasons.append("⚠️ Suspiciously round amount detected")

    # ── NEW FEATURE 3: New Card Detection ──
    if data.card_age_days <= 30:
        risk_score += 20
        risk_reasons.append("🚨 Very new card (less than 1 month old)")
    elif data.card_age_days <= 90:
        risk_score += 10
        risk_reasons.append("⚠️ Relatively new card (less than 3 months old)")

    # ── NEW FEATURE 4: Unusual Hour + Foreign Combo ──
    if data.is_foreign_transaction and (
        data.transaction_hour >= 23 or data.transaction_hour <= 4):
        risk_score += 20
        risk_reasons.append("🚨 Foreign transaction at unusual hour — High Alert!")

    # ── NEW FEATURE 5: Multiple Risk Factors Combo ──
    high_risk_factors = sum([
        not data.location_match,
        data.is_foreign_transaction,
        data.transaction_hour >= 23 or data.transaction_hour <= 4,
        data.previous_fraud_count >= 1,
        data.amount > 20000
    ])
    if high_risk_factors >= 4:
        risk_score += 25
        risk_reasons.append("🚨 Multiple high-risk factors detected simultaneously!")
    elif high_risk_factors >= 3:
        risk_score += 10
        risk_reasons.append("⚠️ Several risk factors present together")

    # ── Risk Level ──
    if risk_score >= 60:
        risk_level = "🔴 High Risk"
        fraud_detected = True
        message = "⚠️ FRAUD Detected!"
        advice = "Contact your bank immediately and block your card!"
    elif risk_score >= 35:
        risk_level = "🟡 Medium Risk"
        fraud_detected = False
        message = "⚠️ Suspicious Transaction!"
        advice = "Be careful — if you did not make this transaction, report it to your bank."
    else:
        risk_level = "🟢 Low Risk"
        fraud_detected = False
        message = "✅ Transaction is Safe"
        advice = "This transaction looks completely safe."

    return {
        "result": {
            "fraud_detected": fraud_detected,
            "message": message,
            "risk_level": risk_level,
            "risk_score": f"{min(risk_score, 100)}/100",
            "advice": advice
        },
        "risk_reasons": risk_reasons if risk_reasons else ["✅ No suspicious activity detected"],
        "transaction_summary": {
            "amount": f"₹{data.amount}",
            "time": f"{data.transaction_hour}:00",
            "merchant": data.merchant_category,
            "type": data.transaction_type,
            "foreign_transaction": "Yes ⚠️" if data.is_foreign_transaction else "No",
            "location_safe": "Yes" if data.location_match else "No ⚠️",
            "card_age": f"{data.card_age_days} days",
            "avg_daily_spend": f"₹{round(data.avg_monthly_spend/30, 2)}",
            "fraud_history": f"{data.previous_fraud_count} time(s)"
        },
        "note": "⚠️ Approximate result. Use /predict/technical for accurate ML model result."
    }


# ── Endpoint 2: Technical ──
@app.post("/predict/technical", tags=["Technical User"])
def predict_technical(data: TechnicalTransaction):

    if data.Amount <= 0:
        raise HTTPException(status_code=400,
            detail="❌ Amount must be greater than 0!")
    if data.Amount > 50000:
        raise HTTPException(status_code=400,
            detail="❌ Amount cannot exceed ₹50,000!")

    input_dict = data.dict()
    input_dict["Amount"] = amount_scaler.transform(
        [[input_dict["Amount"]]])[0][0]
    input_dict["Time"] = time_scaler.transform(
        [[input_dict["Time"]]])[0][0]

    df = pd.DataFrame([input_dict])
    df = df[feature_names]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if probability >= 0.7:
        risk_level = "🔴 High Risk"
        advice = "Contact your bank immediately and block your card!"
    elif probability >= 0.4:
        risk_level = "🟡 Medium Risk"
        advice = "Be careful — if you did not make this transaction, report it to your bank."
    else:
        risk_level = "🟢 Low Risk"
        advice = "This transaction looks completely safe."

    return {
        "result": {
            "fraud_detected": bool(prediction),
            "message": "⚠️ FRAUD Detected!" if prediction == 1 else "✅ Transaction is Safe",
            "risk_level": risk_level,
            "fraud_probability": f"{round(float(probability) * 100, 2)}%",
            "advice": advice
        },
        "note": "✅ Accurate result from ML model."
    }