from pydantic import BaseModel, Field
from typing import Literal, Optional

class SimpleTransaction(BaseModel):

    # ── Existing fields ──
    amount: float = Field(
        ..., example=1500.00,
        description="Transaction amount in rupees (Required)")

    transaction_hour: int = Field(
        ..., example=14,
        description="Hour of transaction 0-23 (0=midnight, 12=noon)")

    merchant_category: Literal["grocery","restaurant","online","atm","travel","other"] = Field(
        ..., example="online",
        description="Where did transaction happen?")

    transaction_type: Literal["purchase","withdrawal","transfer"] = Field(
        ..., example="purchase",
        description="Type of transaction")

    location_match: bool = Field(
        ..., example=True,
        description="Did transaction happen at registered location?")

    is_foreign_transaction: bool = Field(
        ..., example=False,
        description="Was this from a foreign country?")

    previous_fraud_count: int = Field(
        ..., example=0,
        description="How many times fraud occurred before?")

    # ── New fields ──
    avg_monthly_spend: float = Field(
        ..., example=5000.00,
        description="Your average monthly spending in rupees")

    card_age_days: int = Field(
        ..., example=365,
        description="How many days old is your card? (e.g. 30=1 month, 365=1 year)")

    is_round_amount: Optional[bool] = Field(
        None, example=False,
        description="Is the amount a round figure? e.g. 10000, 20000 (true/false) — leave empty to auto-detect")