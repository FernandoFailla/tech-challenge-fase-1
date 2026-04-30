from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, populate_by_name=True
    )

    customer_id: str = Field(
        ..., alias="customerID", description="ID unico do cliente"
    )
    gender: str = Field(..., alias="gender")
    senior_citizen: int = Field(..., alias="SeniorCitizen", ge=0, le=1)
    partner: str = Field(..., alias="Partner")
    dependents: str = Field(..., alias="Dependents")
    tenure: int = Field(
        ...,
        ge=0,
        le=120,
        description="Numero de meses que o cliente permaneceu na empresa",
    )
    phone_service: str = Field(..., alias="PhoneService")
    multiple_lines: str = Field(..., alias="MultipleLines")
    internet_service: str = Field(..., alias="InternetService")
    online_security: str = Field(..., alias="OnlineSecurity")
    online_backup: str = Field(..., alias="OnlineBackup")
    device_protection: str = Field(..., alias="DeviceProtection")
    tech_support: str = Field(..., alias="TechSupport")
    streaming_tv: str = Field(..., alias="StreamingTV")
    streaming_movies: str = Field(..., alias="StreamingMovies")
    contract: str = Field(
        ...,
        alias="Contract",
        description="Tipo de contrato (ex: Month-to-month)",
    )
    paperless_billing: str = Field(..., alias="PaperlessBilling")
    payment_method: str = Field(..., alias="PaymentMethod")
    monthly_charges: float = Field(
        ...,
        alias="MonthlyCharges",
        description="Valor cobrado mensalmente do cliente",
    )
    total_charges: float | None = Field(
        None,
        alias="TotalCharges",
        description="Valor total cobrado do cliente",
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    churn_probability: float = Field(
        ...,
        description="Probabilidade do cliente cancelar o servico (0.0 a 1.0)",
    )
    churn_prediction: bool = Field(
        ..., description="Predicao binaria de churn (True/False)"
    )
