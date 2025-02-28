import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(
        ...,
        example=10,
        alias="education-num"
    )
    marital_status: str = Field(
        ...,
        example="Married-civ-spouse",
        alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(
        ...,
        example="United-States",
        alias="native-country"
    )


encoder_path = "model/encoder.pkl"
model_path = "model/model.pkl"

encoder = load_model(encoder_path)
model = load_model(model_path)


app = FastAPI()


@app.get("/")
async def get_root():
    """Return a welcome message."""
    return {"message": "Hello from the API!"}


@app.post("/data/")
async def post_inference(data: Data):
    data_dict = data.dict()
    data_converted = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    df = pd.DataFrame.from_dict(data_converted)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_processed, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=None
    )
    preds = inference(model, X_processed)
    result = apply_label(preds)
    return {"result": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
