from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
# Load model

models = load("models/models.pkl")

# Create the app
app = FastAPI()

# Define the input structure
class GameData(BaseModel):
    ydstogo: int
    score_diff: int
    game_seconds_remaining: int
    qtr: int
    half_seconds_remaining: int
    yardline_100: int
    posteam_timeouts_remaining: int

@app.get("/")
def root():
    return {"status": "API is up"}

@app.post("/predict")
def predict(data: GameData):
    X = [[
        data.ydstogo,
        data.score_diff,
        data.game_seconds_remaining,
        data.qtr,
        data.half_seconds_remaining,
        data.yardline_100,
        data.posteam_timeouts_remaining
    ]]
    prediction = model.predict(X)[0]
    return {"decision": prediction}
