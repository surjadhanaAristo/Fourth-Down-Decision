from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from mangum import Mangum
# Load model
models = load("models/models.pkl")

# Create the app
app = FastAPI()

# Define input schema first!
class PlayData(BaseModel):
    ydstogo: int
    score_differential: int
    game_seconds_remaining: int
    qtr: int
    half_seconds_remaining: int
    posteam_timeouts_remaining: int
    yardline_100: int
    

# Root route to confirm server is up
@app.get("/")
def root():
    return {"status": "API is up"}

# Predict route
@app.post("/predict")
def predict(play: PlayData):
    try:
        print("Received input:", play.dict())

        pbp = pd.DataFrame([play.dict()])
        print("DataFrame:", pbp.to_dict())

        go_ep   = models["go_ep"].predict(pbp)[0]
        go_epa  = models["go_epa"].predict(pbp)[0]
        go_wp   = models["go_wp"].predict(pbp)[0]

        punt_ep   = models["punt_ep"].predict(pbp)[0]
        punt_epa  = models["punt_epa"].predict(pbp)[0]
        punt_wp   = models["punt_wp"].predict(pbp)[0]

        fg_ep   = models["field_goal_ep"].predict(pbp)[0]
        fg_epa  = models["field_goal_epa"].predict(pbp)[0]
        fg_wp   = models["field_goal_wp"].predict(pbp)[0]

        options = {"Go": go_wp, "Punt": punt_wp, "Field Goal": fg_wp}
        best_play = max(options, key=options.get)

        return {
            "go": {"ep": go_ep, "epa": go_epa, "wp": go_wp},
            "punt": {"ep": punt_ep, "epa": punt_epa, "wp": punt_wp},
            "field_goal": {"ep": fg_ep, "epa": fg_epa, "wp": fg_wp},
            "recommended_play": best_play
        }

    except Exception as e:
        print("Prediction error:", str(e))
        return {"error": str(e)}
handler = Mangum(app)