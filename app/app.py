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
# Define expected input fields
class PlayData(BaseModel):
    ydstogo: int
    score_diff: int
    game_seconds_remaining: int
    qtr: int
    half_seconds_remaining: int
    yardline_100: int
    posteam_timeouts_remaining: int

@app.post("/predict")
def predict(play: PlayData):
    # Convert to a DataFrame
    pbp = pd.DataFrame([play.dict()])

    # Get predictions
    go_ep   = models["go_ep"].predict(pbp)[0]
    go_epa  = models["go_epa"].predict(pbp)[0]
    go_wp   = models["go_wp"].predict(pbp)[0]

    punt_ep   = models["punt_ep"].predict(pbp)[0]
    punt_epa  = models["punt_epa"].predict(pbp)[0]
    punt_wp   = models["punt_wp"].predict(pbp)[0]

    fg_ep   = models["field_goal_ep"].predict(pbp)[0]
    fg_epa  = models["field_goal_epa"].predict(pbp)[0]
    fg_wp   = models["field_goal_wp"].predict(pbp)[0]

    # Decide best play by WP
    options = {
        "Go": go_wp,
        "Punt": punt_wp,
        "Field Goal": fg_wp
    }
    best_play = max(options, key=options.get)

    return {
        "go": {
            "ep": go_ep,
            "epa": go_epa,
            "wp": go_wp
        },
        "punt": {
            "ep": punt_ep,
            "epa": punt_epa,
            "wp": punt_wp
        },
        "field_goal": {
            "ep": fg_ep,
            "epa": fg_epa,
            "wp": fg_wp
        },
        "recommended_play": best_play
    }
