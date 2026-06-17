"""
=======================================================
 AI Public Transport Crowding Predictor — Hyderabad
 Phase 3: FastAPI Backend
=======================================================

What this file does:
- Loads your trained ML model + encoders (from Phase 2)
- Creates a web API with a /predict endpoint
- Takes route, stop, time, weather etc. as input
- Returns crowd level prediction + alternative route

HOW TO RUN:
  1. Make sure this file is in the SAME folder as:
     crowd_model.pkl, feature_encoders.pkl,
     target_encoder.pkl, feature_columns.pkl, transport_dataset.csv
  2. Open Anaconda Prompt in this folder
  3. Type: uvicorn main:app --reload
  4. Open browser to: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import random
from datetime import datetime

# ─────────────────────────────────────────────────────
# STEP 1: CREATE THE APP
# ─────────────────────────────────────────────────────

app = FastAPI(
    title="Hyderabad Transport Crowding Predictor",
    description="Predicts bus crowd levels for TSRTC routes",
    version="1.0"
)

# CORS lets our future website (running on a different port/domain)
# talk to this backend. Without this, browsers block the request.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # for development; we'll restrict this later
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
# STEP 2: LOAD THE SAVED MODEL + ENCODERS (runs once at startup)
# ─────────────────────────────────────────────────────

print("📦 Loading model and encoders...")

model           = joblib.load("crowd_model.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")
target_encoder  = joblib.load("target_encoder.pkl")
feature_cols    = joblib.load("feature_columns.pkl")

# Load the dataset too — we use it to look up real routes/stops/coords
# and to find "next bus" alternatives
dataset = pd.read_csv("transport_dataset.csv")

print("✅ Model and data loaded successfully!")
print(f"   Routes available: {sorted(dataset['route_number'].unique())}")


# ─────────────────────────────────────────────────────
# STEP 3: DEFINE THE INPUT FORMAT (what the user sends us)
# ─────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    route_number: str       # e.g. "47K"
    stop_name: str           # e.g. "Ameerpet"
    hour: int                # 0-23, e.g. 8 for 8 AM
    day_of_week: int          # 0=Monday ... 6=Sunday
    weather: str = "Clear"   # default value if not provided
    is_holiday: int = 0
    local_event: str = "None"
    vehicle_type: str = "Standard Bus"


# ─────────────────────────────────────────────────────
# STEP 4: HELPER FUNCTION — SAFELY ENCODE USER INPUT
# ─────────────────────────────────────────────────────

def safe_encode(encoder, value, column_name):
    """
    Converts a text value (like 'Heavy Rain') into the number
    the model expects. If the value wasn't seen during training,
    we fall back to the most common known value instead of crashing.
    """
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # Unknown value — fall back to the first known class
        fallback = encoder.classes_[0]
        print(f"⚠️  Unknown '{column_name}' value '{value}', using fallback '{fallback}'")
        return encoder.transform([fallback])[0]


def get_time_period(hour):
    if 5 <= hour < 9:    return "Morning Rush"
    elif 9 <= hour < 12: return "Mid Morning"
    elif 12 <= hour < 15:return "Afternoon"
    elif 15 <= hour < 20:return "Evening Rush"
    elif 20 <= hour < 23:return "Night"
    else:                return "Late Night / Early Morning"


def build_feature_row(req: PredictionRequest):
    """
    Takes the user's request and builds a single row of data,
    encoded exactly the same way our training data was encoded.
    This row is what we feed into model.predict().
    """
    # Look up buses_per_hour for this route from our dataset
    route_data = dataset[dataset['route_number'] == req.route_number]
    if len(route_data) > 0:
        buses_per_hour = int(route_data.iloc[0]['buses_per_hour'])
    else:
        buses_per_hour = 5  # default fallback

    time_period = get_time_period(req.hour)
    is_weekend = 1 if req.day_of_week >= 5 else 0

    row = {
        'route_number':   safe_encode(feature_encoders['route_number'], req.route_number, 'route_number'),
        'stop_name':      safe_encode(feature_encoders['stop_name'], req.stop_name, 'stop_name'),
        'day_of_week':    req.day_of_week,
        'is_weekend':     is_weekend,
        'hour':           req.hour,
        'time_period':    safe_encode(feature_encoders['time_period'], time_period, 'time_period'),
        'vehicle_type':   safe_encode(feature_encoders['vehicle_type'], req.vehicle_type, 'vehicle_type'),
        'buses_per_hour': buses_per_hour,
        'weather':        safe_encode(feature_encoders['weather'], req.weather, 'weather'),
        'is_holiday':     req.is_holiday,
        'local_event':    safe_encode(feature_encoders['local_event'], req.local_event, 'local_event'),
    }

    # Return as a DataFrame with EXACT same column order as training
    return pd.DataFrame([row])[feature_cols]


# ─────────────────────────────────────────────────────
# STEP 5: PREDICTION LOGIC (reusable for main + alt route)
# ─────────────────────────────────────────────────────

def predict_crowd(req: PredictionRequest):
    X = build_feature_row(req)

    # Predict crowd level class
    pred_class = model.predict(X)[0]
    crowd_level = target_encoder.inverse_transform([pred_class])[0]

    # Get prediction probabilities (confidence scores)
    probs = model.predict_proba(X)[0]
    confidence = float(max(probs)) * 100

    # Estimate an occupancy % based on crowd level
    # (rough midpoint of each band, with the model's confidence influencing it)
    occupancy_ranges = {"Low": (10, 30), "Medium": (31, 70), "High": (71, 98)}
    low, high = occupancy_ranges.get(crowd_level, (31, 70))
    estimated_occupancy = int(low + (high - low) * random.random())

    return {
        "crowd_level": crowd_level,
        "estimated_occupancy_pct": estimated_occupancy,
        "confidence_pct": round(confidence, 1)
    }


def find_alternative(req: PredictionRequest, current_result):
    """
    Looks at other routes passing near similar conditions and
    suggests the one with the lowest predicted crowd level.
    """
    all_routes = [r for r in dataset['route_number'].unique() if r != req.route_number]
    random.shuffle(all_routes)

    best_alt = None
    for alt_route in all_routes[:5]:   # check up to 5 alternatives for speed
        # Use a stop from that route (pick its first available stop)
        alt_stops = dataset[dataset['route_number'] == alt_route]['stop_name'].unique()
        if len(alt_stops) == 0:
            continue
        alt_stop = alt_stops[0]

        alt_req = PredictionRequest(
            route_number=alt_route,
            stop_name=alt_stop,
            hour=req.hour,
            day_of_week=req.day_of_week,
            weather=req.weather,
            is_holiday=req.is_holiday,
            local_event=req.local_event,
            vehicle_type=req.vehicle_type
        )
        alt_result = predict_crowd(alt_req)

        if best_alt is None or alt_result['estimated_occupancy_pct'] < best_alt['estimated_occupancy_pct']:
            best_alt = {
                "route_number": alt_route,
                "stop_name": alt_stop,
                **alt_result
            }

    return best_alt


# ─────────────────────────────────────────────────────
# STEP 6: API ENDPOINTS
# ─────────────────────────────────────────────────────

@app.get("/")
def home():
    """Simple health check — confirms the server is running."""
    return {
        "status": "online",
        "message": "Hyderabad Transport Crowding Predictor API is running!",
        "docs": "Visit /docs to test the API"
    }


@app.get("/routes")
def get_routes():
    """Returns all available route numbers — used to populate dropdowns in the UI."""
    return {"routes": sorted(dataset['route_number'].unique().tolist())}


@app.get("/stops/{route_number}")
def get_stops_for_route(route_number: str):
    """Returns all stops for a given route — used to populate stop dropdown."""
    stops = dataset[dataset['route_number'] == route_number]['stop_name'].unique().tolist()
    if not stops:
        raise HTTPException(status_code=404, detail=f"Route '{route_number}' not found")
    return {"route_number": route_number, "stops": stops}


@app.post("/predict")
def predict(req: PredictionRequest):
    """
    Main prediction endpoint.
    Takes route, stop, time, and conditions → returns crowd prediction
    PLUS a suggested alternative if crowding is high.
    """
    result = predict_crowd(req)

    response = {
        "route_number": req.route_number,
        "stop_name": req.stop_name,
        "time": f"{req.hour:02d}:00",
        "prediction": result,
    }

    # Only suggest an alternative if the main route is crowded
    if result["crowd_level"] in ["Medium", "High"]:
        alternative = find_alternative(req, result)
        response["alternative_suggestion"] = alternative
    else:
        response["alternative_suggestion"] = None

    return response


# ─────────────────────────────────────────────────────
# STEP 7: RUN THE SERVER (only used if running this file directly)
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
