"""
=======================================================
 AI Public Transport Crowding Predictor — Hyderabad
 Phase 4: Streamlit Web Dashboard
=======================================================

What this file does:
- Creates a visual website where users pick route/stop/time
- Calls our FastAPI backend (/predict) to get the prediction
- Displays the result with colors, charts, and a map

HOW TO RUN:
  1. Make sure your FastAPI backend (main.py) is ALREADY RUNNING
     in a separate Anaconda Prompt window (uvicorn main:app --reload)
  2. Put this file in the SAME folder as transport_dataset.csv
  3. Open a NEW Anaconda Prompt window in that folder
  4. Type: streamlit run app.py
  5. It will open automatically in your browser
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────
# STEP 1: PAGE CONFIGURATION (must be the first Streamlit command)
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hyderabad Transport Crowd Predictor",
    page_icon="🚌",
    layout="wide"
)

# Backend API URL — change this later when we deploy online
API_URL = "https://transport-crowd-api.onrender.com"


# ─────────────────────────────────────────────────────
# STEP 2: LOAD DATASET (for dropdown options only)
# ─────────────────────────────────────────────────────

@st.cache_data   # caches this so it doesn't reload every interaction (faster app)
def load_data():
    return pd.read_csv("transport_dataset.csv")

df = load_data()


# ─────────────────────────────────────────────────────
# STEP 3: HEADER / TITLE
# ─────────────────────────────────────────────────────

st.title("🚌 Hyderabad Public Transport Crowd Predictor")
st.markdown("Predict how crowded your bus will be **before** it arrives — powered by AI trained on real TSRTC routes and stops.")
st.divider()


# ─────────────────────────────────────────────────────
# STEP 4: SIDEBAR — USER INPUTS
# ─────────────────────────────────────────────────────

st.sidebar.header("🔍 Plan Your Journey")

# Route dropdown (from real data)
all_routes = sorted(df['route_number'].unique())
selected_route = st.sidebar.selectbox("Select Bus Route", all_routes)

# Stop dropdown (filtered to stops on the selected route)
stops_for_route = sorted(df[df['route_number'] == selected_route]['stop_name'].unique())
selected_stop = st.sidebar.selectbox("Boarding Stop", stops_for_route)

# Time picker
selected_time = st.sidebar.time_input("Time of Travel", value=pd.to_datetime("08:30").time())
selected_hour = selected_time.hour

# Day of week
day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_day = st.sidebar.selectbox("Day of Week", day_options)
day_index = day_options.index(selected_day)

# Weather
weather_options = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Hot & Sunny"]
selected_weather = st.sidebar.selectbox("Weather Condition", weather_options)

# Holiday toggle
is_holiday = st.sidebar.checkbox("Is it a public holiday?")

# Local event
event_options = ["None", "IPL Match at Uppal", "Bonalu Festival",
                  "Ganesh Chaturthi Procession", "Election Day",
                  "HITEX Exhibition", "Concert at Shilpakala",
                  "Dussehra Celebrations"]
selected_event = st.sidebar.selectbox("Any Local Event?", event_options)

# Vehicle type
vehicle_options = ["Standard Bus", "AC Bus", "Mini Bus", "Metro Feeder"]
selected_vehicle = st.sidebar.selectbox("Vehicle Type", vehicle_options)

st.sidebar.divider()
predict_button = st.sidebar.button("🔮 Predict Crowd Level", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────
# STEP 5: HELPER — CALL THE BACKEND API
# ─────────────────────────────────────────────────────

def get_prediction(route, stop, hour, dow, weather, holiday, event, vehicle):
    payload = {
        "route_number": route,
        "stop_name": stop,
        "hour": hour,
        "day_of_week": dow,
        "weather": weather,
        "is_holiday": 1 if holiday else 0,
        "local_event": event,
        "vehicle_type": vehicle
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=70)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ReadTimeout:
        return None, "⏳ The backend is waking up from sleep (free hosting tier). Please click Predict again in a few seconds!"
    except requests.exceptions.ConnectionError:
        return None, "⚠️ Cannot connect to backend. It may be restarting — please try again shortly."
    except Exception as e:
        return None, f"⚠️ Error: {str(e)}"


def crowd_color(level):
    """Returns a color based on crowd level — used for visual badges."""
    return {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}.get(level, "#94a3b8")


# ─────────────────────────────────────────────────────
# STEP 6: MAIN AREA — SHOW PREDICTION RESULTS
# ─────────────────────────────────────────────────────

if predict_button:
    with st.spinner("Analyzing route conditions... (first request may take up to a minute if the server was idle)"):
        result, error = get_prediction(
            selected_route, selected_stop, selected_hour, day_index,
            selected_weather, is_holiday, selected_event, selected_vehicle
        )

    if error:
        st.error(error)
    else:
        pred = result["prediction"]
        crowd_level = pred["crowd_level"]
        occupancy = pred["estimated_occupancy_pct"]
        confidence = pred["confidence_pct"]
        color = crowd_color(crowd_level)

        # ── Main result card ──
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Route", f"{selected_route}")
            st.caption(f"📍 Boarding at {selected_stop}")

        with col2:
            st.metric("Estimated Occupancy", f"{occupancy}%")
            st.caption(f"🕐 at {selected_time.strftime('%I:%M %p')}")

        with col3:
            st.markdown(
                f"""
                <div style="background-color:{color}; padding:10px 20px; border-radius:10px; text-align:center;">
                    <span style="color:white; font-size:24px; font-weight:bold;">{crowd_level} Crowd</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption(f"🎯 Model confidence: {confidence}%")

        st.divider()

        # ── Occupancy gauge chart ──
        st.subheader("📊 Occupancy Level")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=occupancy,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "#dcfce7"},
                    {'range': [30, 70], 'color': "#fef3c7"},
                    {'range': [70, 100], 'color': "#fee2e2"},
                ],
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # ── Alternative route suggestion ──
        if result.get("alternative_suggestion"):
            alt = result["alternative_suggestion"]
            st.divider()
            st.subheader("💡 Suggested Alternative")

            alt_color = crowd_color(alt["crowd_level"])
            st.markdown(
                f"""
                <div style="border:2px solid {alt_color}; padding:15px; border-radius:10px;">
                    <b>Route {alt['route_number']}</b> at <b>{alt['stop_name']}</b><br>
                    Estimated Occupancy: <b>{alt['estimated_occupancy_pct']}%</b> 
                    (<span style="color:{alt_color}; font-weight:bold;">{alt['crowd_level']}</span>)
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.success("✅ This route has good availability — no need for an alternative!")

else:
    st.info("👈 Fill in your journey details in the sidebar and click **Predict Crowd Level** to get started.")


# ─────────────────────────────────────────────────────
# STEP 7: EXTRA TAB — PEAK HOUR ANALYSIS (uses dataset directly)
# ─────────────────────────────────────────────────────

st.divider()
st.subheader("📈 Peak Hour Analysis Across All Routes")

hourly_avg = df.groupby('hour')['occupancy_pct'].mean().reset_index()
fig2 = px.bar(
    hourly_avg, x='hour', y='occupancy_pct',
    labels={'hour': 'Hour of Day', 'occupancy_pct': 'Avg Occupancy %'},
    color='occupancy_pct', color_continuous_scale='Reds'
)
fig2.update_layout(height=350)
st.plotly_chart(fig2, use_container_width=True)

st.caption("Built with FastAPI + XGBoost + Streamlit | Trained on real TSRTC stop data")
