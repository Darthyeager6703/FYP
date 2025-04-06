import base64
from scipy import io
import io
import torch
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from flask import Flask, request, jsonify
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import matplotlib
# Add import for Kalman filter
from filterpy.kalman import KalmanFilter
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

# Create a directory for offline graphs if it doesn't exist
os.makedirs("static/graphs", exist_ok=True)

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Load dataset
df = pd.read_csv("processed_data.csv", parse_dates=["Time of Reading"])
df.drop(columns=["UV Index"], errors="ignore", inplace=True)
df = df.groupby("Time of Reading").mean().reset_index()
df = df.sort_values("Time of Reading").set_index("Time of Reading").asfreq("h").interpolate()
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

# ✅ Normalize & Prepare Data
scaler = Scaler()
series = TimeSeries.from_dataframe(df, value_cols=df.columns)
series = scaler.fit_transform(series)
train, val = series.split_before(0.8)

# ✅ Load or Train TFT Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "tft_model.pth"

if os.path.exists(model_path):
    choice = input("🤔 A pre-trained model is detected. Use it? yes or no -> ")
    if choice.lower() == "yes":
        model = TFTModel.load(model_path)
else:
    model = TFTModel(
        input_chunk_length=24, output_chunk_length=12, hidden_size=64,
        lstm_layers=1, num_attention_heads=4, dropout=0.1, batch_size=32,
        n_epochs=50, add_relative_index=True, optimizer_kwargs={"lr": 1e-3}
    )
    model.fit(train, val_series=val, verbose=True)
    model.save(model_path)

# ✅ Get TFT Predictions
future_dates = [datetime.today() + timedelta(hours=i) for i in range(24)]
preds = model.predict(n=24)
preds_df = scaler.inverse_transform(preds).pd_dataframe().reset_index()

# ✅ Real-time Data Storage
trend_window = deque(maxlen=30)  # Store last 30 seconds of data
risk_history = deque(maxlen=100)  # Store last 100 risk assessments
vitals_history = deque(maxlen=100)  # Store history of vitals

# Store both raw and filtered values for potential visualization
raw_values = [[] for _ in range(5)]
filtered_values = [[] for _ in range(5)]

# ✅ Initialize vitals as empty (default to 0 if not received)
current_vitals = {}
filtered_vitals = {}

last_risk_assessment = {
    "Risk (%)": 0,
    "Status": "Waiting for data...",
    "timestamp": datetime.now().isoformat(),
    "vitals": {}
}

# ✅ Initialize Kalman Filters for each vital parameter
# Global Kalman filter objects
kf = [
    KalmanFilter(dim_x=2, dim_z=1),  # V0 - Atmospheric Temp
    KalmanFilter(dim_x=2, dim_z=1),  # V1 - Humidity
    KalmanFilter(dim_x=2, dim_z=1),  # V2 - Body Temp
    KalmanFilter(dim_x=2, dim_z=1),  # V3 - SpO2
    KalmanFilter(dim_x=2, dim_z=1)   # V4 - Heart Rate
]

# Set up the Kalman filters with specialized settings for each parameter
def setup_kalman_filters():
    # Custom Kalman filter settings for each parameter
    kf_settings = [
        {"Q": [[1e-5, 0], [0, 1e-5]], "R": 1e-2},  # V0 - Atmospheric Temp (slow variations)
        {"Q": [[5e-5, 0], [0, 5e-5]], "R": 2.5e-2},  # V1 - Humidity (medium variability)
        {"Q": [[1e-5, 0], [0, 1e-5]], "R": 0.3},  # V2 - Body Temp (smooth, precise, ±2°C flexibility)
        {"Q": [[5e-5, 0], [0, 5e-5]], "R": 0.2},  # V3 - SpO2 (±3 tolerance, reacts <95%)
        {"Q": [[1e-6, 0], [0, 1e-6]], "R": 1.0}   # V4 - Heart Rate (rigid, prevents erratic jumps)
    ]

    # Apply settings to each Kalman filter
    for i in range(5):
        kf[i].x = np.array([[0.], [0.]])  # Initial state (position, velocity)
        kf[i].F = np.array([[1., 1.], [0., 1.]])  # State transition matrix
        kf[i].H = np.array([[1., 0.]])  # Measurement function
        kf[i].P *= 1e6  # Large initial uncertainty
        kf[i].R = kf_settings[i]["R"]  # Measurement noise
        kf[i].Q = np.array(kf_settings[i]["Q"])  # Process noise

# Function to filter incoming vital readings through Kalman filters
def filter_vitals(raw_vitals):
    filtered = {}
    
    # Process each vital through its corresponding Kalman filter
    for i in range(5):
        vital_key = f"V{i}"
        value = raw_vitals.get(vital_key, 0)
        
        # Store raw value
        raw_values[i].append(value)
        if len(raw_values[i]) > 100:  # Keep history limited
            raw_values[i].pop(0)
        
        # Apply Kalman filter
        kf[i].predict()
        kf[i].update(value)
        filtered_value = round(float(kf[i].x[0, 0]), 3)  # Get position estimate
        
        # Store filtered value
        filtered_values[i].append(filtered_value)
        if len(filtered_values[i]) > 100:  # Keep history limited
            filtered_values[i].pop(0)
        
        # Add to filtered vitals dictionary
        filtered[vital_key] = filtered_value
    
    return filtered

# ✅ Risk Calculation Based on Trends & Model
def calculate_heatstroke_risk(vitals):
    body_temp = vitals.get("V2", 0)  # Body Temperature (°C)
    heart_rate = vitals.get("V4", 0)  # Heart Rate (bpm)
    humidity = vitals.get("V1", 0)  # Relative Humidity (%)
    spo2 = vitals.get("V3", 0)  # SpO2 (Oxygen Saturation)
    atm_temp = vitals.get("V0", 0)  # Atmospheric Temperature (°C)
    
    # Estimate Solar Radiation (W/m²) from Atmospheric Temperature
    solar_radiation = max(0, (atm_temp - 15) * 50)  # Rough estimation
    
    # Compute risk score based on vitals
    risk_score = (
        (body_temp / 42) * 40 + (heart_rate / 200) * 25 + 
        (humidity / 100) * 20 + (solar_radiation / 1000) * 10 - (spo2 / 100) * 15
    )
    live_risk = min(max(risk_score, 0), 100)
    
    # Compare against TFT predictions
    tft_pred_temp = preds_df["Body Temperature (°C)"].mean()
    deviation = abs(body_temp - tft_pred_temp) / tft_pred_temp * 100 if tft_pred_temp != 0 else 0
    model_adjusted_risk = min(live_risk + deviation, 100)
    
    # Analyze trends
    trend_window.append(model_adjusted_risk)
    moving_avg = np.mean(trend_window) if trend_window else 0
    spikes = sum(1 for r in trend_window if r > moving_avg + 10)  # Count major spikes
    
    # Adjust final risk based on trends
    if moving_avg > 80:
        risk_label = "UNSTABLE 🚨"
    elif spikes > 5 or moving_avg > 60:
        risk_label = "Monitor Closely ⚠️"
    else:
        risk_label = "Stable ✅"
    
    # Store this assessment with timestamp and vitals
    assessment = {
        "Risk (%)": round(model_adjusted_risk, 2),
        "Status": risk_label,
        "timestamp": datetime.now().isoformat(),
        "vitals": vitals.copy()
    }
    
    # Update global variables
    global last_risk_assessment
    last_risk_assessment = assessment
    risk_history.append(assessment)
    
    # Store vitals history for visualization
    vitals_with_timestamp = vitals.copy()
    vitals_with_timestamp["timestamp"] = datetime.now().isoformat()
    vitals_history.append(vitals_with_timestamp)
    
    return assessment

# ✅ Visualization Functions
def generate_feature_importance():
    """Generate feature importance visualization"""
    # In a real implementation, get this from the model
    # Here using placeholder data similar to what TFT would provide
    feature_importance = {
        "Body Temperature (°C)": 0.85,
        "Heart Rate (bpm)": 0.75,
        "SpO2 (%)": 0.65,
        "Humidity (%)": 0.45,
        "Atmospheric Temperature (°C)": 0.40
    }
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    ax.barh([features[i] for i in sorted_idx], [importance[i] for i in sorted_idx], color='skyblue')
    ax.set_title('Feature Importance for Heatstroke Risk Prediction')
    ax.set_xlabel('Importance Score')
    
    # Save offline copy
    plt.tight_layout()
    fig.savefig('static/graphs/feature_importance.png')
    
    # Convert to base64 for API response
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return {
        'image': img_base64,
        'data': feature_importance
    }

def generate_forecast_distribution():
    """Generate forecast distribution visualization"""
    # Extract forecast data
    forecast_values = preds_df["Body Temperature (°C)"].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(forecast_values, kde=True, ax=ax)
    ax.set_title('Distribution of Forecasted Body Temperature')
    ax.set_xlabel('Body Temperature (°C)')
    ax.set_ylabel('Frequency')
    
    # Save offline copy
    plt.tight_layout()
    fig.savefig('static/graphs/forecast_distribution.png')
    
    # Convert to base64 for API response
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Calculate statistics for the distribution
    stats = {
        'mean': float(np.mean(forecast_values)),
        'median': float(np.median(forecast_values)),
        'std': float(np.std(forecast_values)),
        'min': float(np.min(forecast_values)),
        'max': float(np.max(forecast_values)),
    }
    
    return {
        'image': img_base64,
        'data': stats
    }

def generate_trend_rolling_mean():
    """Generate trend visualization using rolling mean"""
    if len(risk_history) < 2:
        return {'image': '', 'data': {'error': 'Not enough data points yet'}}
    
    # Extract data
    timestamps = [datetime.fromisoformat(item["timestamp"]) for item in risk_history]
    risk_values = [item["Risk (%)"] for item in risk_history]
    
    # Calculate rolling mean
    risk_series = pd.Series(risk_values)
    rolling_window = min(5, len(risk_values))  # Adjust window size based on available data
    rolling_mean = risk_series.rolling(window=rolling_window).mean()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timestamps, risk_values, 'o-', label='Risk Values', alpha=0.7)
    ax.plot(timestamps, rolling_mean, 'r-', label=f'Rolling Mean (window={rolling_window})', linewidth=2)
    
    ax.set_title('Trend of Heatstroke Risk with Rolling Mean')
    ax.set_xlabel('Time')
    ax.set_ylabel('Risk (%)')
    ax.legend()
    ax.grid(True)
    
    # Format x-axis to show time properly
    fig.autofmt_xdate()
    
    # Save offline copy
    plt.tight_layout()
    fig.savefig('static/graphs/risk_trend.png')
    
    # Convert to base64 for API response
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    trend_data = {
        'timestamps': [ts.isoformat() for ts in timestamps],
        'risk_values': risk_values,
        'rolling_mean': rolling_mean.fillna(0).tolist()
    }
    
    return {
        'image': img_base64,
        'data': trend_data
    }

def generate_volatility():
    """Generate volatility visualization using moving standard deviation"""
    if len(risk_history) < 2:
        return {'image': '', 'data': {'error': 'Not enough data points yet'}}
    
    # Extract data
    timestamps = [datetime.fromisoformat(item["timestamp"]) for item in risk_history]
    risk_values = [item["Risk (%)"] for item in risk_history]
    
    # Calculate moving standard deviation (volatility)
    risk_series = pd.Series(risk_values)
    rolling_window = min(5, len(risk_values))  # Adjust window size based on available data
    volatility = risk_series.rolling(window=rolling_window).std()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timestamps, risk_values, 'o-', label='Risk Values', alpha=0.5)
    ax.plot(timestamps, volatility, 'g-', label=f'Volatility (Moving STD, window={rolling_window})', linewidth=2)
    
    ax.set_title('Volatility of Heatstroke Risk')
    ax.set_xlabel('Time')
    ax.set_ylabel('Risk (%) / Standard Deviation')
    ax.legend()
    ax.grid(True)
    
    # Format x-axis to show time properly
    fig.autofmt_xdate()
    
    # Save offline copy
    plt.tight_layout()
    fig.savefig('static/graphs/risk_volatility.png')
    
    # Convert to base64 for API response
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    volatility_data = {
        'timestamps': [ts.isoformat() for ts in timestamps],
        'risk_values': risk_values,
        'volatility': volatility.fillna(0).tolist()
    }
    
    return {
        'image': img_base64,
        'data': volatility_data
    }

def generate_vitals_plot():
    """Generate plot of input vitals over time"""
    if len(vitals_history) < 2:
        return {'image': '', 'data': {'error': 'Not enough data points yet'}}
    
    # Extract data
    timestamps = [datetime.fromisoformat(item["timestamp"]) for item in vitals_history]
    body_temp = [item["V2"] for item in vitals_history]
    heart_rate = [item["V4"] for item in vitals_history]
    spo2 = [item["V3"] for item in vitals_history]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(timestamps, body_temp, 'r-', label='Body Temperature (°C)', linewidth=2)
    ax.set_ylabel('Body Temperature (°C)', color='r')
    ax.tick_params(axis='y', labelcolor='r')
    
    # Add second y-axis for heart rate
    ax2 = ax.twinx()
    ax2.plot(timestamps, heart_rate, 'b-', label='Heart Rate (bpm)', linewidth=2)
    ax2.set_ylabel('Heart Rate (bpm)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    # Add third y-axis for SpO2
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(timestamps, spo2, 'g-', label='SpO2 (%)', linewidth=2)
    ax3.set_ylabel('SpO2 (%)', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    
    ax.set_title('Vital Signs Over Time')
    ax.set_xlabel('Time')
    
    # Add combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
    
    # Format x-axis to show time properly
    fig.autofmt_xdate()
    
    # Save offline copy
    plt.tight_layout()
    fig.savefig('static/graphs/vitals_plot.png')
    
    # Convert to base64 for API response
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    vitals_data = {
        'timestamps': [ts.isoformat() for ts in timestamps],
        'body_temperature': body_temp,
        'heart_rate': heart_rate,
        'spo2': spo2
    }
    
    return {
        'image': img_base64,
        'data': vitals_data
    }

def generate_risk_plot():
    """Generate plot of risks over time"""
    if len(risk_history) < 2:
        return {'image': '', 'data': {'error': 'Not enough data points yet'}}
    
    # Extract data
    timestamps = [datetime.fromisoformat(item["timestamp"]) for item in risk_history]
    risk_values = [item["Risk (%)"] for item in risk_history]
    statuses = [item["Status"] for item in risk_history]
    
    # Determine color based on risk status
    colors = []
    for status in statuses:
        if "Screwed" in status:
            colors.append('red')
        elif "Monitor" in status:
            colors.append('orange')
        else:
            colors.append('green')
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot line
    ax.plot(timestamps, risk_values, '-', color='gray', alpha=0.5)
    
    # Plot scatter points with colors
    for i, (ts, risk) in enumerate(zip(timestamps, risk_values)):
        ax.scatter(ts, risk, color=colors[i], s=50, alpha=0.7)
    
    # Add warning threshold lines
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
    
    ax.set_title('Heatstroke Risk Assessment Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Risk (%)')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True)
    
    # Format x-axis to show time properly
    fig.autofmt_xdate()
    
    # Save offline copy
    plt.tight_layout()
    fig.savefig('static/graphs/risk_plot.png')
    
    # Convert to base64 for API response
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    risk_data = {
        'timestamps': [ts.isoformat() for ts in timestamps],
        'risk_values': risk_values,
        'statuses': statuses
    }
    
    return {
        'image': img_base64,
        'data': risk_data
    }

# ✅ Flask Endpoints
@app.route("/predictions", methods=["GET"])
def get_predictions():
    return jsonify(preds_df.to_dict(orient="records"))

@app.route("/risk_assessment", methods=["GET"])
def get_risk_assessment():
    # Return the most recent risk assessment
    return jsonify(last_risk_assessment)

@app.route("/risk_history", methods=["GET"])
def get_risk_history():
    # Return the history of risk assessments
    return jsonify(list(risk_history))

@app.route("/graph_data", methods=["GET"])
def get_graph_data():
    # Always generate all graphs - no options
    all_graphs = {
        'feature_importance': generate_feature_importance(),
        'forecast_distribution': generate_forecast_distribution(),
        'trend': generate_trend_rolling_mean(),
        'volatility': generate_volatility(),
        'vitals': generate_vitals_plot(),
        'risk': generate_risk_plot()
    }
    return jsonify(all_graphs)

@app.route("/blynk_data", methods=["POST"])
def receive_blynk_data():
    try:
        # Update current vitals with the received data
        global current_vitals, filtered_vitals
        current_vitals.update(request.json)
        
        # Apply Kalman filtering to the raw vitals
        filtered_vitals = filter_vitals(current_vitals)
        
        # Calculate risk with the filtered vitals
        risk = calculate_heatstroke_risk(filtered_vitals)
        
        # Return both raw and filtered values along with risk
        return jsonify({
            "status": "success", 
            "risk": risk,
            "raw_vitals": current_vitals,
            "filtered_vitals": filtered_vitals
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Add endpoint to get the raw vs filtered data for visualization
@app.route("/filter_data", methods=["GET"])
def get_filter_data():
    return jsonify({
        "raw_values": raw_values,
        "filtered_values": filtered_values
    })

if __name__ == "__main__":
    # Initialize the Kalman filters
    setup_kalman_filters()
    
    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)
