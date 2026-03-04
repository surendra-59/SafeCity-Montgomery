import pandas as pd
import numpy as np
import requests
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Initializing Proactive Environmental Safety Predictor...")
time.sleep(1)

# 1. Load the unified hazard data and risk zones
print("Loading Comma Separated Values files...")
try:
    hazards_dataframe = pd.read_csv('cleaned_environmental_hazards.csv')
    risk_zones = pd.read_csv('siren_risk_zones.csv')
except FileNotFoundError:
    print("Error: Could not find the necessary Comma Separated Values files. Please run data_processor.py first.")
    exit()

# 2. Prepare data and generate "Safe Days"
print("Preparing historical data and generating safe baseline days...")
hazards_dataframe['date'] = pd.to_datetime(hazards_dataframe['date'], format='mixed')
hazards_dataframe['month'] = hazards_dataframe['date'].dt.month

# Our existing data points are confirmed hazards (Target = 1)
hazards_dataframe['hazard_event'] = 1
hazard_features = hazards_dataframe[['latitude', 'longitude', 'month', 'hazard_event']].copy()

# Generate an equal number of "Safe Days" (Target = 0) with random coordinates and months
num_safe_days = len(hazard_features)
safe_days = pd.DataFrame({
    'latitude': np.random.uniform(hazards_dataframe['latitude'].min(), hazards_dataframe['latitude'].max(), num_safe_days),
    'longitude': np.random.uniform(hazards_dataframe['longitude'].min(), hazards_dataframe['longitude'].max(), num_safe_days),
    'month': np.random.randint(1, 13, num_safe_days),
    'hazard_event': 0
})

# Combine everything into one massive dataset for the Machine Learning model
complete_data = pd.concat([hazard_features, safe_days], ignore_index=True)

features = complete_data[['latitude', 'longitude', 'month']]
target = complete_data['hazard_event']

# 3. The Train/Test Split
print("Splitting data into 80% Training and 20% Testing sets...")
time.sleep(1)
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 4. Train the Model
print("Training the Machine Learning Random Forest algorithm...")
classifier = RandomForestClassifier(n_estimators=50, random_state=42)
classifier.fit(features_train, target_train)

# 5. Evaluate the Model's True Performance on the 20% Test Set
print("\n[SYSTEM TEST] Evaluating model on the hidden 20% of historical data...")
time.sleep(1)
predictions = classifier.predict(features_test)

# Compare the model's predictions against what actually happened
results = pd.DataFrame({'Actual': target_test, 'Predicted': predictions})

# Filter to look specifically at the days where a hazard ACTUALLY occurred
actual_hazards = results[results['Actual'] == 1]
total_real_events = len(actual_hazards)

# Count how many of those real hazards the model successfully predicted
correctly_predicted_events = len(actual_hazards[actual_hazards['Predicted'] == 1])

success_rate = (correctly_predicted_events / total_real_events) * 100

print("-" * 50)
print("📊 HISTORICAL PREDICTION PERFORMANCE REPORT")
print("-" * 50)
print(f"Total actual hazard events hidden in the test set: {total_real_events:,}")
print(f"Events successfully predicted by the model:        {correctly_predicted_events:,}")
print(f"--> TRUE DETECTION RATE:                         {success_rate:.2f}%")
print("-" * 50)
time.sleep(2)

# 6. Identify the most vulnerable zone to monitor
top_risk = risk_zones.iloc[0]
target_address = top_risk['USER_Street_Address']
siren_id = top_risk['USER_Siren_Number']
incident_count = top_risk['historical_incident_count']

# 7. Simulate the Live Storm Trigger based on our Top Risk Zone
print("\n[SYSTEM ALERT] Transitioning to Live Monitoring...")
time.sleep(1)
print("WARNING: Severe thunderstorms predicted. Scanning vulnerable sectors...")
time.sleep(1)

# Grab the coordinates and a high-risk month (e.g., August = 8) for the top risk zone
current_scenario = pd.DataFrame([{
    'latitude': top_risk['Y'] if 'Y' in top_risk else hazards_dataframe['latitude'].median(), 
    'longitude': top_risk['X'] if 'X' in top_risk else hazards_dataframe['longitude'].median(),
    'month': 8 
}])

calculated_probabilities = classifier.predict_proba(current_scenario)
simulated_risk_probability = round(calculated_probabilities[0][1] * 100, 2)

print(f"--> MATCH FOUND: Sector {target_address} shows a {simulated_risk_probability}% probability of structural hazard.")
time.sleep(1)

# 8. Fire the Automated Alert to your communication channel
WEBHOOK_URL = "https://discord.com/api/webhooks/1477852277371834611/duAi9jHBeta_mFeKD197ZPX7Z-aNMG9MjGqvapw6gOQ_o0hMZ0_PBq6B4wRHeK9pCTd0"

payload = {
    "content": "PROACTIVE CITY ALERT: ENVIRONMENTAL HAZARD PREDICTED",
    "embeds": [
        {
            "title": "Dispatch Order: Vulnerable Sector Detected",
            "description": f"**Location:** {target_address} (Siren Node #{siren_id})\n**Risk Level:** CRITICAL ({simulated_risk_probability}% Machine Learning Probability Score)\n**Historical Baseline:** {incident_count} prior incidents.",
            "color": 16711680, 
            "fields": [
                {
                    "name": "Triggering Event", 
                    "value": "Severe rainfall forecast intersecting with high-risk drainage zone."
                },
                {
                    "name": "Recommended Municipal Action", 
                    "value": "Dispatch Ditch Maintenance and Mosquito Spraying Units proactively to this sector before standing water accumulates."
                },
                {
                    "name": "System Confidence",
                    "value": f"Model tested at {success_rate:.2f}% historical accuracy."
                }
            ],
            "footer": {
                "text": "City of Montgomery - Smart Infrastructure Predictive Model"
            }
        }
    ]
}

print("\nTransmitting automated JavaScript Object Notation dispatch order over Hypertext Transfer Protocol...")

response = requests.post(WEBHOOK_URL, json=payload)

if response.status_code == 204:
    print("\nSUCCESS: Dispatch alert successfully delivered to the communication channel!")
else:
    print(f"\nERROR: Failed to send alert. Status code: {response.status_code}")