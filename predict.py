import joblib

# Load trained models
category_model = joblib.load("models/category_model.pkl")
priority_model = joblib.load("models/priority_model.pkl")

print("===================================")
print("   SUPPORT TICKET CLASSIFICATION")
print("===================================")

# Get ticket from user
ticket = input("\nEnter your support ticket: ")

# Predict category
category_prediction = category_model.predict([ticket])[0]

# Predict priority
priority_prediction = priority_model.predict([ticket])[0]

# Display two outputs
print("\n========== PREDICTION RESULTS ==========")

print("Ticket:", ticket)

print("Predicted Category:", category_prediction)

print("Predicted Priority:", priority_prediction)

print("========================================")