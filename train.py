import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load datasets
train = pd.read_csv("data/train.csv")

validation = pd.read_csv("data/validation.csv")

# Input text
X_train = train["text"]
X_val = validation["text"]

# Two target labels
y_category_train = train["category"]
y_category_val = validation["category"]

y_priority_train = train["priority"]
y_priority_val = validation["priority"]


# Function to train model
def train_model(X_train, y_train, X_val, y_val, model_name):

    # Create pipeline
    pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5)
    )),

    ("model", LogisticRegression(max_iter=1000))
])

    # Parameters to test
    parameters = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "model__C": [0.1, 1, 10]
    }

    # Cross-validation
    grid = GridSearchCV(
        pipeline,
        parameters,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    # Train model
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Predict validation data
    predictions = best_model.predict(X_val)

    # Evaluate
    accuracy = accuracy_score(y_val, predictions)

    f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    print("\n-----------------------------")
    print(model_name)
    print("-----------------------------")

    print("Best parameters:", grid.best_params_)

    print("Accuracy:", accuracy)

    print("Macro F1:", f1)

    print("\nClassification Report:")
    print(classification_report(y_val, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, predictions))

    # Save model
    joblib.dump(
        best_model,
        f"models/{model_name}_model.pkl"
    )

    print("Model saved successfully!")

    return best_model


# Train category model
category_model = train_model(
    X_train,
    y_category_train,
    X_val,
    y_category_val,
    "category"
)


# Train priority model
priority_model = train_model(
    X_train,
    y_priority_train,
    X_val,
    y_priority_val,
    "priority"
)
# Find incorrect predictions for category
predictions = category_model.predict(X_val)

errors = validation.copy()

errors["actual"] = y_category_val.values

errors["predicted"] = predictions

errors = errors[
    errors["actual"] != errors["predicted"]
]

print("\n10 Confusing Category Tickets:")

print(
    errors[
        ["text", "actual", "predicted"]
    ].head(10)
)

# Save errors
errors.head(10).to_csv(
    "data/confusing_tickets.csv",
    index=False
)