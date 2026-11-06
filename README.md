# Support Ticket Classification

## Project Description

This project uses Machine Learning to classify customer
support tickets into categories and priorities.

Two Logistic Regression models are created:

1. Category Classification
2. Priority Classification

## Dataset Description

The dataset contains 600 synthetically generated support tickets.

Each ticket contains three columns:

- text: Customer support message
- category: Ticket category
- priority: Ticket priority

Categories:
- Billing
- Technical
- Account
- Shipping
- Cancellation

Priorities:
- Low
- Medium
- High

Dataset source:
Created using Python with predefined ticket templates.
No external dataset was used.

## Data Cleaning

The following preprocessing steps were performed:

- Converted text to lowercase.
- Removed leading and trailing spaces.
- Removed duplicate tickets.
- Removed missing values.

## Dataset Splitting

The dataset was divided into:

- Training: 70%
- Validation: 15%
- Testing: 15%

Stratified sampling was used to preserve
the combined category and priority distribution.

## Machine Learning Models

TF-IDF Vectorizer and Logistic Regression were used.

Two separate models were trained:

- Category prediction
- Priority prediction

GridSearchCV was used to tune:

- ngram_range
- C

Five-fold cross-validation was performed.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Macro F1 Score
- Confusion Matrix

## Model Saving

Models were saved using Joblib.

Files:

- models/category_model.pkl
- models/priority_model.pkl

## Bonus

Character n-gram TF-IDF features were tested
and compared with word n-gram features.