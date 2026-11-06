import pandas as pd
import random

random.seed(42)

tickets = []

categories = {
    "Billing": [
        "I was charged twice for my payment",
        "I need a refund for my order",
        "My invoice amount is incorrect",
        "Payment failed but money was deducted",
        "I have a question about my subscription bill"
    ],

    "Technical": [
        "The application keeps crashing",
        "I cannot log into the website",
        "The page is loading very slowly",
        "I am getting an unexpected error",
        "The software is not working properly"
    ],

    "Account": [
        "I want to reset my password",
        "My account has been locked",
        "I cannot update my profile",
        "I need to change my email address",
        "I cannot access my account"
    ],

    "Shipping": [
        "My order has not arrived",
        "The package is delayed",
        "I received the wrong product",
        "I want to track my shipment",
        "My delivery address is incorrect"
    ],

    "Cancellation": [
        "I want to cancel my order",
        "Please cancel my subscription",
        "I want to stop my service",
        "How can I cancel my purchase?",
        "Please process my cancellation"
    ]
}

priorities = ["Low", "Medium", "High"]

priority_words = {
    "Low": [
        "when possible",
        "not urgent",
        "at your convenience"
    ],
    "Medium": [
        "please help me soon",
        "this is important",
        "I need assistance"
    ],
    "High": [
        "urgent",
        "immediately",
        "this is an emergency"
    ]
}

for i in range(600):

    category = random.choice(list(categories.keys()))

    message = random.choice(categories[category])

    priority = random.choice(priorities)

    extra_text = random.choice(priority_words[priority])

    ticket_text = message + " " + extra_text

    tickets.append({
        "text": ticket_text,
        "category": category,
        "priority": priority
    })

df = pd.DataFrame(tickets)

df.to_csv("data/support_tickets.csv", index=False)

print("Dataset created successfully!")
print("Total tickets:", len(df))

print(df.head())