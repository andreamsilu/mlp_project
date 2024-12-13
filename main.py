from src.train import train_model
from src.test import evaluate_model

if __name__ == "__main__":
    # Train the model on dataSet1.csv
    print("Training the model...")
    model = train_model()

    # Evaluate the model on dataSet2.csv
    print("\nEvaluating the model...")
    evaluate_model(model)
