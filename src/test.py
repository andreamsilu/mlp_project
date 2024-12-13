from src.data_loader import load_data, normalize_data
from src.utils import accuracy

def evaluate_model(model):
    """
    Evaluate the MLP model using dataSet2.csv.
    """
    # Load test data
    data, targets = load_data('data/dataSet2.csv')
    data = normalize_data(data)

    # Evaluate
    predictions = [model.feedforward(inputs).index(max(model.feedforward(inputs))) for inputs in data]
    acc = accuracy(predictions, targets)
    print(f"Accuracy: {acc * 100:.2f}%")
