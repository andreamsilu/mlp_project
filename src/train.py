from src.model import MLP
from src.data_loader import load_data, normalize_data
from src.utils import one_hot_encode

def train_model():
    """
    Train the MLP model using dataSet1.csv.
    """
    # Load and preprocess data
    data, targets = load_data('data/dataSet1.csv')
    data = normalize_data(data)
    targets = one_hot_encode(targets, 10)

    # Initialize model
    input_size = len(data[0])
    hidden_size = 64
    output_size = 10
    model = MLP(input_size, hidden_size, output_size)

    # Training loop
    learning_rate = 0.1
    epochs = 100
    for epoch in range(epochs):
        total_error = 0
        for inputs, target in zip(data, targets):
            outputs = model.feedforward(inputs)
            model.backpropagate(inputs, target, learning_rate)
            total_error += sum((t - o) ** 2 for t, o in zip(target, outputs))
        print(f"Epoch {epoch + 1}, Error: {total_error / len(data):.6f}")
    return model
