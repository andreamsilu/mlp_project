from src.data_loader import load_data, normalize_data
from src.model import MLP
from src.utils import split_data, one_hot_encode, accuracy

def two_fold_test():
    """
    Perform two-fold testing on the MLP model.
    """
    # Load and preprocess data
    data1, targets1 = load_data('data/dataSet1.csv')
    data2, targets2 = load_data('data/dataSet2.csv')
    data = data1 + data2
    targets = targets1 + targets2

    data = normalize_data(data)
    targets = one_hot_encode(targets, 10)

    # Split into two folds
    fold1_data, fold2_data, fold1_targets, fold2_targets = split_data(data, targets)

    # Fold 1: Train on Fold 1, Test on Fold 2
    model = train_fold(fold1_data, fold1_targets)
    accuracy1 = test_fold(model, fold2_data, fold2_targets)
    print(f"Fold 1 Accuracy: {accuracy1 * 100:.2f}%")

    # Fold 2: Train on Fold 2, Test on Fold 1
    model = train_fold(fold2_data, fold2_targets)
    accuracy2 = test_fold(model, fold1_data, fold1_targets)
    print(f"Fold 2 Accuracy: {accuracy2 * 100:.2f}%")

    # Average Accuracy
    avg_accuracy = (accuracy1 + accuracy2) / 2
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")

def train_fold(data, targets):
    """
    Train the MLP model on a given fold.
    """
    model = MLP(input_size=len(data[0]), hidden_size=64, output_size=10)
    learning_rate = 0.1
    epochs = 100
    for epoch in range(epochs):
        for inputs, target in zip(data, targets):
            model.feedforward(inputs)
            model.backpropagate(inputs, target, learning_rate)
    return model

def test_fold(model, data, targets):
    """
    Test the MLP model on a given fold.
    """
    correct = 0
    for inputs, target in zip(data, targets):
        predictions = model.feedforward(inputs)
        predicted_label = predictions.index(max(predictions))
        actual_label = target.index(1)
        if predicted_label == actual_label:
            correct += 1
    return correct / len(data)