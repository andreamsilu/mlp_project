def one_hot_encode(targets, num_classes):
    """
    Convert integer targets to one-hot encoded vectors.
    """
    return [[1 if i == target else 0 for i in range(num_classes)] for target in targets]

def accuracy(predictions, targets):
    """
    Compute accuracy as the fraction of correct predictions.
    """
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets)
