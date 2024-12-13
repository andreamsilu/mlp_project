import csv

def load_data(file_path):
    """
    Load data from a CSV file.
    Each row represents features followed by the target label.
    """
    data, targets = [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(x) for x in row[:-1]])  # Features
            targets.append(int(row[-1]))             # Target
    return data, targets

def normalize_data(data):
    """
    Normalize data to range [0, 1].
    """
    min_val = min(min(row) for row in data)
    max_val = max(max(row) for row in data)
    return [[(x - min_val) / (max_val - min_val) for x in row] for row in data]