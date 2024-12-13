# Multilayer Perceptron (MLP) Implementation

This project implements a Multilayer Perceptron (MLP) from scratch using Python without relying on external machine learning libraries. It includes functionality for training the model on a dataset (`dataSet1.csv`) and testing it on another dataset (`dataSet2.csv`).

## **Project Structure**

mlp_project/
├── venv/                  # Virtual environment folder
├── data/                  # Dataset folder
│   ├── dataSet1.csv       # Training dataset
│   ├── dataSet2.csv       # Testing dataset
├── src/                   # Source code folder
│   ├── __init__.py
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # MLP model implementation
│   ├── train.py           # Training logic
│   ├── test.py            # Testing logic
│   ├── utils.py           # Utility functions
├── main.py                # Entry point of the project
├── README.md              # Documentation
├── requirements.txt       # Dependencies list


## **Getting Started**

### **Prerequisites**
- Python 3.8 or higher
- A terminal or IDE with support for virtual environments

### **Setup**
1. Clone the repository:
   git clone <repository-url>
   cd mlp_project


2. Create and activate a virtual environment:
   - **On Linux/macOS:**
     python3 -m venv venv
     source venv/bin/activate
    
   - **On Windows:**
     python -m venv venv
     venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Place your datasets in the `data/` directory:
   - `dataSet1.csv` (for training)
   - `dataSet2.csv` (for testing)

### **Run the Project**
1. Execute the main script:

   python main.py


   The script will:
   - Train the MLP model on `dataSet1.csv`.
   - Evaluate the trained model on `dataSet2.csv`.



## **Project Components**

### **1. `main.py`**
The entry point of the project. It orchestrates the training and evaluation of the MLP.

### **2. `src/data_loader.py`**
Handles data loading and normalization.

### **3. `src/model.py`**
Defines the MLP architecture with feedforward and backpropagation mechanisms.

### **4. `src/train.py`**
Implements the training logic, including error computation and weight updates.

### **5. `src/test.py`**
Evaluates the model's accuracy on a testing dataset.

### **6. `src/utils.py`**
Provides utility functions for one-hot encoding and accuracy computation.

---

## **Dataset Format**
Ensure your datasets (`dataSet1.csv` and `dataSet2.csv`) are formatted as follows:
- Each row represents a sample.
- The first columns are the input features (floats).
- The last column is the target label (integer).


## **How It Works**
1. **Training:**
   - Reads `dataSet1.csv` and normalizes the data.
   - Initializes random weights and biases.
   - Uses feedforward and backpropagation to minimize error.

2. **Evaluation:**
   - Reads `dataSet2.csv` and normalizes the data.
   - Runs the trained model to predict outputs.
   - Calculates the accuracy of predictions.

---

## **Results**
- **Training Progress:** Displays error after each epoch during training.
- **Evaluation Output:** Prints the accuracy of the model on `dataSet2.csv`.

---

## **Customization**
- Modify hyperparameters like the number of hidden neurons, learning rate, and number of epochs in `src/train.py`.
- Update the dataset paths in `main.py` if needed.

---
 

# mlp_project
