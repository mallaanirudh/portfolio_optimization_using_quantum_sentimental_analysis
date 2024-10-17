import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from projectq import MainEngine
from projectq.ops import Measure, CNOT, Ry

# Load the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\malla\Downloads\creditcard.csv')

# Display basic info about the DataFrame
print(df.shape)
print(df.info())
print(df.describe())

# Prepare features and target variable
X = df.drop('Class', axis=1)  # Features
y = df['Class']  # Target variable

# Scale the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_scaled, y, test_size=0.2, random_state=42)

# Classical model for anomaly detection
isolation_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
isolation_forest.fit(X_train)
y_pred = isolation_forest.predict(X_test)
y_pred_binary = [1 if x == -1 else 0 for x in y_pred]

# Print classical model results
print(classification_report(y_test, y_pred_binary))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_binary))

# Quantum Prediction Function
def quantum_predict(sample):
    num_qubits = len(sample)
    engine = MainEngine()  # Create a ProjectQ main engine
    qubits = engine.allocate_qubit(num_qubits)  # Allocate qubits

    # Angle encoding using RY gates
    for i in range(num_qubits):
        Ry(sample[i])(qubits[i])  # Use RY rotation

    # Apply a simple quantum circuit (Entanglement)
    for i in range(num_qubits - 1):
        CNOT | (qubits[i], qubits[i + 1])

    # Measure the qubits
    results = []
    for i in range(num_qubits):
        Measure | qubits[i]
        results.append(int(qubits[i]))  # Record measurement result

    engine.flush()  # Execute the circuit
    return results

# Generate quantum predictions for the test set
y_pred_quantum = []
for sample in X_test:  # Iterate through the scaled test set
    quantum_result = quantum_predict(sample)
    y_pred_quantum.append(quantum_result)

# Convert quantum predictions to binary (example logic)
final_predictions = []
for classical_pred, quantum_result in zip(y_pred_binary, y_pred_quantum):
    if classical_pred == 1 or sum(quantum_result) > len(quantum_result) / 2:  # Example condition
        final_predictions.append(1)  # Anomaly
    else:
        final_predictions.append(0)  # Normal

# Print combined model results
print("Combined Model Predictions:")
print(classification_report(y_test, final_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, final_predictions))
