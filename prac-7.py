from flask import Flask, render_template_string
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator  # Using Aer instead of BasicAer
import matplotlib.pyplot as plt
import uuid
import os

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>Quantum Hello World</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin: 40px; }
        h1 { color: #333; }
        img { max-width: 500px; margin: 20px auto; display: block; }
    </style>
</head>
<body>
    <h1>Quantum Hello World</h1>
    <p>Basic quantum circuit with one qubit in superposition</p>
    <p>Measurement results (512 shots): {{ result }}</p>
    <img src="/static/{{ histogram_path }}" alt="Measurement Results">
</body>
</html>
"""

@app.route("/")
def quantum_hello():
    # Create a simple quantum circuit
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Apply Hadamard gate (creates superposition)
    qc.measure(0, 0)
    
    # Use AerSimulator
    simulator = AerSimulator()
    job = simulator.run(qc, shots=512)
    result = job.result().get_counts()
    
    # Save the histogram
    os.makedirs("static", exist_ok=True)
    filename = f"hist_{uuid.uuid4().hex[:8]}.png"
    save_path = os.path.join("static", filename)
    
    fig = plot_histogram(result)
    fig.savefig(save_path)
    plt.close(fig)
    
    return render_template_string(HTML_TEMPLATE, result=result, histogram_path=filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)