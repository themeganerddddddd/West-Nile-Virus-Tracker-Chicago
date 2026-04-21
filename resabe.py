from joblib import load, dump

# Load the model (original file created with Python 3.13+)
original_filename = "model_sp_compatible.joblib"
new_filename = "model_sp_repacked.joblib"

# Load and re-save using current Python environment (3.10 or 3.11)
model = load(original_filename)
dump(model, new_filename)

print(f"Repacked model saved as {new_filename}")
