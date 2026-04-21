from joblib import load

model_inf = load('model_inf_ensemble.joblib')
model_sp  = load('model_sp_calibrated.joblib')

print("model_inf feature names:")
print(list(model_inf.feature_names_in_))

print("\nmodel_sp feature names:")
print(list(model_sp.feature_names_in_))
