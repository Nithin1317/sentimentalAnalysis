import joblib

# Assuming 'model' is your trained machine learning model
# Replace 'model' with the actual name of your model
model = "pymodel.py"

# Specify the file name to save the model
model_file_name = 'model.joblib'

# Save the model to a file
joblib.dump(model, model_file_name)

print(f'Model saved as {model_file_name}')
