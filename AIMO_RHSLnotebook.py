import os
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from kaggle_evaluation import aimo_2_inference_server

# Global variables for the model pipeline
model_pipeline = None

def train_model(training_file: str):
    """
    Train the supervised learning model using the provided data
    """
    global model_pipeline

    # Load the training dataset
    train_data = pd.read_csv(training_file)

    # Separate issues (features) and responses (labels)
    x_train = train_data['problem']
    y_train = train_data['answer']

    # Create the model pipeline
    model_pipeline = Pipeline([
        ("Vectorizer", TfidfVectorizer()), # Text vectorization
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)) # Regressor
    ])

    # Train the model
    model_pipeline.fit(x_train, y_train)
    print("Model trained successfully!")

def solve_problem(problem: str):
    """
    Uses the trained model to predict the answer to the given problem
    """
    global model_pipeline

    # Predict using the trained model
    prediction = model_pipeline.predict([problem])[0]

    # Ensure the result is in the correct format
    return int(prediction) % 1000

# Predictor function required by the competition API
def predict(id_: pl.DataFrame, problem: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """
    Generates a prediction for each problem using the provided data
    """

    try:
        # Unpack values
        id_ = id_.to_series()[0]
        question = problem.to_series()[0]

        # Solve the problem
        prediction = solve_problem(question)

        # Return the prediction in the required format
        return pl.DataFrame({'id': [id_], 'answer': [prediction]})
    except Exception as e:
        print(f"Error generating forecast for ID: {id_}, error: {e}")
        return pl.DataFrame({'id': [id_], 'answer': [0]})
    
# Inference Server Configuration
inference_server = aimo_2_inference_server.AIMO2InferenceServer(predict)

if __name__ == "__main__":
    # Check if it is a local or competition performance
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Train the model with the training data
        train_model('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv')

        # Start the server for the hidden test set
        inference_server.serve()
    else:
        # Train the model with local data
        train_model('train.csv')  # Arquivo de treinamento local

        # Run locally for testing with input set
        inference_server.run_local_gateway((
            '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
        ))