import os
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import kaggle_evaluation.aimo_2_inference_server as inference_server

# Preprocess and global model
vectorizer = TfidfVectorizer(max_features=1000)
model = XGBClassifier(n_estimators=100, use_label_encoder=False, objective="multi:softmax", random_state=42)

# Load data
data = pd.read_csv("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv")

# Process problems and answers
data['processed_problem'] = list(vectorizer.fit_transform(data['problem']).toarray())
data['processed_answer'] = data['answer'] % 1000  # Mod 1000 as defined for the competition

# Encode target variable
label_encoder = LabelEncoder()
data['encoded_answer'] = label_encoder.fit_transform(data['processed_answer'])

# Analyse initial distribution
class_counts = data['encoded_answer'].value_counts()
print("Distribuição inicial das classes:")
print(class_counts)

# Increase the data replicating instances (larger number of repetitions to meet conditions)
augmented_data = data.copy()
for _ in range(10):  # Increase the number of repetitions to generate more data
    augmented_data = pd.concat([augmented_data, data.copy()], ignore_index=True)

# Update distribution after increase
class_counts = augmented_data['encoded_answer'].value_counts()
print("Distribution after data augmentation:")
print(class_counts)

# Prepare features and target
X = pd.DataFrame(augmented_data['processed_problem'].tolist())
y = augmented_data['encoded_answer']

# Adjust test_size to avoid problems
test_size = max(10, len(augmented_data) // 10)  # Guarantees at least 10 samples
train_size = len(augmented_data) - test_size

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size / len(augmented_data), random_state=42, stratify=y)

# Verify the size of the set
print(f"Size of train set: {len(X_train)}, Size of validation set: {len(X_val)}")

# Train the model
model.fit(X_train, y_train)

# Validate the model
y_pred_val = model.predict(X_val)

# Custom scoring function
def competition_score(y_true, y_pred):
    correct_count = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    total = len(y_true)
    return (correct_count / total) * 100  # Convert to percentage

score_val = competition_score(y_val, y_pred_val)
print(f"Validation Competition Score: {score_val:.2f}")

# Prediction function
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Makes a prediction"""
    # Extracting values
    id_ = id_.to_pandas().iloc[0, 0]
    question = question.to_pandas().iloc[0, 0]

    # Transforming the question
    question_vector = vectorizer.transform([question]).toarray()

    # Make the prediction
    prediction_encoded = model.predict(question_vector)[0]
    prediction = label_encoder.inverse_transform([int(prediction_encoded)])[0]  # Convert back to original scale

    return pd.DataFrame({'id': [id_], 'answer': [prediction]})

# Inference server
inference_server = inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # Local evaluation
    test_data = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv')
    predictions = []

    for _, row in test_data.iterrows():
        id_ = pl.DataFrame({'id': [row['id']]})
        question = pl.DataFrame({'problem': [row['problem']]})
        prediction = predict(id_, question)
        predictions.append(prediction['answer'].iloc[0])

    # Calculate and show the Kaggle competition score
    test_data['predicted_answer'] = predictions
    test_data['processed_answer'] = test_data['answer'] % 1000

    kaggle_score = competition_score(test_data['processed_answer'], test_data['predicted_answer'])
    print(f"Final Competition Score: {kaggle_score:.2f}")

    # Save the results
    test_data[['id', 'predicted_answer']].to_csv('submission.csv', index=False)