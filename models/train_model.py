import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
from typing import Tuple, Dict, Any

class CandidateSelectionModel:
    def __init__(self, kernel: str = 'linear'):
        self.scaler = StandardScaler()
        self.model = SVC(kernel=kernel)
        self.kernel = kernel
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training."""
        X = df[['experience_years', 'technical_score']].values
        y = df['label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model."""
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, experience_years: float, technical_score: float) -> int:
        """Make prediction for a single candidate."""
        features = np.array([[experience_years, technical_score]])
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)[0]
    
    def save_model(self, model_dir: str = 'models/saved'):
        """Save the trained model and scaler."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, f'{model_dir}/svm_model_{self.kernel}.joblib')
        joblib.dump(self.scaler, f'{model_dir}/scaler_{self.kernel}.joblib')
    
    def load_model(self, model_dir: str = 'models/saved'):
        """Load the trained model and scaler."""
        self.model = joblib.load(f'{model_dir}/svm_model_{self.kernel}.joblib')
        self.scaler = joblib.load(f'{model_dir}/scaler_{self.kernel}.joblib')

if __name__ == "__main__":
    # Load data
    from data.generate_data import DataGenerator
    generator = DataGenerator()
    df = generator.load_data()
    
    # Train and evaluate model
    model = CandidateSelectionModel()
    X_train, X_test, y_train, y_test = model.prepare_data(df)
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model.save_model()
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report']) 