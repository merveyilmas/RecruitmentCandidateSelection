import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd


class DecisionBoundaryVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def plot_decision_boundary(self, 
                             model: SVC, 
                             scaler: StandardScaler,
                             X: np.ndarray, 
                             y: np.ndarray,
                             title: str = "SVM Decision Boundary") -> None:
        """Plot the decision boundary of the SVM model."""
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        # Predict for each point in the mesh grid
        Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        
        # Add labels and title
        plt.xlabel('Experience Years (Scaled)')
        plt.ylabel('Technical Score (Scaled)')
        plt.title(title)
        
        # Add colorbar
        plt.colorbar(label='Class')
        
        plt.show()
    
    def plot_data_distribution(self, df: pd.DataFrame) -> None:
        """Plot the distribution of the data."""
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Experience Years vs Technical Score
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df, x='experience_years', y='technical_score', hue='label')
        plt.title('Experience vs Technical Score')
        plt.xlabel('Experience Years')
        plt.ylabel('Technical Score')
        
        # Plot 2: Distribution of Labels
        plt.subplot(1, 2, 2)
        sns.countplot(data=df, x='label')
        plt.title('Distribution of Hired vs Not Hired')
        plt.xlabel('Label (0: Hired, 1: Not Hired)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Load data
    from data.generate_data import DataGenerator
    generator = DataGenerator()
    df = generator.load_data()
    
    # Create visualizer
    visualizer = DecisionBoundaryVisualizer()
    
    # Plot data distribution
    visualizer.plot_data_distribution(df)
    
    # Load and plot decision boundary
    from models.train_model import CandidateSelectionModel
    model = CandidateSelectionModel()
    X_train, X_test, y_train, y_test = model.prepare_data(df)
    model.train(X_train, y_train)
    
    visualizer.plot_decision_boundary(
        model.model,
        model.scaler,
        X_train,
        y_train,
        "SVM Decision Boundary (Training Data)"
    ) 