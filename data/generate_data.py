import numpy as np
import pandas as pd
from faker import Faker
from typing import Tuple
import os
from pathlib import Path

class DataGenerator:
    def __init__(self, seed: int = 42):
        self.faker = Faker()
        self.faker.seed_instance(seed)
        np.random.seed(seed)
        self.base_dir = Path(__file__).parent  # Sadece data klasörünü işaret ediyor
    
    def generate_candidate_data(self, n_samples: int = 200) -> pd.DataFrame:
        """Generate candidate data with experience years and technical score."""
        data = []
        
        for _ in range(n_samples):
            experience_years = np.random.uniform(0, 10)
            technical_score = np.random.uniform(0, 100)
            
            # Label candidates based on rules
            if experience_years < 2 and technical_score < 60:
                label = 1  # Not hired
            else:
                label = 0  # Hired
            
            data.append({
                'experience_years': experience_years,
                'technical_score': technical_score,
                'label': label
            })
        
        return pd.DataFrame(data)
    
    def save_data(self, df: pd.DataFrame, filename: str = 'candidate_data.csv'):
        """Save generated data to CSV file."""
        # Create directory if it doesn't exist
        data_dir = self.base_dir / 'processed'
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_dir / filename, index=False)
    
    def load_data(self, filename: str = 'candidate_data.csv') -> pd.DataFrame:
        """Load data from CSV file."""
        data_path = self.base_dir / 'processed' / filename
        return pd.read_csv(data_path)

if __name__ == "__main__":
    # Generate and save data
    generator = DataGenerator()
    df = generator.generate_candidate_data()
    generator.save_data(df)
    
    print("Data generated and saved successfully!")
    print("\nSample data:")
    print(df.head()) 