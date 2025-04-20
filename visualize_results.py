import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app.utils.visualition.visualization import DecisionBoundaryVisualizer
from data.generate_data import DataGenerator
from models.train_model import CandidateSelectionModel
import os

def main():
    # 1. Veriyi oluştur veya yükle
    print("Veri hazırlanıyor...")
    generator = DataGenerator()
    
    # Eğer veri dosyası yoksa oluştur
    data_path = 'app/data/processed/candidate_data.csv'
    if not os.path.exists(data_path):
        print("Veri dosyası oluşturuluyor...")
        df = generator.generate_candidate_data()
        generator.save_data(df)
    else:
        print("Mevcut veri dosyası yükleniyor...")
        df = generator.load_data()
    
    # 2. Modeli eğit
    print("Model eğitiliyor...")
    model = CandidateSelectionModel()
    X_train, X_test, y_train, y_test = model.prepare_data(df)
    model.train(X_train, y_train)
    
    # 3. Görselleştiriciyi oluştur
    visualizer = DecisionBoundaryVisualizer()
    
    # 4. Veri dağılımını görselleştir
    print("Veri dağılımı görselleştiriliyor...")
    visualizer.plot_data_distribution(df)
    
    # 5. Karar sınırını görselleştir
    print("Karar sınırı görselleştiriliyor...")
    visualizer.plot_decision_boundary(
        model.model,
        model.scaler,
        X_train,
        y_train,
        "SVM Karar Sınırı (Eğitim Verisi)"
    )

if __name__ == "__main__":
    main() 