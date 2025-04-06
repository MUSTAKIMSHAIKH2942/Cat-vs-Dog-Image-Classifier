import os
import sys
from src.config import MODEL_PATH
from src.data_processing.data_preprocessor import DataPreprocessor
from src.data_processing.data_loader import DataLoader
from src.model.model_factory import ModelFactory
from src.utils.visualizer import TrainingVisualizer

def main():
    try:
        # Step 1: Preprocess the data
        print("Starting data preprocessing...")
        preprocessor = DataPreprocessor()
        preprocessor.preprocess_data()
        
        # Step 2: Load the data
        print("\nLoading data...")
        data_loader = DataLoader()
        train_generator = data_loader.get_train_generator()
        validation_generator = data_loader.get_validation_generator()
        test_generator = data_loader.get_test_generator()
        
        # Verify data was loaded
        if (train_generator.samples == 0 or 
            validation_generator.samples == 0 or 
            test_generator.samples == 0):
            raise ValueError("No training data found. Check your data directories.")
        
        # Step 3: Create and train the model
        print("\nCreating model...")
        model_factory = ModelFactory()
        model = model_factory.create_model('cnn')
        model.build()
        model.compile()
        
        print("\nTraining model...")
        history = model.train(train_generator, validation_generator, epochs=10)
        
        # Step 4: Save the model
        print("\nSaving model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")
        
        # Step 5: Visualize results
        print("\nVisualizing training results...")
        TrainingVisualizer.plot_training_history(history)
        
        print("\nTraining completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())