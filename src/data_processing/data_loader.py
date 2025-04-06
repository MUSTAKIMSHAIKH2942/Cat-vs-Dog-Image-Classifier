import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import PROCESSED_DATA_DIR, IMAGE_SIZE, BATCH_SIZE

class DataLoader:
    """Class for loading and preprocessing image data"""
    def __init__(self):
        self.train_dir = os.path.join(PROCESSED_DATA_DIR, 'train')
        self.test_dir = os.path.join(PROCESSED_DATA_DIR, 'test')
        self.validation_dir = os.path.join(PROCESSED_DATA_DIR, 'validation')
        
        # Verify directories exist
        self._verify_directory(self.train_dir)
        self._verify_directory(self.test_dir)
        self._verify_directory(self.validation_dir)

    def _verify_directory(self, directory):
        """Verify that directory exists"""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}. Please run data preprocessing first.")

    def get_train_generator(self):
        """Create training data generator with augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        
        return train_datagen.flow_from_directory(
            self.train_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )

    def get_validation_generator(self):
        """Create validation data generator"""
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        return validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )

    def get_test_generator(self):
        """Create test data generator"""
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        return test_datagen.flow_from_directory(
            self.test_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )