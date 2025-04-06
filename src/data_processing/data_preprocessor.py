import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASS_NAMES, TEST_SIZE, VALIDATION_SPLIT

class DataPreprocessor:
    def __init__(self):
        self.raw_data_paths = {
            'cat': os.path.join(RAW_DATA_DIR, 'Cat'),
            'dog': os.path.join(RAW_DATA_DIR, 'Dog')
        }
        
        # Verify raw data directories exist
        for class_name, path in self.raw_data_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Raw data directory not found: {path}")

    def validate_images(self):
        """Remove corrupted images from dataset"""
        for class_name in CLASS_NAMES:
            if not os.path.exists(self.raw_data_paths[class_name]):
                continue
                
            for img_file in os.listdir(self.raw_data_paths[class_name]):
                img_path = os.path.join(self.raw_data_paths[class_name], img_file)
                try:
                    img = Image.open(img_path)
                    img.verify()
                except (IOError, SyntaxError) as e:
                    print(f'Removing corrupted image: {img_path}')
                    os.remove(img_path)

    def create_directory_structure(self):
        """Create processed data directory structure"""
        try:
            for split in ['train', 'test', 'validation']:
                for class_name in CLASS_NAMES:
                    os.makedirs(
                        os.path.join(PROCESSED_DATA_DIR, split, class_name),
                        exist_ok=True
                    )
        except Exception as e:
            print(f"Error creating directory structure: {e}")
            raise

    def split_data(self):
        """Split data into train, test, and validation sets"""
        for class_name in CLASS_NAMES:
            if not os.path.exists(self.raw_data_paths[class_name]):
                continue
                
            # Get list of all images
            images = [f for f in os.listdir(self.raw_data_paths[class_name]) 
                     if os.path.isfile(os.path.join(self.raw_data_paths[class_name], f))]
            
            if not images:
                print(f"No images found for class: {class_name}")
                continue
                
            # Split into train+validation and test
            train_val, test = train_test_split(
                images, 
                test_size=TEST_SIZE,
                random_state=42
            )
            
            # Split train into train and validation
            train, validation = train_test_split(
                train_val,
                test_size=VALIDATION_SPLIT,
                random_state=42
            )
            
            # Copy images to respective directories
            self._copy_images(class_name, 'train', train)
            self._copy_images(class_name, 'test', test)
            self._copy_images(class_name, 'validation', validation)

    def _copy_images(self, class_name, split, images):
        """Helper method to copy images to target directory"""
        src_dir = self.raw_data_paths[class_name]
        dst_dir = os.path.join(PROCESSED_DATA_DIR, split, class_name)
        
        for img in images:
            try:
                src = os.path.join(src_dir, img)
                dst = os.path.join(dst_dir, img)
                shutil.copy(src, dst)
            except Exception as e:
                print(f"Error copying {src} to {dst}: {e}")

    def preprocess_data(self):
        """Execute full preprocessing pipeline"""
        print("Validating images...")
        self.validate_images()
        
        print("Creating directory structure...")
        self.create_directory_structure()
        
        print("Splitting data...")
        self.split_data()
        
        print(f"Data preprocessing completed! Processed data saved to: {PROCESSED_DATA_DIR}")