import pytest
import os
import shutil
from PIL import Image
import numpy as np
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASS_NAMES
from src.data_processing.data_preprocessor import DataPreprocessor

@pytest.fixture(scope="module")
def setup_test_environment():
    """Fixture to create test directories and sample images"""
    # Create test raw data directories
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create 4 test images per class
        for i in range(4):
            img_path = os.path.join(class_dir, f"test_{i}.jpg")
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype='uint8'))
            img.save(img_path)
    
    yield  # This is where the testing happens
    
    # Cleanup
    shutil.rmtree(RAW_DATA_DIR, ignore_errors=True)
    shutil.rmtree(PROCESSED_DATA_DIR, ignore_errors=True)

def test_data_preprocessor_init(setup_test_environment):
    """Test DataPreprocessor initialization"""
    preprocessor = DataPreprocessor()
    assert hasattr(preprocessor, 'raw_data_paths')
    assert all(os.path.exists(path) for path in preprocessor.raw_data_paths.values())

def test_validate_images(setup_test_environment):
    """Test image validation"""
    preprocessor = DataPreprocessor()
    
    # Add a corrupted image
    corrupt_path = os.path.join(RAW_DATA_DIR, CLASS_NAMES[0], "corrupt.jpg")
    with open(corrupt_path, 'w') as f:
        f.write("not an image")
    
    preprocessor.validate_images()
    assert not os.path.exists(corrupt_path)

def test_create_directory_structure(setup_test_environment):
    """Test directory structure creation"""
    preprocessor = DataPreprocessor()
    preprocessor.create_directory_structure()
    
    for split in ['train', 'test', 'validation']:
        for class_name in CLASS_NAMES:
            dir_path = os.path.join(PROCESSED_DATA_DIR, split, class_name)
            assert os.path.exists(dir_path)

def test_split_data(setup_test_environment):
    """Test data splitting functionality"""
    preprocessor = DataPreprocessor()
    preprocessor.create_directory_structure()
    preprocessor.split_data()
    
    # Check files were distributed
    for class_name in CLASS_NAMES:
        train_dir = os.path.join(PROCESSED_DATA_DIR, 'train', class_name)
        test_dir = os.path.join(PROCESSED_DATA_DIR, 'test', class_name)
        val_dir = os.path.join(PROCESSED_DATA_DIR, 'validation', class_name)
        
        assert len(os.listdir(train_dir)) > 0
        assert len(os.listdir(test_dir)) > 0
        assert len(os.listdir(val_dir)) > 0

def test_full_preprocess_pipeline(setup_test_environment):
    """Test complete preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_data()
    
    # Verify final output
    for split in ['train', 'test', 'validation']:
        for class_name in CLASS_NAMES:
            dir_path = os.path.join(PROCESSED_DATA_DIR, split, class_name)
            assert os.path.exists(dir_path)
            assert len(os.listdir(dir_path)) > 0