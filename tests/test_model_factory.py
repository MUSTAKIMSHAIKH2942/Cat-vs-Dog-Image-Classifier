import pytest
from src.model.model_factory import ModelFactory
from src.model.base_model import BaseModel
from src.model.cnn_model import CNNModel
from src.config import IMAGE_SIZE

def test_model_factory_creation():
    """Test ModelFactory creates correct model types"""
    # Test CNN model creation
    cnn_model = ModelFactory.create_model('cnn')
    assert isinstance(cnn_model, CNNModel)
    assert isinstance(cnn_model, BaseModel)
    
    # Verify input shape
    assert cnn_model.input_shape == (*IMAGE_SIZE, 3)
    assert cnn_model.num_classes == 1

def test_model_factory_invalid_type():
    """Test ModelFactory with invalid model type"""
    with pytest.raises(ValueError) as excinfo:
        ModelFactory.create_model('invalid_type')
    assert "Unknown model type" in str(excinfo.value)

def test_cnn_model_build():
    """Test CNN model building"""
    model = CNNModel(input_shape=(*IMAGE_SIZE, 3), num_classes=1)
    built_model = model.build()
    
    # Verify model structure
    assert built_model.layers[0].__class__.__name__ == 'Conv2D'
    assert built_model.layers[-1].__class__.__name__ == 'Dense'
    assert built_model.layers[-1].activation.__name__ == 'sigmoid'

def test_model_compilation():
    """Test model compilation"""
    model = ModelFactory.create_model('cnn')
    model.build()
    model.compile()
    
    # Verify compilation
    assert hasattr(model.model, 'optimizer')
    assert hasattr(model.model, 'loss')
    assert hasattr(model.model, 'metrics')

@pytest.mark.parametrize("optimizer,loss", [
    ('adam', 'binary_crossentropy'),
    ('sgd', 'mse')
])
def test_model_compilation_params(optimizer, loss):
    """Test model compilation with different parameters"""
    model = ModelFactory.create_model('cnn')
    model.build()
    model.compile(optimizer=optimizer, loss=loss)
    
    assert model.model.optimizer.__class__.__name__.lower() == optimizer
    assert model.model.loss == loss