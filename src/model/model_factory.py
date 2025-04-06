from src.model.cnn_model import CNNModel
from src.config import IMAGE_SIZE

class ModelFactory:
    """Factory class for creating different model types"""
    @staticmethod
    def create_model(model_type, input_shape=(*IMAGE_SIZE, 3), num_classes=1):
        """
        Create a model instance based on the specified type
        
        Args:
            model_type (str): Type of model to create ('cnn')
            input_shape (tuple): Input shape for the model
            num_classes (int): Number of output classes
            
        Returns:
            BaseModel: An instance of the requested model type
        """
        if model_type == 'cnn':
            return CNNModel(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")