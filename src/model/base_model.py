from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.models import Model

class BaseModel(ABC):
    """Abstract base class for all models"""
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    @abstractmethod
    def build(self):
        """Build the model architecture"""
        pass

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        """Compile the model"""
        if not self.model:
            raise ValueError("Model has not been built yet. Call build() first.")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_generator, validation_generator, epochs=10):
        """Train the model"""
        if not self.model:
            raise ValueError("Model has not been built yet. Call build() first.")
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs
        )
        return history

    def save(self, filepath):
        """Save the model"""
        if not self.model:
            raise ValueError("Model has not been built yet. Call build() first.")
        self.model.save(filepath)

    def summary(self):
        """Print model summary"""
        if not self.model:
            raise ValueError("Model has not been built yet. Call build() first.")
        return self.model.summary()