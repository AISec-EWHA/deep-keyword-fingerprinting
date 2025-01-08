import os
import logging
from tensorflow.keras.callbacks import Callback

def setup_logger(log_file):
    logger = logging.getLogger("SharedLogger")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class EpochLogger(Callback):
    def __init__(self, logger):
        super(EpochLogger, self).__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss')
        train_acc = logs.get('accuracy')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')
        
        self.logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
        )