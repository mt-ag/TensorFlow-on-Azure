from keras.callbacks import Callback
from azureml.core.run import Run
import numpy as np

class ComputeMetrics(Callback):
    def __init__(self, run):
      self.azureRun = run
 
    def on_epoch_end(self, epoch, logs):
        self.azureRun.log('best_val_acc', np.float(logs['val_acc']))