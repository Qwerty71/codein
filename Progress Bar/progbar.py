from tensorflow import keras
from tqdm import tqdm_notebook as tqdm
from IPython.display import display
import matplotlib.pyplot as plt

class ProgressCallback(keras.callbacks.Callback):
  def __init__(self, m, epochs, batchSize, valSplit, leavePlots=True, plot=False):
    self.plot = plot
    self.leavePlots = leavePlots
    self.epochs = epochs
    self.trainSize = int(m * (1 - valSplit))
    self.valSize = int(m * valSplit)
    self.batchSize = batchSize
    self.completed = 0
    self.trainErrors, self.valErrors, self.trainAcc, self.valAcc = [],[],[],[]
    if self.plot:
      self.fig, (self.axLoss, self.axAcc) = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
      self.ide = display(self.axLoss.figure, display_id=True)

  def on_train_begin(self, logs=None):
    self.initialProgbar = tqdm(total=self.epochs, desc = "Epochs Completed... ", position = 1)

  def on_train_batch_end(self, batch, logs=None):
    self.progbar.update(1)

  def on_epoch_begin(self, epoch, logs=None):
    self.progbar = tqdm(total = self.trainSize/self.batchSize, position = 0, leave = self.leavePlots)
    self.progbar.set_description("Epoch {}, Training... ".format(epoch))

  def on_epoch_end(self, epoch, logs=None):
    self.initialProgbar.update(1)
    self.progbar.close()
    print("Loss:", logs['loss'], ", Accuracy:", logs['accuracy'], ", Validation Loss:", logs['val_loss'], ", Validation Accuracy", logs['val_accuracy'])
    self.trainErrors.append(logs['loss'])
    self.valErrors.append(logs['val_loss'])
    self.trainAcc.append(logs['accuracy'])
    self.valAcc.append(logs['val_accuracy'])

    if self.plot:
      self.axLoss.cla()
      self.axLoss.plot(list(range(len(self.trainErrors))), self.trainErrors, label="Train Loss", color='blue')
      self.axLoss.plot(list(range(len(self.valErrors))), self.valErrors, label="Val Loss", color='red')
      self.axLoss.legend()

      self.axAcc.cla()
      self.axAcc.plot(list(range(len(self.trainAcc))), self.trainAcc, label="Train Accuracy", color='blue')
      self.axAcc.plot(list(range(len(self.valAcc))), self.valAcc, label="Val Accuracy", color='red')
      self.axAcc.legend()

      self.axAcc.set_ylim(ymin=0, ymax=1)
      self.axAcc.set_xlim(xmin=0, xmax=self.epochs)
      self.axLoss.set_xlim(xmin=0, xmax=self.epochs)
      self.axLoss.set_ylim(ymin=0)

      self.ide.update(self.axLoss.figure)
