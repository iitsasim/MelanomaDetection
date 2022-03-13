import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def model_eval(self):
    results = self.model.evaluate(self.test_dataset, batch_size=20)
    print("test loss, test acc:", results)
    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = self.model.predict(self.test_dataset[:3])
    print("predictions shape:", predictions.shape)

def display_training_curves(training, validation, title, subplot):
    if subplot % 10 == 1:  # set up the subplots on the first call
        plt.subplots(figsize=(20, 20), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
    plt.savefig(title+'_ml.png')
