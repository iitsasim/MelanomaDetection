import numpy as np
import tensorflow as tf
import self as self
import tensorflow.keras.applications.xception as xcep
import tensorflow.keras.layers as layers
import melanoma_model_perf as mp

IMAGE_SIZE = [1024, 1024]
EPOCHS = 1
imSize = 256

def make_model(output_bias=None, metrics=None):
    with self.strategy.scope():
        model = tf.keras.Sequential([
            xcep.Xception(
                input_shape=(imSize, imSize, 3),
                weights='imagenet',
                include_top=False
            ),
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        return model


def lrfn(epoch):
    LR_START = 0.00001
    LR_MAX = 0.00005 * self.strategy.num_replicas_in_sync
    LR_MIN = 0.00001
    LR_RAMPUP_EPOCHS = 5
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = .8

    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr


def set_weights(self):
    self.initial_bias = np.log([self.malignant / self.benign])
    print(self.initial_bias)

    weight_for_0 = (1 / self.benign) * (self.total_img) / 2.0
    weight_for_1 = (1 / self.malignant) * (self.total_img) / 2.0

    self.class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))


def train_model(self):
    set_weights(self)
    with self.strategy.scope():
        model = make_model(output_bias=self.initial_bias, metrics=tf.keras.metrics.AUC(name='auc'))

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("melanoma_tl_model.h5",
                                                       save_best_only=True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                         restore_best_weights=True)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

    history = model.fit(
        self.train_dataset, epochs=EPOCHS,
        steps_per_epoch=self.STEPS_PER_EPOCH,
        validation_data=self.valid_dataset,
        validation_steps=self.VALID_STEPS,
        callbacks=[checkpoint_cb, early_stopping_cb, lr_schedule],
        class_weight=self.class_weight
    )
    mp.display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 211)
    mp.display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 212)

