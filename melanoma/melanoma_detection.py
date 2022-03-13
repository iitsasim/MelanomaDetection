import tensorflow as tf
import self as self
import melanoma_data_prep1 as dt
import melanoma_model as ml
import melanoma_model_perf as mp

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    self.strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    self.strategy = tf.distribute.get_strategy()
print('Number of replicas:', self.strategy.num_replicas_in_sync)

print(tf.__version__)

self.BATCH_SIZE = 16 * self.strategy.num_replicas_in_sync


def build_model_save(self):
    dt.prepare_data(self)
    ml.make_model()
    ml.train_model(self)

def load_model(name):
    self.model = load_model(name)

def predict_eval(eval=True):
    print('Computing predictions...')
    load_model("melanoma_tl_model_V3.h5")
    test_images_ds = self.test_dataset.map(lambda image, idnum: image)
    probabilities = self.model.predict(test_images_ds)
    print("predictions shape:", probabilities.shape)

    if eval:
        mp.model_eval(self)




if __name__ == "__main__":
   build_model_save(self)