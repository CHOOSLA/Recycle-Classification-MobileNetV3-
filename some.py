import tensorflow as tf
import numpy as np
from keras import backend as K

from freezenSession import freeze_session

K.set_learning_phase(0)


model = tf.keras.models.load_model('moblieNet_Large_train_99_val_96.h5')
model.summary()

print(model.outputs)
print(model.inputs)

input_dir = './input/'
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
class_names = ['can', 'glass', 'paper', 'plastic']

pred_dir = './predict/'
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    pred_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

predict = model.predict(test_dataset)

val_labels = np.argmax(model.predict(test_dataset))
print(class_names[val_labels])

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('moblieNet_Large_train_99_val_96.tflite', 'wb') as f:
  f.write(tflite_model)