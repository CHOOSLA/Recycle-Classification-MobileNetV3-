import tensorflow as tf

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
input_dir = './input/'


train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    input_dir,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    input_dir,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

class_names = train_dataset.class_names
print(class_names)
