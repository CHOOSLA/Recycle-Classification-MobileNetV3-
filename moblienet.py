import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

FAST_RUN = False
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

class_names = ['can', 'glass', 'paper', 'plastic']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

print(class_names_label)

input_dir = './input/'
categories = []
paths = []

for (root, dirs, files) in os.walk(input_dir):
    for dir in dirs:
        for file in os.listdir(root + dir + '/'):
            paths.append(root + dir + '/' + file)
            categories.append(dir)

df = pd.DataFrame({
    'filename': paths,
    'category': categories
})

print(paths)
print(len(categories))
print(df.head())

sample = random.choice(paths)
image = load_img(sample)
plt.imshow(image)

from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best-model.h5', monitor='val_accuracy', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch'
)
earlystop = EarlyStopping(patience=5)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

callbacks = [earlystop,checkpoint]

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)  # 데이터 프레임 내에서 인덱스만들기

print(train_df)
print(validate_df)

_, train_counts = np.unique(train_df['category'], return_counts=True)
_, test_counts = np.unique(validate_df['category'], return_counts=True)
pd.DataFrame({'train': train_counts,
              'test': test_counts},
             index=class_names
             ).plot.bar()
plt.show()

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)

print(example_df)

plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i + 1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

import tensorflow as tf

base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), alpha=1.0, minimalistic=False, include_top=False,
    weights=None, input_tensor=None, classes=1000, pooling=max,
    dropout_rate=0.2, classifier_activation='softmax',
    include_preprocessing=False
)

image_batch, label_batch = next(iter(train_generator))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


dense_layer=tf.keras.layers.Dense(512,activation='relu')(feature_batch_average)
dropout_layer = tf.keras.layers.Dropout(0.5)(dense_layer)
predict_layer = tf.keras.layers.Dense(4,activation='softmax')(dropout_layer)

prediction_layer = tf.keras.layers.Dense(4, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = predict_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print(train_generator.classes)
epochs=3 if FAST_RUN else 50
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=total_validate // batch_size,
                    steps_per_epoch=total_train // batch_size,
                    callbacks=callbacks)

model.save('moblienet-best.h5')