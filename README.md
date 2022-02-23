# MobileNetV3를 사용한 분리수거 분류 어플
 
 
### 어플리케이션 프론트 엔드 부분은 Flutter 팀에서 제작함
![2022-02-23 17 28 42](https://user-images.githubusercontent.com/87767242/155284095-6b1abd8e-9dbe-45e4-aa08-52918e1b1da6.png)


### 내부의 ML 파트는 모바일 어플리케이션인 것을 감안하여 MobileNetV3 를 사용하여 분류를 실행

#### 레포지토리 안의 mobilenet2.py가 메인임


# 설계과정

### 1. 데이터 준비하기
https://www.kaggle.com/jinfree/recycle-classification-dataset

Kaggle에서 한국의 쓰레기 사진을 저장해놓은 데이터셋을 사용함


### 2. 모델링 설계
#### 기본적으로 MobileNetV3라는 프리트레인 모델을 사용하여 우리가 원하는 모델로 전이 학습시킴
![2022-02-23 17 33 35](https://user-images.githubusercontent.com/87767242/155284691-76c483bf-7111-4baa-97cb-5812c8cc9b2c.png)

### 3. 배치 사이즈, 이미지 사이즈, 데이터 경로 설정
```python
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
input_dir = './input/'
```

### 4. 데이터셋 준비하기
```python
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
>>> ['can','glass','paper','plastic']
```

### 5. 데이터 프리패치 및 데이터 증강 레이어 추가
```python
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
])
```
### 6. 픽셀값을 재조정하는 레이어 추가
```python
preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
```
MobilenetV3가 원하는 픽셀값으로 알아서 전환하여 준다.

### 7. Pre-trained 된 모델 불러오기
```python
MG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')
```

### 8. 기존 파라미터값 고정 및 FC(Fully Connected) 레이어 연결을 위한 전역평균폴링층 추가
```pyhton
base_model.trainable = False # MobileNetV3의 파라미터 값 고정

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
```

### 9. 함수형 API 모델링
```python
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
print(outputs.shape)
model = tf.keras.Model(inputs, outputs)
```

### 10. 손실함수와 옵티마이저 설정
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
```

### 11. 콜백 함수 생성
```python
initial_epochs = 100

checkpoint = ModelCheckpoint(
    'best-model.h5', monitor='val_accuracy', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch'
)
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

callbacks = [earlystop, checkpoint,learning_rate_reduction]
```

### 12. 모델 훈련 시작
```python
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset,
                    callbacks=callbacks)
       ```
