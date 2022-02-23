# MobileNetV3를 사용한 분리수거 분류 어플
 
 
### 어플리케이션 프론트 엔드 부분은 Flutter 팀에서 제작함
![2022-02-23 17 28 42](https://user-images.githubusercontent.com/87767242/155284095-6b1abd8e-9dbe-45e4-aa08-52918e1b1da6.png)


### 내부의 ML 파트는 모바일 어플리케이션인 것을 감안하여 MobileNetV3 를 사용하여 분류를 실행



# 설계과정

### 1. 데이터 준비하기
https://www.kaggle.com/jinfree/recycle-classification-dataset

Kaggle에서 한국의 쓰레기 사진을 저장해놓은 데이터셋을 사용함


### 2. 모델링 설계
#### 기본적으로 MobileNetV3라는 프리트레인 모델을 사용하여 우리가 원하는 모델로 전이 학습시킴
![2022-02-23 17 33 35](https://user-images.githubusercontent.com/87767242/155284691-76c483bf-7111-4baa-97cb-5812c8cc9b2c.png)

### 3. 데이터셋 준비하기
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
