# 필요한 라이브러리 불러오기
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 데이터 준비
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        '/content/[라벨]사과_0.정상',  # 정상 이미지 폴더 경로
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
        '/content/[라벨]사과_1.질병', # 질병 이미지 폴더 경로
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

# 모델 구축
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

# 모델 훈련
history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=100,
      validation_data=validation_generator,
      validation_steps=8,  
      verbose=2)

# 모델 평가
test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
print('\nTest accuracy:', test_acc)

# 실제 이미지 판별
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # 이미지를 배치 형태로 만들어 줌
    predictions = model.predict(img_array)
    score = predictions[0]
    if score < 0.5:
        print("정상")
    else:
        print("질병")

# 실제 이미지 경로
test_image_path = "/content/apple.jpg"
predict_image(test_image_path)
