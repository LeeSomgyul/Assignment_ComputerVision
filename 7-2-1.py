import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리 (CNN은 이미지 형태 유지해야 함!)
x_train = x_train.astype('float32') / 255.0  # shape: (50000, 32, 32, 3)
x_test = x_test.astype('float32') / 255.0

# 라벨 원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# CNN 모델 구성
cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(10, activation='softmax'))

# 컴파일
cnn.compile(loss='categorical_crossentropy',
            optimizer=SGD(learning_rate=0.01),
            metrics=['accuracy'])

# 학습
history = cnn.fit(x_train, y_train,
                  batch_size=128,
                  epochs=50,
                  validation_data=(x_test, y_test),
                  verbose=2)

# 평가
res = cnn.evaluate(x_test, y_test, verbose=0)
print("정확률 =", res[1] * 100)

# 손실값 그래프 그리기
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve (손실값 변화)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
