import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

#데이터 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#데이터 전처리
x_train = x_train.reshape(-1, 32 * 32 * 3).astype('float32') / 255.0
x_test = x_test.reshape(-1, 32 * 32 * 3).astype('float32') / 255.0
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

#은닉층
mlp = Sequential()
mlp.add(Dense(512, activation='relu', input_shape=(3072,)))
mlp.add(Dense(256, activation='relu'))
mlp.add(Dense(128, activation='relu'))
mlp.add(Dense(10, activation='softmax'))

#컴파일
mlp.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=0.01),
            metrics=['accuracy'])

#학습 결과를 history 변수에 저장 
history = mlp.fit(x_train,y_train,batch_size=128,epochs=50,validation_data=(x_test,y_test),verbose=2)

#평가
res=mlp.evaluate(x_test,y_test,verbose=0)
print('정확률=', res[1]*100)

# 손실값 그래프 그리기
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve (손실값 변화)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()