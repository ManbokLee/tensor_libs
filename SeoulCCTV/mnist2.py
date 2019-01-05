import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('훈련 이미지: ', train_images.shape)
print('훈련 레이블: ', train_labels.shape)
print('테스트 이미지: ', test_images.shape)
print('테스트 레이블: ', test_labels.shape)
print('\n')


mnist_idx = 100
print('[label]')
print('number label = ' , train_labels[mnist_idx])
print('\n')


print('[image]')
for row in train_images[mnist_idx]:
    for col in row:
        print('%10f' % col, end="")
    print('\n')
print('\n')

plt.figure(figsize = (5,5))
image = train_images[mnist_idx]
plt.imshow(image)
plt.show()

'''
28 * 28 배열에 저장된 손글씨 이미지를 확인
라벨의 경우에는 one-hot vector 가 아닌 실제 해당하는 숫자가 저장됨
'''






