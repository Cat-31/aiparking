from keras.preprocessing.image import img_to_array
from keras.models import load_model, Model
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_output_at_layer(pmodel, layer_name):
    m = Model(inputs=pmodel.input,
              outputs=pmodel.get_layer(layer_name).output)
    return m.predict(image)


weights_path = 'classify.h5'
print("[INFO] loading network...")
model = load_model('classify.h5')

image = cv2.imread('./img100/2/F_IMG_0729.JPG')

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.astype("float") / 255.0
img = image.copy()
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# classify the input image
p = model.predict(image)[0]


proba = p[int(np.argmax(p))]

label_dic = {0: "all free", 1: "left occupied", 2: 'right occupied', 3: 'fully occupied'}
label = "{}: {:.2f}%".format(label_dic[int(np.argmax(p))], proba * 100)
print(label)

for i in range(0, 4):
    label = "{}: {:.2f}%".format(label_dic[i], p[i] * 100)
    print(label)


for layer in model.layers:
    print(layer.name)

conv2d = model.get_layer('conv2d')
x1w = conv2d.get_weights()[0][:, :, 0, :]

# 显示卷积核心
print('--', conv2d.filters, conv2d.kernel_size)

for i in range(0, conv2d.filters):
    plt.subplot(5, 5, i + 1)
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.imshow(x1w[:, :, i], interpolation="nearest", cmap="gray")

cax = plt.axes((0.138, 0.1, 0.75, 0.03))
plt.colorbar(cax=cax, orientation='horizontal')
plt.show()


conv2d_1 = model.get_layer('conv2d_1')
x2w = conv2d_1.get_weights()[0][:, :, 0, :]

print(conv2d_1.filters, conv2d_1.kernel_size)

for i in range(0, conv2d_1.filters):
    plt.subplot(6, 10, i + 1)
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.imshow(x2w[:, :, i], interpolation="nearest", cmap="gray")

cax = plt.axes((0.138, 0.1, 0.75, 0.03))
plt.colorbar(cax=cax, orientation='horizontal')
plt.show()


co = get_output_at_layer(model, 'conv2d')
for i in range(0, conv2d.filters):
    plt.subplot(5, 4, i + 1)
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.imshow(co[0][:, :, i], cmap="gray")

plt.show()


co = get_output_at_layer(model, 'activation')
for i in range(0, conv2d.filters):
    plt.subplot(5, 4, i + 1)
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.imshow(co[0][:, :, i], cmap="gray")
plt.show()


conv2d_1 = model.get_layer('conv2d_1')
x1w = conv2d_1.get_weights()[0][:, :, 0, :]
co = get_output_at_layer(model, 'conv2d_1')

for i in range(0, conv2d_1.filters):
    plt.subplot(10, 5, i + 1)
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.imshow(co[0][:, :, i], cmap="gray")

plt.show()

# 手动计算卷积输出
# for i in range(0, conv2d_1.filters):
#     plt.subplot(5, 4, i + 1)
#     plt.imshow(signal.convolve2d(img, x1w[:, :, i], mode="same"), interpolation="nearest", cmap="gray")
#
# plt.show()


