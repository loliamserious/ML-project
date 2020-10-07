import numpy as np
from keras.preprocessing import image
from keras.models import load_model

img_width, img_height = 150, 150

img_path = '1234158.jpg'
img = image.load_img(img_path, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# 转换成tf格式
x[:, :, :, 0] -= 103.939
x[:, :, :, 1] -= 116.779
x[:, :, :, 2] -= 123.68
# 'RGB'->'BGR'
x = x[:, :, :, ::-1]

# 加载模型
model = load_model('my_model.h5')

# 开始预测
preds = model.predict(x, 1)
preds = np.argmax(preds)
print('图片大小:', x.shape)
print('预测图片类型为:', preds)
class_dic = np.load(open('classes.npy', 'rb'))
print('类型字典', class_dic)

