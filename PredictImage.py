import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as k
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import argparse
import pickle
import os

img_width, img_height = 100, 100

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True, help="path of trained model")
ap.add_argument("-l", "--labelbin", required = True, help="path of label binarizer")
ap.add_argument("-i", "--image", required = True, help="path of input image")
args = vars(ap.parse_args())

#pre-processing the image for classification
img = Image.open(args["image"])
output = img.copy()

img = img.resize((img_width,img_height),Image.LANCZOS)

x = image.img_to_array(img)
x = x/255.0
x = np.expand_dims(x, axis=0)

# transfer image into tf format
x[:, :, :, 0] -= 103.939
x[:, :, :, 1] -= 116.779
x[:, :, :, 2] -= 123.68

# 'RGB'->'BGR'
x = x[:, :, :, ::-1]

# load the trained model and the corresponding lable binarizer
print("loading neural network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"],"rb").read())

# predict the input image
y_pred = model.predict(x, 1)[0]

#build the predicted label text and draw the label on the image
print('Input image size:', x.shape)
print('Classification Results:')
i = 0
for label in lb.classes_:
    print("\t%s ==> %f" % (label,y_pred[i]))
    i += 1

idx = np.argmax(y_pred)
txt = "{}: {:f}%".format(lb.classes_[idx],y_pred[idx]*100)
draw = ImageDraw.Draw(output)
ttfront = ImageFont.truetype('simhei.ttf',50)
draw.text((50,100),txt,font=ttfront)
output.show()


#释放内存
k.clear_session()
tf.reset_default_graph()



