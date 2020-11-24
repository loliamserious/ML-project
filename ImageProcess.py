from PIL import Image
import os
from keras.preprocessing.image import img_to_array
import numpy as np
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.utils import class_weight



img_width= 100
img_height= 100
class_dict={"Entire":0, "Flower":1, "Fruit":2, "Leaf":3, "Stem":4}

# resize origin images into needed size for the model
def resize_by_ratio(root,allfile):
    for i in allfile:
        img_path = root + '\\' +i
        img = Image.open(img_path)
        img = img.resize((img_width,img_height),Image.LANCZOS)
        img.save(os.path.join(root,i))

# padding the size of images into uniformed square size
def fit_size(root,allfile):
    for i in allfile:
        img_path = root + '\\' +i
        img = Image.open(img_path)
        longer_side = max(img.size)
        horizontal_padding = (longer_side - img.size[0]) / 2
        vertical_padding = (longer_side - img.size[1]) / 2
        img = img.crop(
            (
                -horizontal_padding,
                -vertical_padding,
                img.size[0] + horizontal_padding,
                img.size[1] + vertical_padding
            )
        )
        img.save(os.path.join(root, i))

# prepare the array of dataset for the model
def dataset_to_array(root,subdir,setname):
    data = []
    labels = []
    for k in subdir:
        print("k:",k)
        allfile = os.listdir(root + '\\' + k)

        for i in allfile:
            img_path = root + '\\' + k + '\\' + i
            img = Image.open(img_path)
            x = img_to_array(img)  #shape(64,64,3)
            data.append(x)
            labels.append(k)

    X = np.array(data,dtype=np.float32)
    Y = np.array(labels)
    np.save("%s_data.npy" % setname,X)
    np.save("%s_label.npy" % setname,Y)

# create the class weight for dataset in order to solve balance data
def create_class_weight(y_train):
    class_weight_dict = dict()
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
    for i,key in enumerate(["Entire","Flower","Fruit","Leaf","Stem"]):
        idx = class_dict[key]
        class_weight_dict[idx] = class_weights[i]

    return class_weight_dict

# pca process
def image_pca(data):
    data_pca = PCA(n_components=0.8)
    data = data.reshape(-1,img_width*img_height*3)
    newdata = data_pca.fit_transform(data)
    return newdata

