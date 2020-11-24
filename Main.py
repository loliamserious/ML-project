from ImageProcess import *
from TrainFcLayerWeight import *
from TrainCNNModel import *
from EvaluateModel import *
import os
from keras import backend as k
import tensorflow as tf
import time




train_data_dir = os.getcwd() + '\\data\\trainsample'
test_data_dir = os.getcwd() + '\\data\\testsample'

'''
train_subdir = os.listdir(train_data_dir)

for k in train_subdir:
    allfile = os.listdir(train_data_dir + '\\' + k)
    print(len(allfile))
    #fit_size(train_data_dir + '\\' + k, allfile)
    resize_by_ratio(train_data_dir + '\\' + k, allfile)
'''
'''
test_subdir = os.listdir(test_data_dir)

for k in test_subdir:
    allfile = os.listdir(test_data_dir + '\\' + k)
    print(len(allfile))
    #fit_size(test_data_dir + '\\' + k, allfile)
    resize_by_ratio(test_data_dir + '\\' + k, allfile)
'''

#dataset_to_array(train_data_dir,train_subdir,"train")

X = np.load("train_data.npy")
print(X.shape)
Y = np.load("train_label.npy")
print(Y.shape)

#dataset_to_array(test_data_dir,test_subdir,"test")

X_test = np.load("test_data.npy")
print(X_test.shape)
Y_test = np.load("test_label.npy")
print(Y_test.shape)


class_weight_dict = create_class_weight(Y)
print("classweight:",class_weight_dict)


#start = time.process_time()
#train_fc_weight(X,Y,class_weight)
cnn_train_model(X,Y,class_weight_dict)
#evaluate_model(X_test,Y_test,'cnn_model.h5')
#print("training time: %s sec" % time.process_time()-start)

#释放内存
k.clear_session()
tf.reset_default_graph()




