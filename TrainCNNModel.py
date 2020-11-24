import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
import pickle
import matplotlib.pyplot as plt


img_width, img_height = 100, 100
top_model_weights_path = 'fc_model.h5'
model_path = 'cnn_model.h5'
epochs = 100 #100
batch_size = 32 #32
num_classes = 5
split_num = 5 #5


def cnn_train_model(train_data,train_label,class_weights):
        # construct vgg-16 model without fully connected layer
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        print('Model loaded.',len(base_model.layers))

        # construct fully connected layer
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(num_classes, activation='softmax'))
        print('top-model added')


        #top_model.load_weights(top_model_weights_path)
        #print('top model weight load')

        # concatenate the the non-top vgg-16 model and fully connected layer
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

        # frozen the fist 14 layers of vgg-16 model
        for layer in model.layers[:15]:
                layer.trainable = False

        # set loss function and optimizer
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])


        #checkpoint = ModelCheckpoint(best_weight_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        #callbacks_list = [checkpoint]

        #set k-fold cross validation
        kfold = StratifiedKFold(n_splits=split_num, shuffle=True, random_state=0)
        val_acc_scores = []     #to record the history of val_acc
        train_loss_scores = []  #to record the hostory of train_loss
        train_acc_scores = []   #to record the history of train_acc
        lb = LabelBinarizer()

        for train,validation in kfold.split(train_data,train_label):

                #transform string labels into binarizer
                train_label_bin = lb.fit_transform(train_label)


                # train dataset generator
                train_datagen = ImageDataGenerator(
                        rescale=1. / 255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode="nearest",
                )

                train_generator = train_datagen.flow(
                        train_data[train],
                        train_label_bin[train],
                        batch_size=batch_size,
                )

                # validation dataset generator
                validation_datagen = ImageDataGenerator(rescale=1. / 255)

                validation_generator = validation_datagen.flow(
                        train_data[validation],
                        train_label_bin[validation],
                        batch_size=batch_size,
                )

                # train the model
                h = model.fit_generator(
                        train_generator,
                        samples_per_epoch=len(train_data[train]) // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=len(train_data[validation] // batch_size),
                        class_weight=class_weights,
                )

                #predict the model with validation dataset
                vc_scores = model.evaluate_generator(validation_generator,validation_generator.__sizeof__())

                print("val_%s: %.2f%%" % (model.metrics_names[1],vc_scores[1]*100))

                #add history to corresponding lists
                train_loss_scores.append(h.history["loss"])
                train_acc_scores.append(h.history["acc"])
                val_acc_scores.append(h.history["val_acc"])

        print("val_acc = %.2f%% (+/- %.2f%%)" % (np.mean(val_acc_scores), np.std(val_acc_scores)))
        train_loss_scores = np.array(train_loss_scores).ravel()
        train_acc_scores = np.array(train_acc_scores).ravel()
        val_acc_scores = np.array(val_acc_scores).ravel()

        # save the model to disk
        print("serializing neural network...")
        model.save(model_path)

        # save the label binarizer to disk
        print("serializing label binarizer...")
        f = open("lb.pickle", "wb")
        f.write(pickle.dumps(lb))
        f.close()


        #plot the training loss and validation accuracy
        plt.style.use("ggplot")
        x = np.arange(0, split_num * epochs)
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(x, train_loss_scores, label="train loss")
        plt.plot(x, train_acc_scores, label="train accuracy")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch * K-fold N_split")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc = "upper left")
        plt.subplot(1,2,2)
        plt.plot(x, val_acc_scores)
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch * K-fold N_split")
        plt.ylabel("Validation Accuracy")
        plt.savefig("acc_loss_plot")

