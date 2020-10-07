import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications


img_width, img_height = 150, 150

top_model_weights_path = 'fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 400
epochs = 100
batch_size = 20
num_classes = 4

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('features_validation.npy', 'wb'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('features_train.npy', 'rb'))
    train_labels = np.array(
        [0] * (nb_train_samples // num_classes) + [1] * (nb_train_samples // num_classes)
        + [2] * (nb_train_samples // num_classes) + [3] * (nb_train_samples // num_classes))
    train_labels = to_categorical(train_labels, num_classes)
    validation_data = np.load(open('features_validation.npy', 'rb'))
    validation_labels = np.array(
        [0] * (nb_validation_samples//num_classes) + [1] * (nb_validation_samples//num_classes)
        + [2] * (nb_validation_samples//num_classes) + [3] * (nb_validation_samples//num_classes))
    validation_labels = to_categorical(validation_labels, num_classes)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()




