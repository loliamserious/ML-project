from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense


# 图片大小
img_width, img_height = 150, 150

top_model_weights_path = 'fc_model.h5'
best_weight_path = "weights_best.h5"
model_path = 'my_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 400
epochs = 50
batch_size = 5
num_classes = 4


# 搭建VGG-16网络
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
print('Model loaded.')

# 搭建全连接层
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

# 载入训练好的全连接层权重
top_model.load_weights(top_model_weights_path)

# 连接vGG-16卷积层和全连接层
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# 冻结前14层参数
for layer in model.layers[:15]:
    layer.trainable = False

# 设置一个较低的学习率训练
model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                metrics=['accuracy'])

# 准备训练数据
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
class_dictionary = train_generator.class_indices

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

# 记录最好的一次训练结果
checkpoint = ModelCheckpoint(best_weight_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callable_list = [checkpoint]

# fine-tune网络开始训练
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=callable_list)

# 保存模型
model.save(model_path)