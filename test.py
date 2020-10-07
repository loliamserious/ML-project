from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

img_width, img_height = 150, 150
test_data_dir = 'data/test'
nb_test_samples = 100

# 加载模型
model = load_model('my_model.h5')

# 准备test数据
test_gen = ImageDataGenerator(rescale=1. / 255)
generator = test_gen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        shuffle=False)

# 计算测试准确率
scores = model.evaluate_generator(
        generator, 100)
print(scores)