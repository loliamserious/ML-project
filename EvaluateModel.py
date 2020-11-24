from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

batch_size = 32
class_dict={"Entire":0, "Flower":1, "Fruit":2, "Leaf":3, "Stem":4}

def evaluate_model(test_data,test_label,model_path):
        # load the model
        model = load_model(model_path)
        test_data = test_data/255.0

        #predict test data
        y_pred = model.predict(test_data,batch_size=batch_size,verbose=1)
        y_pred_bool = np.argmax(y_pred,axis=1).astype(np.int32)
        #change the predict label from string into integer
        test_label_bool = []
        for i in range(len(test_label)):
                idx = class_dict[test_label[i]]
                test_label_bool.append(idx)
        test_label_bool = np.array(test_label_bool)

        print('\n',classification_report(test_label_bool,y_pred_bool))
        print("test accuracy:",accuracy_score(test_label_bool,y_pred_bool))