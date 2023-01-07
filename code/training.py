import pandas as pd 
import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt
import warnings
import re
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
import PIL
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from utils import *
from sklearn_evaluation import plot

# seed tanimlama
seed = 1842
tf.random.set_seed(seed)
np.random.seed(seed)
warnings.simplefilter('ignore')


def main():
    funcs = [get_wastenetV1, get_wastenetV2, get_vgg16, get_mobilenetv2, get_resnet50, get_inceptionv3]
    models_name = ['wastenetV1', 'wastenetV2', 'vgg16', 'mobilenetv2', 'resnet50', 'inceptionv3']
    for i in range(len(funcs)):
        print('#'*50)
        print(f'Model => {models_name[i]} is training...')
        # veri seti tanimlama
        if models_name[i] == 'mobilenetv2':
            train, valid, test = get_datasets(size=(224, 224))
            test_x, test_y = full_test(test, models_name[i])
        else:
            train, valid, test = get_datasets()
            test_x, test_y = full_test(test, models_name[i])
        # modeli olusturma
        model = funcs[i]()
        # modeli derleme
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'acc'])
        # callback fonksiyonu
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=13, restore_best_weights=True)
        # modeli egitme
        history = model.fit_generator(generator=train, validation_data=valid, epochs=100, callbacks=callback, workers=4)
        # modelin gecmisini kaydetme
        plot_history(history, models_name[i])
        # ornek tahminler
        preds = model_prediction_for_graph(model, test)
        prediction_visualization(preds, test, models_name[i])
        # test verisinin tamamini array haline getirme ve tahmin etme
        test_pred = predict_waste(model, test_x)
        # confusion matrix grafigi olusturma
        cm = confusion_matrix(test_y, test_pred)
        plot_confusion_matrix(cm, classes=train.class_indices.keys(), normalize=False, model_name=models_name[i])
        # classification report tablosu olusturma
        plot_classification_report(test_y, test_pred, list(train.class_indices.keys()), model_name=models_name[i])
        # modeli kaydetme
        save_model(model, f'{models_name[i]}.h5')
        print('#'*50)

if __name__ == '__main__':
    main()
