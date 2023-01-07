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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn_evaluation import plot
import itertools
import PIL

train_file = r"C:\Users\PC\Desktop\garbage\one-indexed-files-notrash_train.txt"
valid_file = r"C:\Users\PC\Desktop\garbage\one-indexed-files-notrash_val.txt"
test_file = r"C:\Users\PC\Desktop\garbage\one-indexed-files-notrash_test.txt"

PATH = r"C:\Users\PC\Desktop\garbage\Garbage classification"

file_list = [train_file, valid_file, test_file]

def add_prefix(x):
  for i in ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']:
    if i in x:
      return i + '/' + x


def change_label(x):
  label_dict = {3: 'cardboard', 2: 'paper', 1: 'glass', 5: 'metal', 4: 'plastic', 6: 'trash'}
  for key, value in label_dict.items():
    if x == key:
      return value


def get_dataframes(files=file_list):
  df_train = pd.read_csv(files[0], sep=' ', header=None, names=['path', 'label'])
  df_valid = pd.read_csv(files[1], sep=' ', header=None, names=['path', 'label'])
  df_test = pd.read_csv(files[2], sep=' ', header=None, names=['path', 'label'])
  
  df_train.path = df_train.path.apply(lambda x: add_prefix(x))
  df_valid.path = df_valid.path.apply(lambda x: add_prefix(x))
  df_test.path = df_test.path.apply(lambda x: add_prefix(x))

  df_train.label = df_train.label.apply(lambda x: change_label(x))
  df_valid.label = df_valid.label.apply(lambda x: change_label(x))
  df_test.label = df_test.label.apply(lambda x: change_label(x))

  return df_train, df_valid, df_test


def get_datasets(path=PATH, size=(256, 256)):
  df_train, df_valid, df_test = get_dataframes()

  train_image_generator = ImageDataGenerator(rescale=1/255,
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          shear_range=0.1,
                                          zoom_range=0.1,
                                          width_shift_range=0.1,
                                          height_shift_range=0.1)

  image_generator = ImageDataGenerator(rescale=1/255)

  #Train & Validation & Test Split
  train_dataset = train_image_generator.flow_from_dataframe(directory=path,
                                                          dataframe=df_train,
                                                          x_col='path',
                                                          y_col='label',
                                                          shuffle=True,
                                                          class_mode='categorical',
                                                          batch_size=32,
                                                          target_size=size)

  validation_dataset = image_generator.flow_from_dataframe(directory=path,
                                                         dataframe=df_valid,
                                                         x_col='path',
                                                         y_col='label',
                                                         shuffle=True,
                                                         class_mode='categorical',
                                                         batch_size=32,
                                                          target_size=size)

  test_dataset = image_generator.flow_from_dataframe(directory=path,
                                                   dataframe=df_test,
                                                   x_col='path',
                                                   y_col='label',
                                                   class_mode='categorical',
                                                   batch_size=32,
                                                   target_size=size)
  return train_dataset, validation_dataset, test_dataset


def first_visualization(main_path=PATH):
  path = r"C:\Users\PC\Desktop\garbage\sample"
  all_items = os.listdir(path)
  all_categories = os.listdir(main_path)
  fig = plt.figure(figsize=(10,8))
  c = 1
  for idx, i in enumerate(all_items):
    plt.subplot(2, 3, c)
    image_path = path + '/' + i
    img = PIL.Image.open(image_path)
    plt.axis('off')
    plt.title(all_categories[idx])
    plt.imshow(img)
    c += 1

  plt.tight_layout()
  plt.grid(False)
  plt.show()


def prediction_visualization(preds, test, model_name='model'):
  test_x, test_y = test.__getitem__(1)
  labels = test.class_indices
  labels = {v: k for k, v in labels.items()}
  plt.clf()
  plt.figure(figsize=(16, 16))
  c = 1
  for i in range(16):
    plt.subplot(4, 4, c)
    img = cv2.resize(test_x[i], dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
    plt.imshow(img)
    plt.title(f'Prediction: {labels[preds[i]]}, True Label: {labels[np.argmax(test_y[i])]}')
    plt.axis('off')
    c += 1
  plt.tight_layout()
  plt.grid(False)
  plt.savefig(f'{model_name}_tahminleri.png')
  



def train_model(model, train, valid, callback, epochs=50):
  history = model.fit(train,
                      epochs=epochs,
                      validation_data=valid,
                      callbacks=[callback])
  return history


def plot_history(history, model_name='model'):
  plt.clf()
  fig = plt.figure(figsize=(10,8))
  plt.subplot(2, 1, 1)
  plt.plot(history.history['acc'], label='train')
  plt.plot(history.history['val_acc'], label='validation')
  plt.title(f'{model_name}: Accuracy')
  plt.legend()

  plt.subplot(2, 1, 2)
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='validation')
  plt.title(f'{model_name}: Loss')
  plt.legend()
  plt.savefig(f'{model_name}_history.png')
  plt.show()
  
  


def model_evaluation(model, test):
  test_loss, test_acc, *is_anything_else_being_returned = model.evaluate(test)
  print('Test accuracy:', test_acc)
  print('Test loss:', test_loss)
  return [test_loss, test_acc]


def model_prediction_for_graph(model, test):
  test_x, _ = test.__getitem__(1)
  test_pred = model.predict(test_x)
  test_pred = np.argmax(test_pred, axis=1)
  return test_pred


def predict_waste(model, test_x):
  test_pred = model.predict(test_x)
  test_pred = np.argmax(test_pred, axis=1)
  return test_pred


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          model_name='model'):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')
  plt.clf()
  print(cm)
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig(f'{model_name}_confusion_matrix.png')
  
    


def save_model(model, model_name):
  # Save the entire model as a SavedModel.
  model.save(model_name)
  print('Model saved successfully')


def load_model(model_name):
  # Recreate the exact same model, including its weights and the optimizer
  new_model = tf.keras.models.load_model(model_name)
  return new_model


def get_wastenetV1():
  inputs = Input(shape=(256, 256, 3))

  conv_1 = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same')(inputs)
  act_1 = tf.keras.layers.Activation('relu')(conv_1)
  pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(act_1)

  conv_2 = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same')(pool_1)
  act_2 = tf.keras.layers.Activation('relu')(conv_2)
  pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(act_2)

  conv_3 = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), padding='same')(pool_2)
  act_3 = tf.keras.layers.Activation('relu')(conv_3)
  pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(act_3)

  conv_4 = tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), padding='same')(pool_3)
  act_4 = tf.keras.layers.Activation('relu')(conv_4)
  pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(act_4)

  flatten = tf.keras.layers.Flatten()(pool_4)

  dense_1 = tf.keras.layers.Dense(512)(flatten)
  act_5 = tf.keras.layers.Activation('relu')(dense_1)
  drop_1 = tf.keras.layers.Dropout(0.5)(act_5)

  dense_2 = tf.keras.layers.Dense(128)(drop_1)
  act_6 = tf.keras.layers.Activation('relu')(dense_2)
  drop_2 = tf.keras.layers.Dropout(0.5)(act_6)

  dense_3 = tf.keras.layers.Dense(6)(drop_2)
  act_7 = tf.keras.layers.Activation('softmax')(dense_3)

  model = tf.keras.Model(inputs=inputs, outputs=act_7)
  
  return model

def get_wastenetV2():
  inputs = Input(shape=(256, 256, 3))

  conv_1 = tf.keras.layers.Conv2D(256, (3,3), strides=(1,1))(inputs)
  act_1 = tf.keras.layers.Activation('relu')(conv_1)
  pool_1 = tf.keras.layers.MaxPool2D(pool_size=(3,3))(act_1)

  conv_2 = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1))(pool_1)
  act_2 = tf.keras.layers.Activation('relu')(conv_2)
  pool_2 = tf.keras.layers.MaxPool2D(pool_size=(3,3))(act_2)

  conv_3 = tf.keras.layers.Conv2D(512, (3,3), strides=(1,1))(pool_2)
  act_3 = tf.keras.layers.Activation('relu')(conv_3)
  pool_3 = tf.keras.layers.MaxPool2D(pool_size=(3,3))(act_3)

  flatten = tf.keras.layers.Flatten()(pool_3)

  dense_1 = tf.keras.layers.Dense(512)(flatten)
  act_4 = tf.keras.layers.Activation('relu')(dense_1)
  drop_1 = tf.keras.layers.Dropout(0.5)(act_4)

  dense_2 = tf.keras.layers.Dense(6)(drop_1)
  act_5 = tf.keras.layers.Activation('softmax')(dense_2)

  model = tf.keras.Model(inputs=inputs, outputs=act_5)

  return model


def get_vgg16():
  base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  predictions = tf.keras.layers.Dense(6, activation='softmax')(x)

  model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

  base_model.trainable = False
        
  return model


def get_mobilenetv2():
  base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)

  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  predictions = tf.keras.layers.Dense(6, activation='softmax')(x)

  model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

  base_model.trainable = False
        
  return model


def get_mobilenetv3large():
  base_model = tf.keras.applications.MobileNetV3Large(weights='imagenet', include_top=False)

  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  predictions = tf.keras.layers.Dense(6, activation='softmax')(x)

  model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

  base_model.trainable = False
        
  return model


def get_mobilenetv3small():
  base_model = tf.keras.applications.MobileNetV3Small(weights='imagenet', include_top=False)

  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  predictions = tf.keras.layers.Dense(6, activation='softmax')(x)

  model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

  base_model.trainable = False
        
  return model


def get_resnet50():
  base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  predictions = tf.keras.layers.Dense(6, activation='softmax')(x)

  model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

  base_model.trainable = False
        
  return model
  

def get_inceptionv3():
  base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)

  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  predictions = tf.keras.layers.Dense(6, activation='softmax')(x)

  model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

  base_model.trainable = False
        
  return model


def plot_classification_report(test_y, test_pred, classes, model_name='model'):
  plt.clf()
  plot.classification_report(test_y, test_pred, target_names=classes)
  plt.savefig(f'{model_name}_classification_report.png')
  
  
  


def full_test(test, model_name):

  if model_name == 'mobilenetv2':
    test_x = np.zeros((431, 224, 224, 3))
  else:
    test_x = np.zeros((431, 256, 256, 3))

  for i in range(14):
    if i != 13:
      test_x[i*32:(i+1)*32] = test[i][0]
    else:
      test_x[i*32:] = test[i][0]

  test_y = np.zeros((431, 6))

  for i in range(14):
    if i != 13:
      test_y[i*32:(i+1)*32] = test[i][1]
    else:
      test_y[i*32:] = test[i][1]

  test_y = np.argmax(test_y, axis=1)

  return (test_x, test_y)