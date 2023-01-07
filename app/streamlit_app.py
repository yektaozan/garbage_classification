import streamlit as st  
import tensorflow as tf
import numpy as np
import PIL
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def show_explore_page():
    st.title('Explore Page')
    st.write('This page is for exploring the models and metrics.')
    model_name = st.selectbox('Select a model', ['vgg16', 'mobilenetv2'])
    get_result = st.button("Get Results")
    if get_result:
        img_path_dict = {'History': '_history.png',
                         'Confusion Matrix': '_confusion_matrix.png',
                         'Classification Report': '_classification_report.png'}

        for img_name, img_path in img_path_dict.items():
            st.write(img_name)
            img = PIL.Image.open('garbage_classification/results/{model_name}{img_path}'.format(model_name=model_name, img_path=img_path)).resize((800, 600))
            st.image(img)
            st.write('#'*50)

# modeli yukleme
def load_model(model_name):
    model = tf.keras.models.load_model(model_name)
    return model

# label degistirme
def change_label(x):
  label_dict = {3: 'cardboard', 2: 'paper', 1: 'glass', 5: 'metal', 4: 'plastic', 6: 'trash'}
  for key, value in label_dict.items():
    if x == key:
      return value

# tahmin etme
def predict_waste(model, img_array):
    prediction = model.predict(img_array)
    return np.argmax(prediction)

# fotograflari tahmin etme
def prepare(img_path, model):
    img_size = 256
    img_array = load_img(img_path, target_size=(img_size, img_size))
    img_array = img_to_array(img_array)
    img_array = img_array/255
    img_array = np.expand_dims(img_array, axis=0)
    prediction = change_label(predict_waste(model, img_array))
    return prediction

def show_predict_page():
    st.title('Prediction Page')
    model_name = st.selectbox('Select a model', ['vgg16', 'mobilenetv2'])
    model = load_model("garbage_classification/models/{}.h5".format(model_name))
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        img = PIL.Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.')
        st.write('')
        st.write('Classifying...')
        prediction = prepare(uploaded_file, model)
        st.write(prediction)
        st.write('Done!')


page = st.sidebar.selectbox("Explore Or Predict", ("Explore", "Predict"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()

