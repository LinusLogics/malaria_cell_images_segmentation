import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import PIL.Image
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
import time

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Dropout,Dense,Flatten,BatchNormalization,Conv2D
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras import  optimizers
from keras import backend as K


infected = os.listdir('/work/sunill/aireletaed/srinivas_rao/Test_1/Dataset/cell_images/Parasitized/')
uninfected = os.listdir('/work/sunill/aireletaed/srinivas_rao/Test_1/Dataset/cell_images/Uninfected/')


input_path_infected = "/work/sunill/aireletaed/srinivas_rao/Test_1/Dataset/cell_images/Parasitized/" #glob.glob(os.path.join(path, "Parasitized/*.png"))  
input_path_uninfected = "/work/sunill/aireletaed/srinivas_rao/Test_1/Dataset/cell_images/Uninfected/" #glob.glob(os.path.join(path, "Uninfected/*.png"))

def Convert_Image_to_Array(input_path_infected, input_path_uninfected):
    data = []
    labels = []

    for i in infected:
        try:
            image = cv2.imread(input_path_infected+i)
            image_resized = cv2.resize(image,(64,64))
            image_array = img_to_array(image_resized)
            data.append(image_array)
            labels.append(1)
        except:
            print('error while reading :',i)

    for i in uninfected:
        try:
            image = cv2.imread(input_path_uninfected+i)
            image_resized = cv2.resize(image,(64,64))
            image_array = img_to_array(image_resized)
            data.append(image_array)
            labels.append(0)
        except:
            print('error while reading :',i) 
    
    data_array = np.array(data)
    labels_array = np.array(labels)
    
    idx = np.arange(data_array.shape[0]) #get all the indices of data and labels
    np.random.shuffle(idx) #randomly shuffle
    image_data = data_array[idx] #get shuffled indices data
    labels = labels_array[idx] #get shuffled labels
    image_data = image_data/255
    return image_data, labels

def Generate_train_val_test_Data(image_data, labels, num_of_classes):
    X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size = 0.15, random_state = 42)
    X_train, X_val, y_train, y_val =  train_test_split(X_train, y_train, test_size = 0.17, random_state = 42)
    y_train = np_utils.to_categorical(y_train, num_classes = num_of_classes)
    y_val = np_utils.to_categorical(y_val, num_classes = num_of_classes)
    y_test = np_utils.to_categorical(y_test, num_classes = num_of_classes)
    return X_train, X_val, X_test, y_train, y_val, y_test




def generate_data_aug(X_train, y_train):
    datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1)  
    datagen.fit(X_train)
    data_from_generator = datagen.flow(X_train,y_train, batch_size = 86)
    return data_from_generator


from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lr=0.00001)

def Create_Model_CNN(height, width, classes, channels):
    model = Sequential()
    inputShape = (height, width, channels)
    model.add(Conv2D(16, (3,3),padding='same', activation = 'relu', input_shape = inputShape))
    model.add(Conv2D(16, (3,3),padding='same', activation = 'relu', input_shape = inputShape))
    model.add(MaxPooling2D(2,2))
    
    model.add(Conv2D(32, (3,3),padding='same', activation = 'relu', input_shape = inputShape))
    model.add(Conv2D(32, (3,3),padding='same', activation = 'relu', input_shape = inputShape))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3,3),padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3,3),padding='same', activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    
    model.add(Conv2D(128, (3,3),padding='same', activation = 'relu'))
    model.add(Conv2D(128, (3,3),padding='same', activation = 'relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(classes, activation = 'softmax'))
    
    return model


def compile_build_fit_model(model, optmizer_param, metric_param, loss_param,X_train,y_train, X_val, y_val, callbackParameter):
    model.compile(optimizer = optmizer_param , loss = loss_param, metrics = [metric_param])
    model_history = model.fit(X_train,y_train,epochs = 30, validation_data = (X_val,y_val),verbose = 1, steps_per_epoch = X_train.shape[0] // 86, callbacks=[callbackParameter])
    return model_history



image_data, labels = Convert_Image_to_Array(input_path_infected, input_path_uninfected )

X_train,X_val, X_test, y_train, y_val, y_test = Generate_train_val_test_Data(image_data, labels, num_of_classes = 2)
print('X_train shape: ', X_train.shape)
print('X_val shape: ', X_val.shape)
print('X_test shape: ', X_test.shape)
print('y_train shape: ', y_train.shape)
print('y_val shape: ', y_val.shape)
print('y_test shape: ', y_test.shape)

#data_from_generator = generate_data_aug(X_train, y_train)

# CNN model training
cnn_model = Create_Model_CNN(height=64, width=64, channels=3, classes=2)
cnn_model_history = compile_build_fit_model(cnn_model, 
                                        optmizer_param='adam', 
                                        metric_param='accuracy', 
                                        loss_param='categorical_crossentropy', 
                                        X_train = X_train,
                                        y_train = y_train,
                                        X_val = X_val,
                                        y_val = y_val, 
                                        callbackParameter=learning_rate_reduction)


print('Fitting the model on data completed.\n training accuracy : {} \n Training loss : {}'.format(cnn_model_history.history['accuracy'][-1],cnn_model_history.history['loss'][-1]))

