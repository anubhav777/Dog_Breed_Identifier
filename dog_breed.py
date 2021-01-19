import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras.models import Model

vgg16=VGG16(input_shape=[224,224,3],weights='imagenet',include_top=False)

for layer in vgg16.layers:
    layer.trainable=False

x= Flatten()(vgg16.output)

predictions=Dense(120,activation='softmax')(x)

model = Model(inputs=vgg16.input,outputs=predictions)

df=pd.read_csv('labels.csv')
n_train=8176
X_train=df.loc[:n_train,:]
X_test=df.loc[:n_train,:]
X_test

train_data=ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

def apply_ext(name):
    return name+".jpg"
df['id'] = df['id'].apply(apply_ext)

train_set=train_data.flow_from_dataframe(dataframe=X_train,directory='train/',x_col="id",y_col="breed",target_size=(224,224),batch_size=32,class_mode='categorical')

test_data=ImageDataGenerator(rescale=1/255)
test_set=test_data.flow_from_dataframe(dataframe=X_test,directory='train/',x_col="id",y_col="breed",target_size=(224,224),batch_size=32,class_mode='categorical')

imagepath='train/'
allimages=[]
for filename in os.listdir(imagepath):
    single_image = image.load_img(f'train/{filename}',target_size=(64,64))
    single_image = image.img_to_array(single_image)
    print(single_image)
    
# cnn = tf.keras.models.Sequential()
# cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
# cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
# cnn.add(tf.keras.layers.Flatten())
# cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
# cnn.add(tf.keras.layers.Dense(units=len(df['breed'].unique()),activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_set,validation_data=test_set,epochs=35,steps_per_epoch=len(train_set),)
model.save("model.h5")
imagepath='test/'
allimages=[]
alldata=[]
for filename in os.listdir(imagepath):
    single_image = image.load_img(f'test/{filename}',target_size=(64,64))
    single_image = image.img_to_array(single_image)
    single_image = np.expand_dims(single_image,axis=0)
    result=model.predict(single_image)
    oldata=[]
    for i in range(len(result[0])):
        newarr={'dogname':list(train_set.class_indices)[i],'value':result[0][i],'imagename':filename}
        oldata.append(newarr)
    alldata.append(oldata)
#     print(result[0])
list(train_set.class_indices)[0]
print(alldata)
   
