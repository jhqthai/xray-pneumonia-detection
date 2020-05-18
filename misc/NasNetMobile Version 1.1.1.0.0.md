```
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# Extract dataset
import os
import tarfile

import keras_preprocessing
from keras_preprocessing.image import ImageDataGenerator # Data preprocessing and augmentation

import sklearn
import numpy as np
```


<p style="color: red;">
The default version of TensorFlow in Colab will soon switch to TensorFlow 2.x.<br>
We recommend you <a href="https://www.tensorflow.org/guide/migrate" target="_blank">upgrade</a> now 
or ensure your notebook will continue to use TensorFlow 1.x via the <code>%tensorflow_version 1.x</code> magic:
<a href="https://colab.research.google.com/notebooks/tensorflow_version.ipynb" target="_blank">more info</a>.</p>




```
print(tf.__version__)
```

    1.15.0
    


```
# Make folder for chest xray data
!mkdir /content/data/

# Make directory to save weights
!mkdir /content/data/model

# Make directory to logs for Tensorboard
!mkdir /content/data/graph

# Download dataset
!wget --no-check-certificate \
    https://s3.eu-central-1.amazonaws.com/public.unit8.co/data/chest_xray.tar.gz \
    -O /content/data/chest_xray.tar.gz
  
tar = tarfile.open("data/chest_xray.tar.gz")
tar.extractall(path='./data/')
os.remove('data/chest_xray.tar.gz')
```

    --2019-11-10 03:37:48--  https://s3.eu-central-1.amazonaws.com/public.unit8.co/data/chest_xray.tar.gz
    Resolving s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)... 52.219.72.151
    Connecting to s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)|52.219.72.151|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1225393795 (1.1G) [application/x-gzip]
    Saving to: ‘/content/data/chest_xray.tar.gz’
    
    /content/data/chest 100%[===================>]   1.14G  21.9MB/s    in 56s     
    
    2019-11-10 03:38:45 (21.0 MB/s) - ‘/content/data/chest_xray.tar.gz’ saved [1225393795/1225393795]
    
    

Change log:
> training_datagen --> ImageDataGenerator

> trainable layer --> All except base

> xx layers NASNetMobile model - base, dense 2

> Optimizer = RMSprop(learning_rate = 0.0001)

> loss = categorical_crosscentropy

> callback = [checkpoints]

> epochs = 100

> no class weight balancing



```
TRAINING_DIR = "/content/data/chest_xray/train"
VALIDATION_DIR = "/content/data/chest_xray/val"
TEST_DIR = "/content/data/chest_xray/test"

training_datagen = ImageDataGenerator(
    # preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rescale = 1./255,
#     rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    # vertical_flip=True
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    rescale = 1./255
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

# Create training data batch
# TODO: Try grayscaling the image to see what will happen
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(224,224), 
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(224,224),
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224,224),
    class_mode='categorical',
    shuffle=False
)

train_shape = train_generator.image_shape

tf.keras.backend.clear_session() # Destroys the current TF graph and creates a new one.

base_model = tf.keras.applications.nasnet.NASNetMobile(weights='imagenet', include_top=False, input_shape=train_shape)

x = base_model.output
x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.33)(x)
# x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(2, 'softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=x)

# model.summary()

# for layer in model.layers[0:20]:
#     layer.trainable = False

for layer in base_model.layers:
  layer.trainable = False

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001) # Lower learning rate by x10

model.compile(loss='categorical_crossentropy',     
              optimizer=optimizer, 
              metrics=['accuracy'])

# Callbacks stuff
# Function to save the weights of the model after each epoch
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    '/content/data/model/weights.epoch_{epoch:02d}.hdf5',
    monitor='val_accuracy',
    save_best_only=False,
    save_weights_only=False,
    mode='auto',
    verbose=1
)

# Function to stop training early if there's no improvement
early_stopping_monitor = tf.keras.callbacks.EarlyStopping(patience = 3, monitor = "val_loss", mode="auto", verbose = 1)

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1, mode='max')

classweight = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(train_generator.labels), train_generator.labels)
print(classweight)

batch_size = 32
epochs = 100

# Training process
history = model.fit_generator(
    generator=train_generator, 
    # steps_per_epoch=train_generator.samples//batch_size, 
    epochs=epochs,
    # callbacks=[early_stopping_monitor],
    callbacks=[checkpoint],
    # shuffle=True, 
    validation_data=validation_generator, 
    # validation_steps= validation_generator//batch_size, #no because it's gonna be 0... if leave alone its len(generator) which is equal to 1. 
    # class_weight=classweight,
    verbose = 1
)

### Plot training
import matplotlib.pyplot as plt
def plot_learning_curves(history):
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.subplot(1,2,2)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.tight_layout()
    
plot_learning_curves(history)

## Load best weight
idx = np.argmin(history.history['val_loss']) 
model.load_weights("/content/data/model/weights.epoch_{:02d}.hdf5".format(idx + 1))

print("Loading the best model")
print("epoch: {}, val_loss: {}, val_acc: {}".format(idx + 1, history.history['val_loss'][idx], history.history['val_acc'][idx]))

## Evaluate the model
test_loss, test_acc = model.evaluate_generator(generator=test_generator, verbose=1)

## Analytics
from sklearn.metrics import accuracy_score, confusion_matrix

test_generator.reset()
test_preds = model.predict_generator(test_generator, verbose=1)
test_preds = np.argmax(test_preds,axis=1)

acc = accuracy_score(test_generator.classes, test_preds)*100
cm = confusion_matrix(test_generator.classes, test_preds)
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)
# plot_confusion_matrix(cm, target_names=['NORMAL', 'PNEUMONIA'], normalize=False)


print('\nTEST METRICS ----------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

print('\nTRAIN METRIC ----------------------')
print('Train acc: {}%'.format(np.round((history.history['acc'][-1])*100, 14)))
```

    Found 5216 images belonging to 2 classes.
    Found 16 images belonging to 2 classes.
    Found 624 images belonging to 2 classes.
    [1.9448173  0.67303226]
    Epoch 1/100
    162/163 [============================>.] - ETA: 0s - loss: 0.3146 - acc: 0.8767Epoch 1/100
      1/163 [..............................] - ETA: 22:20 - loss: 1.0502 - acc: 0.5625
    Epoch 00001: saving model to /content/data/model/weights.epoch_01.hdf5
    163/163 [==============================] - 305s 2s/step - loss: 0.3139 - acc: 0.8769 - val_loss: 1.0502 - val_acc: 0.5625
    Epoch 2/100
    162/163 [============================>.] - ETA: 0s - loss: 0.2164 - acc: 0.9198Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 2.3544 - acc: 0.5000
    Epoch 00002: saving model to /content/data/model/weights.epoch_02.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.2159 - acc: 0.9199 - val_loss: 2.3544 - val_acc: 0.5000
    Epoch 3/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1909 - acc: 0.9290Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 4.4237 - acc: 0.5000
    Epoch 00003: saving model to /content/data/model/weights.epoch_03.hdf5
    163/163 [==============================] - 96s 592ms/step - loss: 0.1910 - acc: 0.9291 - val_loss: 4.4237 - val_acc: 0.5000
    Epoch 4/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1788 - acc: 0.9358Epoch 1/100
      1/163 [..............................] - ETA: 46s - loss: 1.4570 - acc: 0.6250
    Epoch 00004: saving model to /content/data/model/weights.epoch_04.hdf5
    163/163 [==============================] - 97s 595ms/step - loss: 0.1780 - acc: 0.9360 - val_loss: 1.4570 - val_acc: 0.6250
    Epoch 5/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1630 - acc: 0.9412Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 3.8332 - acc: 0.5000
    Epoch 00005: saving model to /content/data/model/weights.epoch_05.hdf5
    163/163 [==============================] - 98s 599ms/step - loss: 0.1628 - acc: 0.9411 - val_loss: 3.8332 - val_acc: 0.5000
    Epoch 6/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1509 - acc: 0.9452Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 3.3706 - acc: 0.5000
    Epoch 00006: saving model to /content/data/model/weights.epoch_06.hdf5
    163/163 [==============================] - 98s 599ms/step - loss: 0.1508 - acc: 0.9452 - val_loss: 3.3706 - val_acc: 0.5000
    Epoch 7/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1569 - acc: 0.9460Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 3.0819 - acc: 0.5000
    Epoch 00007: saving model to /content/data/model/weights.epoch_07.hdf5
    163/163 [==============================] - 98s 600ms/step - loss: 0.1577 - acc: 0.9457 - val_loss: 3.0819 - val_acc: 0.5000
    Epoch 8/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1486 - acc: 0.9487Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 4.3488 - acc: 0.5000
    Epoch 00008: saving model to /content/data/model/weights.epoch_08.hdf5
    163/163 [==============================] - 98s 604ms/step - loss: 0.1477 - acc: 0.9490 - val_loss: 4.3488 - val_acc: 0.5000
    Epoch 9/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1443 - acc: 0.9504Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 4.6339 - acc: 0.5000
    Epoch 00009: saving model to /content/data/model/weights.epoch_09.hdf5
    163/163 [==============================] - 98s 603ms/step - loss: 0.1456 - acc: 0.9502 - val_loss: 4.6339 - val_acc: 0.5000
    Epoch 10/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1515 - acc: 0.9483Epoch 1/100
      1/163 [..............................] - ETA: 49s - loss: 3.0252 - acc: 0.5000
    Epoch 00010: saving model to /content/data/model/weights.epoch_10.hdf5
    163/163 [==============================] - 99s 605ms/step - loss: 0.1509 - acc: 0.9486 - val_loss: 3.0252 - val_acc: 0.5000
    Epoch 11/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1376 - acc: 0.9537Epoch 1/100
      1/163 [..............................] - ETA: 47s - loss: 4.7336 - acc: 0.5000
    Epoch 00011: saving model to /content/data/model/weights.epoch_11.hdf5
    163/163 [==============================] - 99s 609ms/step - loss: 0.1393 - acc: 0.9534 - val_loss: 4.7336 - val_acc: 0.5000
    Epoch 12/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1319 - acc: 0.9520Epoch 1/100
      1/163 [..............................] - ETA: 48s - loss: 4.2742 - acc: 0.5000
    Epoch 00012: saving model to /content/data/model/weights.epoch_12.hdf5
    163/163 [==============================] - 98s 604ms/step - loss: 0.1325 - acc: 0.9519 - val_loss: 4.2742 - val_acc: 0.5000
    Epoch 13/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1524 - acc: 0.9493Epoch 1/100
      1/163 [..............................] - ETA: 47s - loss: 3.1455 - acc: 0.5000
    Epoch 00013: saving model to /content/data/model/weights.epoch_13.hdf5
    163/163 [==============================] - 99s 609ms/step - loss: 0.1528 - acc: 0.9494 - val_loss: 3.1455 - val_acc: 0.5000
    Epoch 14/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1245 - acc: 0.9554Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 3.1773 - acc: 0.5000
    Epoch 00014: saving model to /content/data/model/weights.epoch_14.hdf5
    163/163 [==============================] - 99s 609ms/step - loss: 0.1248 - acc: 0.9553 - val_loss: 3.1773 - val_acc: 0.5000
    Epoch 15/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1366 - acc: 0.9512Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 2.9495 - acc: 0.5000
    Epoch 00015: saving model to /content/data/model/weights.epoch_15.hdf5
    163/163 [==============================] - 99s 606ms/step - loss: 0.1358 - acc: 0.9515 - val_loss: 2.9495 - val_acc: 0.5000
    Epoch 16/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1260 - acc: 0.9549Epoch 1/100
      1/163 [..............................] - ETA: 47s - loss: 3.1615 - acc: 0.5000
    Epoch 00016: saving model to /content/data/model/weights.epoch_16.hdf5
    163/163 [==============================] - 100s 611ms/step - loss: 0.1268 - acc: 0.9548 - val_loss: 3.1615 - val_acc: 0.5000
    Epoch 17/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1318 - acc: 0.9560Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 4.2920 - acc: 0.5000
    Epoch 00017: saving model to /content/data/model/weights.epoch_17.hdf5
    163/163 [==============================] - 99s 609ms/step - loss: 0.1312 - acc: 0.9563 - val_loss: 4.2920 - val_acc: 0.5000
    Epoch 18/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1280 - acc: 0.9570Epoch 1/100
      1/163 [..............................] - ETA: 46s - loss: 2.1244 - acc: 0.5625
    Epoch 00018: saving model to /content/data/model/weights.epoch_18.hdf5
    163/163 [==============================] - 99s 606ms/step - loss: 0.1293 - acc: 0.9567 - val_loss: 2.1244 - val_acc: 0.5625
    Epoch 19/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1296 - acc: 0.9587Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 5.6963 - acc: 0.5000
    Epoch 00019: saving model to /content/data/model/weights.epoch_19.hdf5
    163/163 [==============================] - 100s 612ms/step - loss: 0.1301 - acc: 0.9580 - val_loss: 5.6963 - val_acc: 0.5000
    Epoch 20/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1212 - acc: 0.9632Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 3.8702 - acc: 0.5000
    Epoch 00020: saving model to /content/data/model/weights.epoch_20.hdf5
    163/163 [==============================] - 100s 611ms/step - loss: 0.1205 - acc: 0.9634 - val_loss: 3.8702 - val_acc: 0.5000
    Epoch 21/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1264 - acc: 0.9587Epoch 1/100
      1/163 [..............................] - ETA: 46s - loss: 5.0109 - acc: 0.5000
    Epoch 00021: saving model to /content/data/model/weights.epoch_21.hdf5
    163/163 [==============================] - 99s 610ms/step - loss: 0.1264 - acc: 0.9586 - val_loss: 5.0109 - val_acc: 0.5000
    Epoch 22/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1049 - acc: 0.9659Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 3.9262 - acc: 0.5000
    Epoch 00022: saving model to /content/data/model/weights.epoch_22.hdf5
    163/163 [==============================] - 100s 613ms/step - loss: 0.1048 - acc: 0.9657 - val_loss: 3.9262 - val_acc: 0.5000
    Epoch 23/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1148 - acc: 0.9616Epoch 1/100
      1/163 [..............................] - ETA: 50s - loss: 3.9483 - acc: 0.5000
    Epoch 00023: saving model to /content/data/model/weights.epoch_23.hdf5
    163/163 [==============================] - 100s 611ms/step - loss: 0.1144 - acc: 0.9617 - val_loss: 3.9483 - val_acc: 0.5000
    Epoch 24/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1204 - acc: 0.9579Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 5.2347 - acc: 0.5000
    Epoch 00024: saving model to /content/data/model/weights.epoch_24.hdf5
    163/163 [==============================] - 99s 610ms/step - loss: 0.1197 - acc: 0.9582 - val_loss: 5.2347 - val_acc: 0.5000
    Epoch 25/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1196 - acc: 0.9626Epoch 1/100
      1/163 [..............................] - ETA: 46s - loss: 3.0126 - acc: 0.5000
    Epoch 00025: saving model to /content/data/model/weights.epoch_25.hdf5
    163/163 [==============================] - 100s 613ms/step - loss: 0.1204 - acc: 0.9622 - val_loss: 3.0126 - val_acc: 0.5000
    Epoch 26/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1260 - acc: 0.9576Epoch 1/100
      1/163 [..............................] - ETA: 48s - loss: 3.4674 - acc: 0.5000
    Epoch 00026: saving model to /content/data/model/weights.epoch_26.hdf5
    163/163 [==============================] - 100s 613ms/step - loss: 0.1256 - acc: 0.9576 - val_loss: 3.4674 - val_acc: 0.5000
    Epoch 27/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1122 - acc: 0.9641Epoch 1/100
      1/163 [..............................] - ETA: 49s - loss: 6.6356 - acc: 0.5000
    Epoch 00027: saving model to /content/data/model/weights.epoch_27.hdf5
    163/163 [==============================] - 100s 613ms/step - loss: 0.1116 - acc: 0.9643 - val_loss: 6.6356 - val_acc: 0.5000
    Epoch 28/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1108 - acc: 0.9655Epoch 1/100
      1/163 [..............................] - ETA: 46s - loss: 4.9004 - acc: 0.5000
    Epoch 00028: saving model to /content/data/model/weights.epoch_28.hdf5
    163/163 [==============================] - 99s 610ms/step - loss: 0.1103 - acc: 0.9657 - val_loss: 4.9004 - val_acc: 0.5000
    Epoch 29/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1154 - acc: 0.9603Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 4.8081 - acc: 0.5000
    Epoch 00029: saving model to /content/data/model/weights.epoch_29.hdf5
    163/163 [==============================] - 100s 611ms/step - loss: 0.1148 - acc: 0.9605 - val_loss: 4.8081 - val_acc: 0.5000
    Epoch 30/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1158 - acc: 0.9643Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 5.5584 - acc: 0.5000
    Epoch 00030: saving model to /content/data/model/weights.epoch_30.hdf5
    163/163 [==============================] - 98s 603ms/step - loss: 0.1152 - acc: 0.9645 - val_loss: 5.5584 - val_acc: 0.5000
    Epoch 31/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1228 - acc: 0.9608Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 3.7674 - acc: 0.5000
    Epoch 00031: saving model to /content/data/model/weights.epoch_31.hdf5
    163/163 [==============================] - 96s 586ms/step - loss: 0.1231 - acc: 0.9603 - val_loss: 3.7674 - val_acc: 0.5000
    Epoch 32/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1206 - acc: 0.9603Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 5.7070 - acc: 0.5000
    Epoch 00032: saving model to /content/data/model/weights.epoch_32.hdf5
    163/163 [==============================] - 93s 572ms/step - loss: 0.1201 - acc: 0.9603 - val_loss: 5.7070 - val_acc: 0.5000
    Epoch 33/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1221 - acc: 0.9637Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 4.6656 - acc: 0.5000
    Epoch 00033: saving model to /content/data/model/weights.epoch_33.hdf5
    163/163 [==============================] - 94s 574ms/step - loss: 0.1218 - acc: 0.9636 - val_loss: 4.6656 - val_acc: 0.5000
    Epoch 34/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1094 - acc: 0.9651Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 4.5878 - acc: 0.5000
    Epoch 00034: saving model to /content/data/model/weights.epoch_34.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.1108 - acc: 0.9647 - val_loss: 4.5878 - val_acc: 0.5000
    Epoch 35/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1171 - acc: 0.9612Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 4.3260 - acc: 0.5000
    Epoch 00035: saving model to /content/data/model/weights.epoch_35.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.1166 - acc: 0.9615 - val_loss: 4.3260 - val_acc: 0.5000
    Epoch 36/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1058 - acc: 0.9651Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 4.4338 - acc: 0.5000
    Epoch 00036: saving model to /content/data/model/weights.epoch_36.hdf5
    163/163 [==============================] - 93s 570ms/step - loss: 0.1051 - acc: 0.9653 - val_loss: 4.4338 - val_acc: 0.5000
    Epoch 37/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1196 - acc: 0.9618Epoch 1/100
      1/163 [..............................] - ETA: 1:29 - loss: 5.4527 - acc: 0.5000
    Epoch 00037: saving model to /content/data/model/weights.epoch_37.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.1190 - acc: 0.9618 - val_loss: 5.4527 - val_acc: 0.5000
    Epoch 38/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1191 - acc: 0.9626Epoch 1/100
      1/163 [..............................] - ETA: 41s - loss: 3.9768 - acc: 0.5000
    Epoch 00038: saving model to /content/data/model/weights.epoch_38.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1190 - acc: 0.9626 - val_loss: 3.9768 - val_acc: 0.5000
    Epoch 39/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1071 - acc: 0.9628Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 4.7560 - acc: 0.5000
    Epoch 00039: saving model to /content/data/model/weights.epoch_39.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1068 - acc: 0.9628 - val_loss: 4.7560 - val_acc: 0.5000
    Epoch 40/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1069 - acc: 0.9653Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 4.4206 - acc: 0.5000
    Epoch 00040: saving model to /content/data/model/weights.epoch_40.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.1066 - acc: 0.9653 - val_loss: 4.4206 - val_acc: 0.5000
    Epoch 41/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1115 - acc: 0.9630Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 5.2622 - acc: 0.5000
    Epoch 00041: saving model to /content/data/model/weights.epoch_41.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1117 - acc: 0.9628 - val_loss: 5.2622 - val_acc: 0.5000
    Epoch 42/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1075 - acc: 0.9639Epoch 1/100
      1/163 [..............................] - ETA: 42s - loss: 5.3675 - acc: 0.5000
    Epoch 00042: saving model to /content/data/model/weights.epoch_42.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1068 - acc: 0.9641 - val_loss: 5.3675 - val_acc: 0.5000
    Epoch 43/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1101 - acc: 0.9645Epoch 1/100
      1/163 [..............................] - ETA: 40s - loss: 5.8816 - acc: 0.5000
    Epoch 00043: saving model to /content/data/model/weights.epoch_43.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.1098 - acc: 0.9645 - val_loss: 5.8816 - val_acc: 0.5000
    Epoch 44/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1145 - acc: 0.9622Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 5.3769 - acc: 0.5000
    Epoch 00044: saving model to /content/data/model/weights.epoch_44.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.1151 - acc: 0.9622 - val_loss: 5.3769 - val_acc: 0.5000
    Epoch 45/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1148 - acc: 0.9610Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 4.0523 - acc: 0.5000
    Epoch 00045: saving model to /content/data/model/weights.epoch_45.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.1142 - acc: 0.9613 - val_loss: 4.0523 - val_acc: 0.5000
    Epoch 46/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1236 - acc: 0.9628Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 2.9665 - acc: 0.5000
    Epoch 00046: saving model to /content/data/model/weights.epoch_46.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.1229 - acc: 0.9630 - val_loss: 2.9665 - val_acc: 0.5000
    Epoch 47/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0909 - acc: 0.9705Epoch 1/100
      1/163 [..............................] - ETA: 46s - loss: 5.1887 - acc: 0.5000
    Epoch 00047: saving model to /content/data/model/weights.epoch_47.hdf5
    163/163 [==============================] - 93s 568ms/step - loss: 0.0907 - acc: 0.9705 - val_loss: 5.1887 - val_acc: 0.5000
    Epoch 48/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1134 - acc: 0.9632Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 5.1818 - acc: 0.5000
    Epoch 00048: saving model to /content/data/model/weights.epoch_48.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.1128 - acc: 0.9634 - val_loss: 5.1818 - val_acc: 0.5000
    Epoch 49/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1170 - acc: 0.9622Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 4.7142 - acc: 0.5000
    Epoch 00049: saving model to /content/data/model/weights.epoch_49.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1165 - acc: 0.9622 - val_loss: 4.7142 - val_acc: 0.5000
    Epoch 50/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1063 - acc: 0.9672Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 4.9555 - acc: 0.5000
    Epoch 00050: saving model to /content/data/model/weights.epoch_50.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.1075 - acc: 0.9670 - val_loss: 4.9555 - val_acc: 0.5000
    Epoch 51/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1165 - acc: 0.9674Epoch 1/100
      1/163 [..............................] - ETA: 46s - loss: 5.5071 - acc: 0.5000
    Epoch 00051: saving model to /content/data/model/weights.epoch_51.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.1162 - acc: 0.9674 - val_loss: 5.5071 - val_acc: 0.5000
    Epoch 52/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1059 - acc: 0.9662Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 5.6460 - acc: 0.5000
    Epoch 00052: saving model to /content/data/model/weights.epoch_52.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1067 - acc: 0.9657 - val_loss: 5.6460 - val_acc: 0.5000
    Epoch 53/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1161 - acc: 0.9651Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 3.9479 - acc: 0.5000
    Epoch 00053: saving model to /content/data/model/weights.epoch_53.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1154 - acc: 0.9653 - val_loss: 3.9479 - val_acc: 0.5000
    Epoch 54/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1036 - acc: 0.9649Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 4.5147 - acc: 0.5000
    Epoch 00054: saving model to /content/data/model/weights.epoch_54.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.1030 - acc: 0.9651 - val_loss: 4.5147 - val_acc: 0.5000
    Epoch 55/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1141 - acc: 0.9632Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 3.8783 - acc: 0.5000
    Epoch 00055: saving model to /content/data/model/weights.epoch_55.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1140 - acc: 0.9630 - val_loss: 3.8783 - val_acc: 0.5000
    Epoch 56/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1104 - acc: 0.9645Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 5.0406 - acc: 0.5000
    Epoch 00056: saving model to /content/data/model/weights.epoch_56.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.1114 - acc: 0.9645 - val_loss: 5.0406 - val_acc: 0.5000
    Epoch 57/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0974 - acc: 0.9689Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 5.4624 - acc: 0.5000
    Epoch 00057: saving model to /content/data/model/weights.epoch_57.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.0968 - acc: 0.9691 - val_loss: 5.4624 - val_acc: 0.5000
    Epoch 58/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0989 - acc: 0.9709Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 3.2762 - acc: 0.5000
    Epoch 00058: saving model to /content/data/model/weights.epoch_58.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.0992 - acc: 0.9705 - val_loss: 3.2762 - val_acc: 0.5000
    Epoch 59/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0981 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 5.6076 - acc: 0.5000
    Epoch 00059: saving model to /content/data/model/weights.epoch_59.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.0977 - acc: 0.9682 - val_loss: 5.6076 - val_acc: 0.5000
    Epoch 60/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1098 - acc: 0.9655Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 6.1544 - acc: 0.5000
    Epoch 00060: saving model to /content/data/model/weights.epoch_60.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.1102 - acc: 0.9653 - val_loss: 6.1544 - val_acc: 0.5000
    Epoch 61/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0991 - acc: 0.9697Epoch 1/100
      1/163 [..............................] - ETA: 42s - loss: 5.8724 - acc: 0.5000
    Epoch 00061: saving model to /content/data/model/weights.epoch_61.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.0991 - acc: 0.9697 - val_loss: 5.8724 - val_acc: 0.5000
    Epoch 62/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1002 - acc: 0.9668Epoch 1/100
      1/163 [..............................] - ETA: 42s - loss: 3.5251 - acc: 0.5000
    Epoch 00062: saving model to /content/data/model/weights.epoch_62.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0996 - acc: 0.9670 - val_loss: 3.5251 - val_acc: 0.5000
    Epoch 63/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1082 - acc: 0.9641Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 3.0714 - acc: 0.5625
    Epoch 00063: saving model to /content/data/model/weights.epoch_63.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.1094 - acc: 0.9640 - val_loss: 3.0714 - val_acc: 0.5625
    Epoch 64/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0973 - acc: 0.9691Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 5.2249 - acc: 0.5000
    Epoch 00064: saving model to /content/data/model/weights.epoch_64.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.0979 - acc: 0.9689 - val_loss: 5.2249 - val_acc: 0.5000
    Epoch 65/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1079 - acc: 0.9639Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 2.9878 - acc: 0.5625
    Epoch 00065: saving model to /content/data/model/weights.epoch_65.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.1073 - acc: 0.9641 - val_loss: 2.9878 - val_acc: 0.5625
    Epoch 66/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0854 - acc: 0.9713Epoch 1/100
      1/163 [..............................] - ETA: 42s - loss: 3.7700 - acc: 0.5000
    Epoch 00066: saving model to /content/data/model/weights.epoch_66.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0855 - acc: 0.9712 - val_loss: 3.7700 - val_acc: 0.5000
    Epoch 67/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0991 - acc: 0.9697Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 4.4717 - acc: 0.5000
    Epoch 00067: saving model to /content/data/model/weights.epoch_67.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.0986 - acc: 0.9699 - val_loss: 4.4717 - val_acc: 0.5000
    Epoch 68/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1005 - acc: 0.9684Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 3.5143 - acc: 0.5000
    Epoch 00068: saving model to /content/data/model/weights.epoch_68.hdf5
    163/163 [==============================] - 92s 562ms/step - loss: 0.1022 - acc: 0.9680 - val_loss: 3.5143 - val_acc: 0.5000
    Epoch 69/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0997 - acc: 0.9693Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 4.4872 - acc: 0.5000
    Epoch 00069: saving model to /content/data/model/weights.epoch_69.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0999 - acc: 0.9691 - val_loss: 4.4872 - val_acc: 0.5000
    Epoch 70/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1004 - acc: 0.9672Epoch 1/100
      1/163 [..............................] - ETA: 40s - loss: 3.9240 - acc: 0.5625
    Epoch 00070: saving model to /content/data/model/weights.epoch_70.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.1012 - acc: 0.9672 - val_loss: 3.9240 - val_acc: 0.5625
    Epoch 71/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0929 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 46s - loss: 2.7305 - acc: 0.5625
    Epoch 00071: saving model to /content/data/model/weights.epoch_71.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.0950 - acc: 0.9697 - val_loss: 2.7305 - val_acc: 0.5625
    Epoch 72/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1206 - acc: 0.9637Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 5.1932 - acc: 0.5000
    Epoch 00072: saving model to /content/data/model/weights.epoch_72.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.1200 - acc: 0.9640 - val_loss: 5.1932 - val_acc: 0.5000
    Epoch 73/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0907 - acc: 0.9691Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 5.8781 - acc: 0.5000
    Epoch 00073: saving model to /content/data/model/weights.epoch_73.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0902 - acc: 0.9693 - val_loss: 5.8781 - val_acc: 0.5000
    Epoch 74/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1048 - acc: 0.9659Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 4.2677 - acc: 0.5000
    Epoch 00074: saving model to /content/data/model/weights.epoch_74.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.1044 - acc: 0.9659 - val_loss: 4.2677 - val_acc: 0.5000
    Epoch 75/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1018 - acc: 0.9678Epoch 1/100
      1/163 [..............................] - ETA: 41s - loss: 4.9657 - acc: 0.5000
    Epoch 00075: saving model to /content/data/model/weights.epoch_75.hdf5
    163/163 [==============================] - 91s 561ms/step - loss: 0.1013 - acc: 0.9680 - val_loss: 4.9657 - val_acc: 0.5000
    Epoch 76/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0955 - acc: 0.9728Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 4.4976 - acc: 0.5000
    Epoch 00076: saving model to /content/data/model/weights.epoch_76.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0949 - acc: 0.9730 - val_loss: 4.4976 - val_acc: 0.5000
    Epoch 77/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1013 - acc: 0.9668Epoch 1/100
      1/163 [..............................] - ETA: 42s - loss: 4.6798 - acc: 0.5000
    Epoch 00077: saving model to /content/data/model/weights.epoch_77.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.1007 - acc: 0.9670 - val_loss: 4.6798 - val_acc: 0.5000
    Epoch 78/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0976 - acc: 0.9678Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 4.9274 - acc: 0.5000
    Epoch 00078: saving model to /content/data/model/weights.epoch_78.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.0980 - acc: 0.9674 - val_loss: 4.9274 - val_acc: 0.5000
    Epoch 79/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1084 - acc: 0.9699Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 3.4269 - acc: 0.5625
    Epoch 00079: saving model to /content/data/model/weights.epoch_79.hdf5
    163/163 [==============================] - 91s 561ms/step - loss: 0.1079 - acc: 0.9701 - val_loss: 3.4269 - val_acc: 0.5625
    Epoch 80/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0975 - acc: 0.9703Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 4.4476 - acc: 0.5000
    Epoch 00080: saving model to /content/data/model/weights.epoch_80.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.0974 - acc: 0.9703 - val_loss: 4.4476 - val_acc: 0.5000
    Epoch 81/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1075 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 41s - loss: 2.9928 - acc: 0.5625
    Epoch 00081: saving model to /content/data/model/weights.epoch_81.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1082 - acc: 0.9686 - val_loss: 2.9928 - val_acc: 0.5625
    Epoch 82/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1036 - acc: 0.9711Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 4.9711 - acc: 0.5000
    Epoch 00082: saving model to /content/data/model/weights.epoch_82.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1030 - acc: 0.9712 - val_loss: 4.9711 - val_acc: 0.5000
    Epoch 83/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0968 - acc: 0.9703Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 7.2739 - acc: 0.5000
    Epoch 00083: saving model to /content/data/model/weights.epoch_83.hdf5
    163/163 [==============================] - 92s 562ms/step - loss: 0.0962 - acc: 0.9705 - val_loss: 7.2739 - val_acc: 0.5000
    Epoch 84/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0937 - acc: 0.9697Epoch 1/100
      1/163 [..............................] - ETA: 42s - loss: 7.3478 - acc: 0.5000
    Epoch 00084: saving model to /content/data/model/weights.epoch_84.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.0931 - acc: 0.9699 - val_loss: 7.3478 - val_acc: 0.5000
    Epoch 85/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1186 - acc: 0.9695Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 4.8109 - acc: 0.5000
    Epoch 00085: saving model to /content/data/model/weights.epoch_85.hdf5
    163/163 [==============================] - 92s 562ms/step - loss: 0.1181 - acc: 0.9695 - val_loss: 4.8109 - val_acc: 0.5000
    Epoch 86/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0973 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 3.0385 - acc: 0.5625
    Epoch 00086: saving model to /content/data/model/weights.epoch_86.hdf5
    163/163 [==============================] - 92s 562ms/step - loss: 0.0969 - acc: 0.9682 - val_loss: 3.0385 - val_acc: 0.5625
    Epoch 87/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0909 - acc: 0.9691Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 6.4692 - acc: 0.5000
    Epoch 00087: saving model to /content/data/model/weights.epoch_87.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.0911 - acc: 0.9688 - val_loss: 6.4692 - val_acc: 0.5000
    Epoch 88/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1000 - acc: 0.9695Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 4.4860 - acc: 0.5000
    Epoch 00088: saving model to /content/data/model/weights.epoch_88.hdf5
    163/163 [==============================] - 91s 560ms/step - loss: 0.0995 - acc: 0.9697 - val_loss: 4.4860 - val_acc: 0.5000
    Epoch 89/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0925 - acc: 0.9689Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 2.5723 - acc: 0.5625
    Epoch 00089: saving model to /content/data/model/weights.epoch_89.hdf5
    163/163 [==============================] - 92s 562ms/step - loss: 0.0922 - acc: 0.9689 - val_loss: 2.5723 - val_acc: 0.5625
    Epoch 90/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0942 - acc: 0.9715Epoch 1/100
      1/163 [..............................] - ETA: 45s - loss: 4.8586 - acc: 0.5000
    Epoch 00090: saving model to /content/data/model/weights.epoch_90.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0944 - acc: 0.9714 - val_loss: 4.8586 - val_acc: 0.5000
    Epoch 91/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1020 - acc: 0.9678Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 5.1361 - acc: 0.5625
    Epoch 00091: saving model to /content/data/model/weights.epoch_91.hdf5
    163/163 [==============================] - 93s 572ms/step - loss: 0.1020 - acc: 0.9678 - val_loss: 5.1361 - val_acc: 0.5625
    Epoch 92/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0958 - acc: 0.9695Epoch 1/100
      1/163 [..............................] - ETA: 43s - loss: 4.4073 - acc: 0.5625
    Epoch 00092: saving model to /content/data/model/weights.epoch_92.hdf5
    163/163 [==============================] - 94s 577ms/step - loss: 0.0958 - acc: 0.9695 - val_loss: 4.4073 - val_acc: 0.5625
    Epoch 93/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1007 - acc: 0.9697Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 5.3362 - acc: 0.5000
    Epoch 00093: saving model to /content/data/model/weights.epoch_93.hdf5
    163/163 [==============================] - 95s 583ms/step - loss: 0.1008 - acc: 0.9697 - val_loss: 5.3362 - val_acc: 0.5000
    Epoch 94/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0951 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 48s - loss: 4.8754 - acc: 0.5000
    Epoch 00094: saving model to /content/data/model/weights.epoch_94.hdf5
    163/163 [==============================] - 96s 591ms/step - loss: 0.0946 - acc: 0.9689 - val_loss: 4.8754 - val_acc: 0.5000
    Epoch 95/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1028 - acc: 0.9693Epoch 1/100
      1/163 [..............................] - ETA: 37s - loss: 7.1666 - acc: 0.5000
    Epoch 00095: saving model to /content/data/model/weights.epoch_95.hdf5
    163/163 [==============================] - 97s 594ms/step - loss: 0.1038 - acc: 0.9688 - val_loss: 7.1666 - val_acc: 0.5000
    Epoch 96/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0866 - acc: 0.9732Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 4.7808 - acc: 0.5000
    Epoch 00096: saving model to /content/data/model/weights.epoch_96.hdf5
    163/163 [==============================] - 97s 598ms/step - loss: 0.0861 - acc: 0.9734 - val_loss: 4.7808 - val_acc: 0.5000
    Epoch 97/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0805 - acc: 0.9749Epoch 1/100
      1/163 [..............................] - ETA: 46s - loss: 5.8630 - acc: 0.5000
    Epoch 00097: saving model to /content/data/model/weights.epoch_97.hdf5
    163/163 [==============================] - 99s 606ms/step - loss: 0.0802 - acc: 0.9749 - val_loss: 5.8630 - val_acc: 0.5000
    Epoch 98/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0929 - acc: 0.9742Epoch 1/100
      1/163 [..............................] - ETA: 44s - loss: 6.1549 - acc: 0.5000
    Epoch 00098: saving model to /content/data/model/weights.epoch_98.hdf5
    163/163 [==============================] - 98s 599ms/step - loss: 0.0951 - acc: 0.9741 - val_loss: 6.1549 - val_acc: 0.5000
    Epoch 99/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1015 - acc: 0.9695Epoch 1/100
      1/163 [..............................] - ETA: 46s - loss: 4.8395 - acc: 0.5000
    Epoch 00099: saving model to /content/data/model/weights.epoch_99.hdf5
    163/163 [==============================] - 100s 616ms/step - loss: 0.1012 - acc: 0.9695 - val_loss: 4.8395 - val_acc: 0.5000
    Epoch 100/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1210 - acc: 0.9659Epoch 1/100
      1/163 [..............................] - ETA: 50s - loss: 6.7680 - acc: 0.5000
    Epoch 00100: saving model to /content/data/model/weights.epoch_100.hdf5
    163/163 [==============================] - 101s 617ms/step - loss: 0.1208 - acc: 0.9659 - val_loss: 6.7680 - val_acc: 0.5000
    Loading the best model
    epoch: 1, val_loss: 1.0501724481582642, val_acc: 0.5625
    20/20 [==============================] - 6s 305ms/step - loss: 1.0415 - acc: 0.6442
    20/20 [==============================] - 10s 488ms/step
    CONFUSION MATRIX ------------------
    [[ 13 221]
     [  1 389]]
    
    TEST METRICS ----------------------
    Accuracy: 64.42307692307693%
    Precision: 63.77049180327868%
    Recall: 99.74358974358975%
    F1-score: 77.8
    
    TRAIN METRIC ----------------------
    Train acc: 96.58742547035219%
    


![png](NasNetMobile%20Version%201.1.1.0.0_files/NasNetMobile%20Version%201.1.1.0.0_4_1.png)

