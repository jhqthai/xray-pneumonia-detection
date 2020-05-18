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

    --2019-11-10 06:30:51--  https://s3.eu-central-1.amazonaws.com/public.unit8.co/data/chest_xray.tar.gz
    Resolving s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)... 52.219.72.147
    Connecting to s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)|52.219.72.147|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1225393795 (1.1G) [application/x-gzip]
    Saving to: ‘/content/data/chest_xray.tar.gz’
    
    /content/data/chest 100%[===================>]   1.14G  21.5MB/s    in 57s     
    
    2019-11-10 06:31:49 (20.4 MB/s) - ‘/content/data/chest_xray.tar.gz’ saved [1225393795/1225393795]
    
    

Change log:
> training_datagen --> ImageDataGenerator

> trainable layer --> All except base

> InceptionResNetV2 model - base, flat, dense

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
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
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
    target_size=(150,150),
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150,150),
    class_mode='categorical',
    shuffle=False
)

train_shape = train_generator.image_shape

tf.keras.backend.clear_session() # Destroys the current TF graph and creates a new one.

base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=train_shape)

# Define the machine learning model
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(2, 'softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=x)

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

# test_loss, test_acc = model.evaluate_generator(generator=test_generator, verbose=1)

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
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5
    219062272/219055592 [==============================] - 7s 0us/step
    [1.9448173  0.67303226]
    Epoch 1/100
    162/163 [============================>.] - ETA: 0s - loss: 0.3660 - acc: 0.8536Epoch 1/100
      1/163 [..............................] - ETA: 21:28 - loss: 2.1733 - acc: 0.5000
    Epoch 00001: saving model to /content/data/model/weights.epoch_01.hdf5
    163/163 [==============================] - 250s 2s/step - loss: 0.3662 - acc: 0.8539 - val_loss: 2.1733 - val_acc: 0.5000
    Epoch 2/100
    162/163 [============================>.] - ETA: 0s - loss: 0.2467 - acc: 0.9107Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 3.1384 - acc: 0.5000
    Epoch 00002: saving model to /content/data/model/weights.epoch_02.hdf5
    163/163 [==============================] - 90s 551ms/step - loss: 0.2466 - acc: 0.9107 - val_loss: 3.1384 - val_acc: 0.5000
    Epoch 3/100
    162/163 [============================>.] - ETA: 0s - loss: 0.2185 - acc: 0.9211Epoch 1/100
      1/163 [..............................] - ETA: 1:02 - loss: 3.1438 - acc: 0.5000
    Epoch 00003: saving model to /content/data/model/weights.epoch_03.hdf5
    163/163 [==============================] - 94s 579ms/step - loss: 0.2189 - acc: 0.9212 - val_loss: 3.1438 - val_acc: 0.5000
    Epoch 4/100
    162/163 [============================>.] - ETA: 0s - loss: 0.2046 - acc: 0.9282Epoch 1/100
      1/163 [..............................] - ETA: 1:04 - loss: 2.3760 - acc: 0.5000
    Epoch 00004: saving model to /content/data/model/weights.epoch_04.hdf5
    163/163 [==============================] - 94s 579ms/step - loss: 0.2046 - acc: 0.9283 - val_loss: 2.3760 - val_acc: 0.5000
    Epoch 5/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1882 - acc: 0.9342Epoch 1/100
      1/163 [..............................] - ETA: 1:08 - loss: 5.1627 - acc: 0.5000
    Epoch 00005: saving model to /content/data/model/weights.epoch_05.hdf5
    163/163 [==============================] - 94s 574ms/step - loss: 0.1874 - acc: 0.9346 - val_loss: 5.1627 - val_acc: 0.5000
    Epoch 6/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1709 - acc: 0.9369Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 4.6479 - acc: 0.5000
    Epoch 00006: saving model to /content/data/model/weights.epoch_06.hdf5
    163/163 [==============================] - 94s 575ms/step - loss: 0.1707 - acc: 0.9367 - val_loss: 4.6479 - val_acc: 0.5000
    Epoch 7/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1717 - acc: 0.9410Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 5.3248 - acc: 0.5000
    Epoch 00007: saving model to /content/data/model/weights.epoch_07.hdf5
    163/163 [==============================] - 94s 578ms/step - loss: 0.1710 - acc: 0.9413 - val_loss: 5.3248 - val_acc: 0.5000
    Epoch 8/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1726 - acc: 0.9375Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 3.1112 - acc: 0.5000
    Epoch 00008: saving model to /content/data/model/weights.epoch_08.hdf5
    163/163 [==============================] - 94s 576ms/step - loss: 0.1720 - acc: 0.9379 - val_loss: 3.1112 - val_acc: 0.5000
    Epoch 9/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1678 - acc: 0.9417Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 4.5461 - acc: 0.5000
    Epoch 00009: saving model to /content/data/model/weights.epoch_09.hdf5
    163/163 [==============================] - 95s 581ms/step - loss: 0.1672 - acc: 0.9419 - val_loss: 4.5461 - val_acc: 0.5000
    Epoch 10/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1537 - acc: 0.9439Epoch 1/100
      1/163 [..............................] - ETA: 1:08 - loss: 2.6268 - acc: 0.5000
    Epoch 00010: saving model to /content/data/model/weights.epoch_10.hdf5
    163/163 [==============================] - 95s 581ms/step - loss: 0.1535 - acc: 0.9440 - val_loss: 2.6268 - val_acc: 0.5000
    Epoch 11/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1524 - acc: 0.9475Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 6.3946 - acc: 0.5000
    Epoch 00011: saving model to /content/data/model/weights.epoch_11.hdf5
    163/163 [==============================] - 94s 579ms/step - loss: 0.1529 - acc: 0.9473 - val_loss: 6.3946 - val_acc: 0.5000
    Epoch 12/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1477 - acc: 0.9483Epoch 1/100
      1/163 [..............................] - ETA: 1:07 - loss: 5.3360 - acc: 0.5000
    Epoch 00012: saving model to /content/data/model/weights.epoch_12.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1473 - acc: 0.9482 - val_loss: 5.3360 - val_acc: 0.5000
    Epoch 13/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1452 - acc: 0.9483Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 4.5724 - acc: 0.5000
    Epoch 00013: saving model to /content/data/model/weights.epoch_13.hdf5
    163/163 [==============================] - 95s 581ms/step - loss: 0.1455 - acc: 0.9480 - val_loss: 4.5724 - val_acc: 0.5000
    Epoch 14/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1469 - acc: 0.9524Epoch 1/100
      1/163 [..............................] - ETA: 1:07 - loss: 2.4982 - acc: 0.5000
    Epoch 00014: saving model to /content/data/model/weights.epoch_14.hdf5
    163/163 [==============================] - 94s 579ms/step - loss: 0.1476 - acc: 0.9521 - val_loss: 2.4982 - val_acc: 0.5000
    Epoch 15/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1509 - acc: 0.9485Epoch 1/100
      1/163 [..............................] - ETA: 1:04 - loss: 4.9143 - acc: 0.5000
    Epoch 00015: saving model to /content/data/model/weights.epoch_15.hdf5
    163/163 [==============================] - 94s 575ms/step - loss: 0.1511 - acc: 0.9482 - val_loss: 4.9143 - val_acc: 0.5000
    Epoch 16/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1472 - acc: 0.9466Epoch 1/100
      1/163 [..............................] - ETA: 1:07 - loss: 4.2595 - acc: 0.5000
    Epoch 00016: saving model to /content/data/model/weights.epoch_16.hdf5
    163/163 [==============================] - 94s 579ms/step - loss: 0.1473 - acc: 0.9465 - val_loss: 4.2595 - val_acc: 0.5000
    Epoch 17/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1347 - acc: 0.9524Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 7.5473 - acc: 0.5000
    Epoch 00017: saving model to /content/data/model/weights.epoch_17.hdf5
    163/163 [==============================] - 94s 577ms/step - loss: 0.1348 - acc: 0.9523 - val_loss: 7.5473 - val_acc: 0.5000
    Epoch 18/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1438 - acc: 0.9487Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 6.0579 - acc: 0.5000
    Epoch 00018: saving model to /content/data/model/weights.epoch_18.hdf5
    163/163 [==============================] - 94s 575ms/step - loss: 0.1440 - acc: 0.9482 - val_loss: 6.0579 - val_acc: 0.5000
    Epoch 19/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1382 - acc: 0.9512Epoch 1/100
      1/163 [..............................] - ETA: 1:07 - loss: 6.3253 - acc: 0.5000
    Epoch 00019: saving model to /content/data/model/weights.epoch_19.hdf5
    163/163 [==============================] - 95s 581ms/step - loss: 0.1379 - acc: 0.9511 - val_loss: 6.3253 - val_acc: 0.5000
    Epoch 20/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1396 - acc: 0.9504Epoch 1/100
      1/163 [..............................] - ETA: 1:03 - loss: 4.4814 - acc: 0.5000
    Epoch 00020: saving model to /content/data/model/weights.epoch_20.hdf5
    163/163 [==============================] - 94s 577ms/step - loss: 0.1392 - acc: 0.9507 - val_loss: 4.4814 - val_acc: 0.5000
    Epoch 21/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1292 - acc: 0.9533Epoch 1/100
      1/163 [..............................] - ETA: 1:09 - loss: 7.8599 - acc: 0.5000
    Epoch 00021: saving model to /content/data/model/weights.epoch_21.hdf5
    163/163 [==============================] - 94s 577ms/step - loss: 0.1293 - acc: 0.9530 - val_loss: 7.8599 - val_acc: 0.5000
    Epoch 22/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1340 - acc: 0.9533Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 4.6502 - acc: 0.5000
    Epoch 00022: saving model to /content/data/model/weights.epoch_22.hdf5
    163/163 [==============================] - 94s 576ms/step - loss: 0.1339 - acc: 0.9532 - val_loss: 4.6502 - val_acc: 0.5000
    Epoch 23/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1314 - acc: 0.9543Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 4.3661 - acc: 0.5000
    Epoch 00023: saving model to /content/data/model/weights.epoch_23.hdf5
    163/163 [==============================] - 94s 580ms/step - loss: 0.1318 - acc: 0.9542 - val_loss: 4.3661 - val_acc: 0.5000
    Epoch 24/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1358 - acc: 0.9549Epoch 1/100
      1/163 [..............................] - ETA: 1:07 - loss: 1.0469 - acc: 0.5625
    Epoch 00024: saving model to /content/data/model/weights.epoch_24.hdf5
    163/163 [==============================] - 95s 582ms/step - loss: 0.1352 - acc: 0.9551 - val_loss: 1.0469 - val_acc: 0.5625
    Epoch 25/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1368 - acc: 0.9502Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 1.6101 - acc: 0.5000
    Epoch 00025: saving model to /content/data/model/weights.epoch_25.hdf5
    163/163 [==============================] - 94s 579ms/step - loss: 0.1366 - acc: 0.9503 - val_loss: 1.6101 - val_acc: 0.5000
    Epoch 26/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1342 - acc: 0.9531Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 5.5799 - acc: 0.5000
    Epoch 00026: saving model to /content/data/model/weights.epoch_26.hdf5
    163/163 [==============================] - 95s 582ms/step - loss: 0.1355 - acc: 0.9530 - val_loss: 5.5799 - val_acc: 0.5000
    Epoch 27/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1352 - acc: 0.9506Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 6.0717 - acc: 0.5000
    Epoch 00027: saving model to /content/data/model/weights.epoch_27.hdf5
    163/163 [==============================] - 94s 579ms/step - loss: 0.1351 - acc: 0.9505 - val_loss: 6.0717 - val_acc: 0.5000
    Epoch 28/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1234 - acc: 0.9558Epoch 1/100
      1/163 [..............................] - ETA: 1:03 - loss: 6.7225 - acc: 0.5000
    Epoch 00028: saving model to /content/data/model/weights.epoch_28.hdf5
    163/163 [==============================] - 94s 574ms/step - loss: 0.1230 - acc: 0.9559 - val_loss: 6.7225 - val_acc: 0.5000
    Epoch 29/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1341 - acc: 0.9502Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 5.2360 - acc: 0.5000
    Epoch 00029: saving model to /content/data/model/weights.epoch_29.hdf5
    163/163 [==============================] - 95s 581ms/step - loss: 0.1338 - acc: 0.9503 - val_loss: 5.2360 - val_acc: 0.5000
    Epoch 30/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1323 - acc: 0.9518Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 6.8842 - acc: 0.5000
    Epoch 00030: saving model to /content/data/model/weights.epoch_30.hdf5
    163/163 [==============================] - 95s 581ms/step - loss: 0.1323 - acc: 0.9517 - val_loss: 6.8842 - val_acc: 0.5000
    Epoch 31/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1211 - acc: 0.9568Epoch 1/100
      1/163 [..............................] - ETA: 1:09 - loss: 6.8019 - acc: 0.5000
    Epoch 00031: saving model to /content/data/model/weights.epoch_31.hdf5
    163/163 [==============================] - 96s 587ms/step - loss: 0.1208 - acc: 0.9571 - val_loss: 6.8019 - val_acc: 0.5000
    Epoch 32/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1314 - acc: 0.9508Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 7.6214 - acc: 0.5000
    Epoch 00032: saving model to /content/data/model/weights.epoch_32.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1315 - acc: 0.9507 - val_loss: 7.6214 - val_acc: 0.5000
    Epoch 33/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1234 - acc: 0.9562Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 5.8043 - acc: 0.5000
    Epoch 00033: saving model to /content/data/model/weights.epoch_33.hdf5
    163/163 [==============================] - 95s 584ms/step - loss: 0.1231 - acc: 0.9565 - val_loss: 5.8043 - val_acc: 0.5000
    Epoch 34/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1240 - acc: 0.9539Epoch 1/100
      1/163 [..............................] - ETA: 1:33 - loss: 3.0037 - acc: 0.5000
    Epoch 00034: saving model to /content/data/model/weights.epoch_34.hdf5
    163/163 [==============================] - 95s 582ms/step - loss: 0.1239 - acc: 0.9540 - val_loss: 3.0037 - val_acc: 0.5000
    Epoch 35/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1231 - acc: 0.9551Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 4.5299 - acc: 0.5000
    Epoch 00035: saving model to /content/data/model/weights.epoch_35.hdf5
    163/163 [==============================] - 95s 581ms/step - loss: 0.1232 - acc: 0.9551 - val_loss: 4.5299 - val_acc: 0.5000
    Epoch 36/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1179 - acc: 0.9562Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 5.6313 - acc: 0.5000
    Epoch 00036: saving model to /content/data/model/weights.epoch_36.hdf5
    163/163 [==============================] - 95s 584ms/step - loss: 0.1187 - acc: 0.9561 - val_loss: 5.6313 - val_acc: 0.5000
    Epoch 37/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1288 - acc: 0.9537Epoch 1/100
      1/163 [..............................] - ETA: 1:07 - loss: 3.3033 - acc: 0.5000
    Epoch 00037: saving model to /content/data/model/weights.epoch_37.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1284 - acc: 0.9538 - val_loss: 3.3033 - val_acc: 0.5000
    Epoch 38/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1265 - acc: 0.9539Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 5.1963 - acc: 0.5000
    Epoch 00038: saving model to /content/data/model/weights.epoch_38.hdf5
    163/163 [==============================] - 95s 583ms/step - loss: 0.1259 - acc: 0.9542 - val_loss: 5.1963 - val_acc: 0.5000
    Epoch 39/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1260 - acc: 0.9525Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 5.1290 - acc: 0.5000
    Epoch 00039: saving model to /content/data/model/weights.epoch_39.hdf5
    163/163 [==============================] - 96s 587ms/step - loss: 0.1256 - acc: 0.9528 - val_loss: 5.1290 - val_acc: 0.5000
    Epoch 40/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1218 - acc: 0.9552Epoch 1/100
      1/163 [..............................] - ETA: 1:07 - loss: 4.6793 - acc: 0.5000
    Epoch 00040: saving model to /content/data/model/weights.epoch_40.hdf5
    163/163 [==============================] - 95s 584ms/step - loss: 0.1215 - acc: 0.9555 - val_loss: 4.6793 - val_acc: 0.5000
    Epoch 41/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1254 - acc: 0.9552Epoch 1/100
      1/163 [..............................] - ETA: 1:04 - loss: 2.8677 - acc: 0.5000
    Epoch 00041: saving model to /content/data/model/weights.epoch_41.hdf5
    163/163 [==============================] - 95s 581ms/step - loss: 0.1251 - acc: 0.9555 - val_loss: 2.8677 - val_acc: 0.5000
    Epoch 42/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1264 - acc: 0.9545Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 5.2576 - acc: 0.5000
    Epoch 00042: saving model to /content/data/model/weights.epoch_42.hdf5
    163/163 [==============================] - 96s 586ms/step - loss: 0.1264 - acc: 0.9544 - val_loss: 5.2576 - val_acc: 0.5000
    Epoch 43/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1147 - acc: 0.9587Epoch 1/100
      1/163 [..............................] - ETA: 1:04 - loss: 0.8690 - acc: 0.5625
    Epoch 00043: saving model to /content/data/model/weights.epoch_43.hdf5
    163/163 [==============================] - 95s 584ms/step - loss: 0.1154 - acc: 0.9584 - val_loss: 0.8690 - val_acc: 0.5625
    Epoch 44/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1233 - acc: 0.9562Epoch 1/100
      1/163 [..............................] - ETA: 1:03 - loss: 3.0306 - acc: 0.5000
    Epoch 00044: saving model to /content/data/model/weights.epoch_44.hdf5
    163/163 [==============================] - 95s 582ms/step - loss: 0.1240 - acc: 0.9559 - val_loss: 3.0306 - val_acc: 0.5000
    Epoch 45/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1192 - acc: 0.9578Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 3.7795 - acc: 0.5000
    Epoch 00045: saving model to /content/data/model/weights.epoch_45.hdf5
    163/163 [==============================] - 96s 589ms/step - loss: 0.1188 - acc: 0.9580 - val_loss: 3.7795 - val_acc: 0.5000
    Epoch 46/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1216 - acc: 0.9570Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 3.9058 - acc: 0.5000
    Epoch 00046: saving model to /content/data/model/weights.epoch_46.hdf5
    163/163 [==============================] - 96s 586ms/step - loss: 0.1211 - acc: 0.9571 - val_loss: 3.9058 - val_acc: 0.5000
    Epoch 47/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1229 - acc: 0.9568Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 2.9049 - acc: 0.5000
    Epoch 00047: saving model to /content/data/model/weights.epoch_47.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1225 - acc: 0.9571 - val_loss: 2.9049 - val_acc: 0.5000
    Epoch 48/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1140 - acc: 0.9593Epoch 1/100
      1/163 [..............................] - ETA: 1:10 - loss: 8.4007 - acc: 0.5000
    Epoch 00048: saving model to /content/data/model/weights.epoch_48.hdf5
    163/163 [==============================] - 94s 579ms/step - loss: 0.1138 - acc: 0.9594 - val_loss: 8.4007 - val_acc: 0.5000
    Epoch 49/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1193 - acc: 0.9551Epoch 1/100
      1/163 [..............................] - ETA: 1:02 - loss: 5.1812 - acc: 0.5000
    Epoch 00049: saving model to /content/data/model/weights.epoch_49.hdf5
    163/163 [==============================] - 95s 584ms/step - loss: 0.1192 - acc: 0.9551 - val_loss: 5.1812 - val_acc: 0.5000
    Epoch 50/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1189 - acc: 0.9551Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 3.5493 - acc: 0.5000
    Epoch 00050: saving model to /content/data/model/weights.epoch_50.hdf5
    163/163 [==============================] - 94s 578ms/step - loss: 0.1190 - acc: 0.9549 - val_loss: 3.5493 - val_acc: 0.5000
    Epoch 51/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1079 - acc: 0.9605Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 2.4951 - acc: 0.5000
    Epoch 00051: saving model to /content/data/model/weights.epoch_51.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1076 - acc: 0.9607 - val_loss: 2.4951 - val_acc: 0.5000
    Epoch 52/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1260 - acc: 0.9552Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 2.7216 - acc: 0.5000
    Epoch 00052: saving model to /content/data/model/weights.epoch_52.hdf5
    163/163 [==============================] - 96s 588ms/step - loss: 0.1256 - acc: 0.9553 - val_loss: 2.7216 - val_acc: 0.5000
    Epoch 53/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1193 - acc: 0.9551Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 2.9159 - acc: 0.5000
    Epoch 00053: saving model to /content/data/model/weights.epoch_53.hdf5
    163/163 [==============================] - 96s 587ms/step - loss: 0.1190 - acc: 0.9551 - val_loss: 2.9159 - val_acc: 0.5000
    Epoch 54/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1165 - acc: 0.9570Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 5.6651 - acc: 0.5000
    Epoch 00054: saving model to /content/data/model/weights.epoch_54.hdf5
    163/163 [==============================] - 96s 587ms/step - loss: 0.1164 - acc: 0.9571 - val_loss: 5.6651 - val_acc: 0.5000
    Epoch 55/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1152 - acc: 0.9599Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 5.4586 - acc: 0.5000
    Epoch 00055: saving model to /content/data/model/weights.epoch_55.hdf5
    163/163 [==============================] - 95s 583ms/step - loss: 0.1153 - acc: 0.9599 - val_loss: 5.4586 - val_acc: 0.5000
    Epoch 56/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1160 - acc: 0.9589Epoch 1/100
      1/163 [..............................] - ETA: 1:07 - loss: 0.8660 - acc: 0.6875
    Epoch 00056: saving model to /content/data/model/weights.epoch_56.hdf5
    163/163 [==============================] - 95s 582ms/step - loss: 0.1169 - acc: 0.9588 - val_loss: 0.8660 - val_acc: 0.6875
    Epoch 57/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1121 - acc: 0.9560Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 5.6143 - acc: 0.5000
    Epoch 00057: saving model to /content/data/model/weights.epoch_57.hdf5
    163/163 [==============================] - 95s 584ms/step - loss: 0.1118 - acc: 0.9561 - val_loss: 5.6143 - val_acc: 0.5000
    Epoch 58/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1192 - acc: 0.9593Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 4.0012 - acc: 0.5000
    Epoch 00058: saving model to /content/data/model/weights.epoch_58.hdf5
    163/163 [==============================] - 94s 578ms/step - loss: 0.1197 - acc: 0.9590 - val_loss: 4.0012 - val_acc: 0.5000
    Epoch 59/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1183 - acc: 0.9568Epoch 1/100
      1/163 [..............................] - ETA: 1:02 - loss: 5.3796 - acc: 0.5000
    Epoch 00059: saving model to /content/data/model/weights.epoch_59.hdf5
    163/163 [==============================] - 95s 583ms/step - loss: 0.1184 - acc: 0.9567 - val_loss: 5.3796 - val_acc: 0.5000
    Epoch 60/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1145 - acc: 0.9593Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 5.0764 - acc: 0.5000
    Epoch 00060: saving model to /content/data/model/weights.epoch_60.hdf5
    163/163 [==============================] - 95s 583ms/step - loss: 0.1147 - acc: 0.9594 - val_loss: 5.0764 - val_acc: 0.5000
    Epoch 61/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1089 - acc: 0.9608Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 4.3240 - acc: 0.5000
    Epoch 00061: saving model to /content/data/model/weights.epoch_61.hdf5
    163/163 [==============================] - 95s 583ms/step - loss: 0.1087 - acc: 0.9607 - val_loss: 4.3240 - val_acc: 0.5000
    Epoch 62/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1089 - acc: 0.9595Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 4.5413 - acc: 0.5000
    Epoch 00062: saving model to /content/data/model/weights.epoch_62.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1090 - acc: 0.9595 - val_loss: 4.5413 - val_acc: 0.5000
    Epoch 63/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1174 - acc: 0.9593Epoch 1/100
      1/163 [..............................] - ETA: 1:03 - loss: 6.4794 - acc: 0.5000
    Epoch 00063: saving model to /content/data/model/weights.epoch_63.hdf5
    163/163 [==============================] - 95s 583ms/step - loss: 0.1171 - acc: 0.9595 - val_loss: 6.4794 - val_acc: 0.5000
    Epoch 64/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1194 - acc: 0.9599Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 8.0238 - acc: 0.5000
    Epoch 00064: saving model to /content/data/model/weights.epoch_64.hdf5
    163/163 [==============================] - 95s 583ms/step - loss: 0.1205 - acc: 0.9595 - val_loss: 8.0238 - val_acc: 0.5000
    Epoch 65/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1114 - acc: 0.9599Epoch 1/100
      1/163 [..............................] - ETA: 1:08 - loss: 3.6245 - acc: 0.5000
    Epoch 00065: saving model to /content/data/model/weights.epoch_65.hdf5
    163/163 [==============================] - 95s 584ms/step - loss: 0.1111 - acc: 0.9601 - val_loss: 3.6245 - val_acc: 0.5000
    Epoch 66/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1222 - acc: 0.9576Epoch 1/100
      1/163 [..............................] - ETA: 1:04 - loss: 3.9132 - acc: 0.5000
    Epoch 00066: saving model to /content/data/model/weights.epoch_66.hdf5
    163/163 [==============================] - 95s 580ms/step - loss: 0.1218 - acc: 0.9576 - val_loss: 3.9132 - val_acc: 0.5000
    Epoch 67/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1147 - acc: 0.9597Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 1.7874 - acc: 0.5625
    Epoch 00067: saving model to /content/data/model/weights.epoch_67.hdf5
    163/163 [==============================] - 95s 580ms/step - loss: 0.1149 - acc: 0.9594 - val_loss: 1.7874 - val_acc: 0.5625
    Epoch 68/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1115 - acc: 0.9593Epoch 1/100
      1/163 [..............................] - ETA: 1:08 - loss: 5.6390 - acc: 0.5000
    Epoch 00068: saving model to /content/data/model/weights.epoch_68.hdf5
    163/163 [==============================] - 95s 583ms/step - loss: 0.1113 - acc: 0.9594 - val_loss: 5.6390 - val_acc: 0.5000
    Epoch 69/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1127 - acc: 0.9601Epoch 1/100
      1/163 [..............................] - ETA: 1:03 - loss: 7.6957 - acc: 0.5000
    Epoch 00069: saving model to /content/data/model/weights.epoch_69.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1124 - acc: 0.9601 - val_loss: 7.6957 - val_acc: 0.5000
    Epoch 70/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1163 - acc: 0.9566Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 3.6260 - acc: 0.5000
    Epoch 00070: saving model to /content/data/model/weights.epoch_70.hdf5
    163/163 [==============================] - 96s 587ms/step - loss: 0.1161 - acc: 0.9567 - val_loss: 3.6260 - val_acc: 0.5000
    Epoch 71/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1262 - acc: 0.9541Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 6.4404 - acc: 0.5000
    Epoch 00071: saving model to /content/data/model/weights.epoch_71.hdf5
    163/163 [==============================] - 96s 588ms/step - loss: 0.1267 - acc: 0.9538 - val_loss: 6.4404 - val_acc: 0.5000
    Epoch 72/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1096 - acc: 0.9593Epoch 1/100
      1/163 [..............................] - ETA: 1:03 - loss: 5.7549 - acc: 0.5000
    Epoch 00072: saving model to /content/data/model/weights.epoch_72.hdf5
    163/163 [==============================] - 95s 586ms/step - loss: 0.1101 - acc: 0.9594 - val_loss: 5.7549 - val_acc: 0.5000
    Epoch 73/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1042 - acc: 0.9622Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 4.4353 - acc: 0.5000
    Epoch 00073: saving model to /content/data/model/weights.epoch_73.hdf5
    163/163 [==============================] - 96s 589ms/step - loss: 0.1040 - acc: 0.9622 - val_loss: 4.4353 - val_acc: 0.5000
    Epoch 74/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1175 - acc: 0.9601Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 4.1697 - acc: 0.5000
    Epoch 00074: saving model to /content/data/model/weights.epoch_74.hdf5
    163/163 [==============================] - 96s 588ms/step - loss: 0.1172 - acc: 0.9601 - val_loss: 4.1697 - val_acc: 0.5000
    Epoch 75/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1216 - acc: 0.9564Epoch 1/100
      1/163 [..............................] - ETA: 1:03 - loss: 6.1661 - acc: 0.5000
    Epoch 00075: saving model to /content/data/model/weights.epoch_75.hdf5
    163/163 [==============================] - 96s 590ms/step - loss: 0.1213 - acc: 0.9565 - val_loss: 6.1661 - val_acc: 0.5000
    Epoch 76/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1102 - acc: 0.9581Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 5.2231 - acc: 0.5000
    Epoch 00076: saving model to /content/data/model/weights.epoch_76.hdf5
    163/163 [==============================] - 96s 587ms/step - loss: 0.1100 - acc: 0.9582 - val_loss: 5.2231 - val_acc: 0.5000
    Epoch 77/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1052 - acc: 0.9628Epoch 1/100
      1/163 [..............................] - ETA: 1:04 - loss: 3.6278 - acc: 0.5000
    Epoch 00077: saving model to /content/data/model/weights.epoch_77.hdf5
    163/163 [==============================] - 96s 587ms/step - loss: 0.1065 - acc: 0.9622 - val_loss: 3.6278 - val_acc: 0.5000
    Epoch 78/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1085 - acc: 0.9618Epoch 1/100
      1/163 [..............................] - ETA: 1:07 - loss: 6.8788 - acc: 0.5000
    Epoch 00078: saving model to /content/data/model/weights.epoch_78.hdf5
    163/163 [==============================] - 96s 592ms/step - loss: 0.1093 - acc: 0.9615 - val_loss: 6.8788 - val_acc: 0.5000
    Epoch 79/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1123 - acc: 0.9601Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 6.4336 - acc: 0.5000
    Epoch 00079: saving model to /content/data/model/weights.epoch_79.hdf5
    163/163 [==============================] - 96s 591ms/step - loss: 0.1122 - acc: 0.9601 - val_loss: 6.4336 - val_acc: 0.5000
    Epoch 80/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1137 - acc: 0.9589Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 5.3855 - acc: 0.5000
    Epoch 00080: saving model to /content/data/model/weights.epoch_80.hdf5
    163/163 [==============================] - 96s 587ms/step - loss: 0.1139 - acc: 0.9588 - val_loss: 5.3855 - val_acc: 0.5000
    Epoch 81/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1064 - acc: 0.9620Epoch 1/100
      1/163 [..............................] - ETA: 1:04 - loss: 6.2654 - acc: 0.5000
    Epoch 00081: saving model to /content/data/model/weights.epoch_81.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1060 - acc: 0.9622 - val_loss: 6.2654 - val_acc: 0.5000
    Epoch 82/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1029 - acc: 0.9616Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 6.9763 - acc: 0.5000
    Epoch 00082: saving model to /content/data/model/weights.epoch_82.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1036 - acc: 0.9613 - val_loss: 6.9763 - val_acc: 0.5000
    Epoch 83/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1113 - acc: 0.9585Epoch 1/100
      1/163 [..............................] - ETA: 1:04 - loss: 6.8694 - acc: 0.5000
    Epoch 00083: saving model to /content/data/model/weights.epoch_83.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1109 - acc: 0.9588 - val_loss: 6.8694 - val_acc: 0.5000
    Epoch 84/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1141 - acc: 0.9585Epoch 1/100
      1/163 [..............................] - ETA: 1:07 - loss: 5.7299 - acc: 0.5000
    Epoch 00084: saving model to /content/data/model/weights.epoch_84.hdf5
    163/163 [==============================] - 96s 587ms/step - loss: 0.1137 - acc: 0.9588 - val_loss: 5.7299 - val_acc: 0.5000
    Epoch 85/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1113 - acc: 0.9614Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 5.8954 - acc: 0.5000
    Epoch 00085: saving model to /content/data/model/weights.epoch_85.hdf5
    163/163 [==============================] - 95s 584ms/step - loss: 0.1113 - acc: 0.9615 - val_loss: 5.8954 - val_acc: 0.5000
    Epoch 86/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1134 - acc: 0.9597Epoch 1/100
      1/163 [..............................] - ETA: 1:04 - loss: 6.3760 - acc: 0.5000
    Epoch 00086: saving model to /content/data/model/weights.epoch_86.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1133 - acc: 0.9597 - val_loss: 6.3760 - val_acc: 0.5000
    Epoch 87/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1104 - acc: 0.9620Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 7.4921 - acc: 0.5000
    Epoch 00087: saving model to /content/data/model/weights.epoch_87.hdf5
    163/163 [==============================] - 95s 582ms/step - loss: 0.1112 - acc: 0.9618 - val_loss: 7.4921 - val_acc: 0.5000
    Epoch 88/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1106 - acc: 0.9618Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 5.9260 - acc: 0.5000
    Epoch 00088: saving model to /content/data/model/weights.epoch_88.hdf5
    163/163 [==============================] - 95s 586ms/step - loss: 0.1105 - acc: 0.9617 - val_loss: 5.9260 - val_acc: 0.5000
    Epoch 89/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1203 - acc: 0.9579Epoch 1/100
      1/163 [..............................] - ETA: 1:09 - loss: 7.1543 - acc: 0.5000
    Epoch 00089: saving model to /content/data/model/weights.epoch_89.hdf5
    163/163 [==============================] - 95s 584ms/step - loss: 0.1200 - acc: 0.9580 - val_loss: 7.1543 - val_acc: 0.5000
    Epoch 90/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1158 - acc: 0.9581Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 3.5203 - acc: 0.5000
    Epoch 00090: saving model to /content/data/model/weights.epoch_90.hdf5
    163/163 [==============================] - 95s 584ms/step - loss: 0.1157 - acc: 0.9580 - val_loss: 3.5203 - val_acc: 0.5000
    Epoch 91/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1065 - acc: 0.9637Epoch 1/100
      1/163 [..............................] - ETA: 1:04 - loss: 7.0161 - acc: 0.5000
    Epoch 00091: saving model to /content/data/model/weights.epoch_91.hdf5
    163/163 [==============================] - 96s 589ms/step - loss: 0.1062 - acc: 0.9638 - val_loss: 7.0161 - val_acc: 0.5000
    Epoch 92/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1140 - acc: 0.9593Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 5.8336 - acc: 0.5000
    Epoch 00092: saving model to /content/data/model/weights.epoch_92.hdf5
    163/163 [==============================] - 96s 589ms/step - loss: 0.1138 - acc: 0.9594 - val_loss: 5.8336 - val_acc: 0.5000
    Epoch 93/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1084 - acc: 0.9581Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 4.9953 - acc: 0.5000
    Epoch 00093: saving model to /content/data/model/weights.epoch_93.hdf5
    163/163 [==============================] - 96s 587ms/step - loss: 0.1083 - acc: 0.9582 - val_loss: 4.9953 - val_acc: 0.5000
    Epoch 94/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1099 - acc: 0.9606Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 5.8071 - acc: 0.5000
    Epoch 00094: saving model to /content/data/model/weights.epoch_94.hdf5
    163/163 [==============================] - 96s 589ms/step - loss: 0.1105 - acc: 0.9605 - val_loss: 5.8071 - val_acc: 0.5000
    Epoch 95/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1095 - acc: 0.9583Epoch 1/100
      1/163 [..............................] - ETA: 1:06 - loss: 5.0207 - acc: 0.5000
    Epoch 00095: saving model to /content/data/model/weights.epoch_95.hdf5
    163/163 [==============================] - 95s 586ms/step - loss: 0.1091 - acc: 0.9586 - val_loss: 5.0207 - val_acc: 0.5000
    Epoch 96/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1094 - acc: 0.9616Epoch 1/100
      1/163 [..............................] - ETA: 1:02 - loss: 5.6921 - acc: 0.5000
    Epoch 00096: saving model to /content/data/model/weights.epoch_96.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1090 - acc: 0.9617 - val_loss: 5.6921 - val_acc: 0.5000
    Epoch 97/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1058 - acc: 0.9608Epoch 1/100
      1/163 [..............................] - ETA: 1:05 - loss: 7.2704 - acc: 0.5000
    Epoch 00097: saving model to /content/data/model/weights.epoch_97.hdf5
    163/163 [==============================] - 96s 590ms/step - loss: 0.1065 - acc: 0.9607 - val_loss: 7.2704 - val_acc: 0.5000
    Epoch 98/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1090 - acc: 0.9616Epoch 1/100
      1/163 [..............................] - ETA: 1:04 - loss: 5.4627 - acc: 0.5000
    Epoch 00098: saving model to /content/data/model/weights.epoch_98.hdf5
    163/163 [==============================] - 95s 585ms/step - loss: 0.1088 - acc: 0.9617 - val_loss: 5.4627 - val_acc: 0.5000
    Epoch 99/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0983 - acc: 0.9660Epoch 1/100
      1/163 [..............................] - ETA: 1:07 - loss: 3.8404 - acc: 0.5000
    Epoch 00099: saving model to /content/data/model/weights.epoch_99.hdf5
    163/163 [==============================] - 96s 590ms/step - loss: 0.0983 - acc: 0.9661 - val_loss: 3.8404 - val_acc: 0.5000
    Epoch 100/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1069 - acc: 0.9618Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 4.4446 - acc: 0.5000
    Epoch 00100: saving model to /content/data/model/weights.epoch_100.hdf5
    163/163 [==============================] - 96s 586ms/step - loss: 0.1065 - acc: 0.9620 - val_loss: 4.4446 - val_acc: 0.5000
    Loading the best model
    epoch: 56, val_loss: 0.8659762740135193, val_acc: 0.6875
    20/20 [==============================] - 9s 443ms/step - loss: 0.9670 - acc: 0.7756
    20/20 [==============================] - 12s 597ms/step
    CONFUSION MATRIX ------------------
    [[104 130]
     [ 10 380]]
    
    TEST METRICS ----------------------
    Accuracy: 77.56410256410257%
    Precision: 74.50980392156863%
    Recall: 97.43589743589743%
    F1-score: 84.44444444444444
    
    TRAIN METRIC ----------------------
    Train acc: 96.20398879051208%
    


![png](Inception-ResNetV2%20Version%201.1.1.0.0_files/Inception-ResNetV2%20Version%201.1.1.0.0_4_1.png)

