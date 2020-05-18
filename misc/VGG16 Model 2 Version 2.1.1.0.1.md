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

    --2019-10-29 04:15:35--  https://s3.eu-central-1.amazonaws.com/public.unit8.co/data/chest_xray.tar.gz
    Resolving s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)... 52.219.72.111
    Connecting to s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)|52.219.72.111|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1225393795 (1.1G) [application/x-gzip]
    Saving to: ‘/content/data/chest_xray.tar.gz’
    
    /content/data/chest 100%[===================>]   1.14G  26.3MB/s    in 46s     
    
    2019-10-29 04:16:22 (25.4 MB/s) - ‘/content/data/chest_xray.tar.gz’ saved [1225393795/1225393795]
    
    

Change log:
> training_datagen --> ImageDataGenerator

> trainable layer --> All except base

> 24 layers VGG16 model

> Optimizer = RMSprop(learning_rate = 0.0001)

> loss = categorical_crosscentropy

> callback = [checkpoints]

> epochs = 100

> no class weight balancing

> **batchsize = 128**



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

base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=train_shape)

x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.33)(x)
x = tf.keras.layers.BatchNormalization()(x)
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

batch_size = 128
epochs = 100

step_size_train = train_generator.n // train_generator.batch_size
step_size_valid = validation_generator.n // validation_generator.batch_size

# Training process
history = model.fit_generator(
    generator=train_generator, 
    steps_per_epoch=step_size_train, 
    epochs=epochs,
    # callbacks=[early_stopping_monitor],
    callbacks=[checkpoint],
    # shuffle=True, 
    validation_data=validation_generator, 
    # validation_steps= step_size_valid, #no because it's gonna be 0... if leave alone its len(generator) which is equal to 1. 
    # class_weight=classweight,
    verbose = 1
)

# test_loss, test_acc = model.evaluate_generator(generator=test_generator, verbose=1)
```

    Found 5216 images belonging to 2 classes.
    Found 16 images belonging to 2 classes.
    Found 624 images belonging to 2 classes.
    [1.9448173  0.67303226]
    Epoch 1/100
    162/163 [============================>.] - ETA: 0s - loss: 0.3731 - acc: 0.8557Epoch 1/100
      1/163 [..............................] - ETA: 4:27 - loss: 0.4169 - acc: 0.9375
    Epoch 00001: saving model to /content/data/model/weights.epoch_01.hdf5
    163/163 [==============================] - 90s 553ms/step - loss: 0.3734 - acc: 0.8558 - val_loss: 0.4169 - val_acc: 0.9375
    Epoch 2/100
    162/163 [============================>.] - ETA: 0s - loss: 0.2292 - acc: 0.9334Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.3405 - acc: 0.7500
    Epoch 00002: saving model to /content/data/model/weights.epoch_02.hdf5
    163/163 [==============================] - 85s 523ms/step - loss: 0.2290 - acc: 0.9337 - val_loss: 0.3405 - val_acc: 0.7500
    Epoch 3/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1887 - acc: 0.9419Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.3502 - acc: 0.7500
    Epoch 00003: saving model to /content/data/model/weights.epoch_03.hdf5
    163/163 [==============================] - 85s 521ms/step - loss: 0.1886 - acc: 0.9417 - val_loss: 0.3502 - val_acc: 0.7500
    Epoch 4/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1627 - acc: 0.9497Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 0.4173 - acc: 0.7500
    Epoch 00004: saving model to /content/data/model/weights.epoch_04.hdf5
    163/163 [==============================] - 84s 516ms/step - loss: 0.1624 - acc: 0.9498 - val_loss: 0.4173 - val_acc: 0.7500
    Epoch 5/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1659 - acc: 0.9450Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.3685 - acc: 0.8125
    Epoch 00005: saving model to /content/data/model/weights.epoch_05.hdf5
    163/163 [==============================] - 84s 517ms/step - loss: 0.1656 - acc: 0.9450 - val_loss: 0.3685 - val_acc: 0.8125
    Epoch 6/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1529 - acc: 0.9475Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.3973 - acc: 0.7500
    Epoch 00006: saving model to /content/data/model/weights.epoch_06.hdf5
    163/163 [==============================] - 84s 518ms/step - loss: 0.1540 - acc: 0.9475 - val_loss: 0.3973 - val_acc: 0.7500
    Epoch 7/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1418 - acc: 0.9514Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.6133 - acc: 0.7500
    Epoch 00007: saving model to /content/data/model/weights.epoch_07.hdf5
    163/163 [==============================] - 86s 525ms/step - loss: 0.1418 - acc: 0.9515 - val_loss: 0.6133 - val_acc: 0.7500
    Epoch 8/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1383 - acc: 0.9510Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.3999 - acc: 0.9375
    Epoch 00008: saving model to /content/data/model/weights.epoch_08.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.1382 - acc: 0.9509 - val_loss: 0.3999 - val_acc: 0.9375
    Epoch 9/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1304 - acc: 0.9570Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.3102 - acc: 0.8125
    Epoch 00009: saving model to /content/data/model/weights.epoch_09.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.1304 - acc: 0.9571 - val_loss: 0.3102 - val_acc: 0.8125
    Epoch 10/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1297 - acc: 0.9543Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.3345 - acc: 0.8125
    Epoch 00010: saving model to /content/data/model/weights.epoch_10.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.1296 - acc: 0.9544 - val_loss: 0.3345 - val_acc: 0.8125
    Epoch 11/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1280 - acc: 0.9543Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.3529 - acc: 0.8125
    Epoch 00011: saving model to /content/data/model/weights.epoch_11.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.1276 - acc: 0.9544 - val_loss: 0.3529 - val_acc: 0.8125
    Epoch 12/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1310 - acc: 0.9512Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2507 - acc: 0.8125
    Epoch 00012: saving model to /content/data/model/weights.epoch_12.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.1310 - acc: 0.9511 - val_loss: 0.2507 - val_acc: 0.8125
    Epoch 13/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1306 - acc: 0.9514Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2854 - acc: 0.8125
    Epoch 00013: saving model to /content/data/model/weights.epoch_13.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.1300 - acc: 0.9517 - val_loss: 0.2854 - val_acc: 0.8125
    Epoch 14/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1248 - acc: 0.9520Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.3107 - acc: 0.8125
    Epoch 00014: saving model to /content/data/model/weights.epoch_14.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.1246 - acc: 0.9523 - val_loss: 0.3107 - val_acc: 0.8125
    Epoch 15/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1326 - acc: 0.9518Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.2202 - acc: 0.9375
    Epoch 00015: saving model to /content/data/model/weights.epoch_15.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.1324 - acc: 0.9519 - val_loss: 0.2202 - val_acc: 0.9375
    Epoch 16/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1229 - acc: 0.9574Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2141 - acc: 0.8750
    Epoch 00016: saving model to /content/data/model/weights.epoch_16.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.1239 - acc: 0.9569 - val_loss: 0.2141 - val_acc: 0.8750
    Epoch 17/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1214 - acc: 0.9578Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.3292 - acc: 0.8750
    Epoch 00017: saving model to /content/data/model/weights.epoch_17.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.1209 - acc: 0.9580 - val_loss: 0.3292 - val_acc: 0.8750
    Epoch 18/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1118 - acc: 0.9610Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1904 - acc: 0.9375
    Epoch 00018: saving model to /content/data/model/weights.epoch_18.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.1134 - acc: 0.9605 - val_loss: 0.1904 - val_acc: 0.9375
    Epoch 19/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1073 - acc: 0.9605Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2356 - acc: 0.8125
    Epoch 00019: saving model to /content/data/model/weights.epoch_19.hdf5
    163/163 [==============================] - 89s 544ms/step - loss: 0.1079 - acc: 0.9603 - val_loss: 0.2356 - val_acc: 0.8125
    Epoch 20/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1264 - acc: 0.9576Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2328 - acc: 0.8125
    Epoch 00020: saving model to /content/data/model/weights.epoch_20.hdf5
    163/163 [==============================] - 89s 549ms/step - loss: 0.1261 - acc: 0.9576 - val_loss: 0.2328 - val_acc: 0.8125
    Epoch 21/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1082 - acc: 0.9620Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2988 - acc: 0.8750
    Epoch 00021: saving model to /content/data/model/weights.epoch_21.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.1083 - acc: 0.9618 - val_loss: 0.2988 - val_acc: 0.8750
    Epoch 22/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1160 - acc: 0.9587Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.4808 - acc: 0.7500
    Epoch 00022: saving model to /content/data/model/weights.epoch_22.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.1155 - acc: 0.9590 - val_loss: 0.4808 - val_acc: 0.7500
    Epoch 23/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1138 - acc: 0.9587Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 0.3481 - acc: 0.8750
    Epoch 00023: saving model to /content/data/model/weights.epoch_23.hdf5
    163/163 [==============================] - 92s 562ms/step - loss: 0.1140 - acc: 0.9588 - val_loss: 0.3481 - val_acc: 0.8750
    Epoch 24/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1029 - acc: 0.9620Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.5066 - acc: 0.7500
    Epoch 00024: saving model to /content/data/model/weights.epoch_24.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1027 - acc: 0.9620 - val_loss: 0.5066 - val_acc: 0.7500
    Epoch 25/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1052 - acc: 0.9633Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2796 - acc: 0.9375
    Epoch 00025: saving model to /content/data/model/weights.epoch_25.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.1056 - acc: 0.9632 - val_loss: 0.2796 - val_acc: 0.9375
    Epoch 26/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1152 - acc: 0.9610Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.4961 - acc: 0.7500
    Epoch 00026: saving model to /content/data/model/weights.epoch_26.hdf5
    163/163 [==============================] - 93s 570ms/step - loss: 0.1148 - acc: 0.9613 - val_loss: 0.4961 - val_acc: 0.7500
    Epoch 27/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1080 - acc: 0.9612Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.4943 - acc: 0.7500
    Epoch 00027: saving model to /content/data/model/weights.epoch_27.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.1077 - acc: 0.9613 - val_loss: 0.4943 - val_acc: 0.7500
    Epoch 28/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1023 - acc: 0.9672Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2127 - acc: 0.8125
    Epoch 00028: saving model to /content/data/model/weights.epoch_28.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.1021 - acc: 0.9674 - val_loss: 0.2127 - val_acc: 0.8125
    Epoch 29/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1045 - acc: 0.9618Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 0.2548 - acc: 0.9375
    Epoch 00029: saving model to /content/data/model/weights.epoch_29.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.1047 - acc: 0.9618 - val_loss: 0.2548 - val_acc: 0.9375
    Epoch 30/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1055 - acc: 0.9599Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 0.4053 - acc: 0.8125
    Epoch 00030: saving model to /content/data/model/weights.epoch_30.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1060 - acc: 0.9597 - val_loss: 0.4053 - val_acc: 0.8125
    Epoch 31/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1066 - acc: 0.9632Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.5711 - acc: 0.6875
    Epoch 00031: saving model to /content/data/model/weights.epoch_31.hdf5
    163/163 [==============================] - 91s 561ms/step - loss: 0.1077 - acc: 0.9630 - val_loss: 0.5711 - val_acc: 0.6875
    Epoch 32/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1058 - acc: 0.9651Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.4007 - acc: 0.8750
    Epoch 00032: saving model to /content/data/model/weights.epoch_32.hdf5
    163/163 [==============================] - 93s 571ms/step - loss: 0.1060 - acc: 0.9651 - val_loss: 0.4007 - val_acc: 0.8750
    Epoch 33/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1109 - acc: 0.9591Epoch 1/100
      1/163 [..............................] - ETA: 1:02 - loss: 0.2471 - acc: 0.8125
    Epoch 00033: saving model to /content/data/model/weights.epoch_33.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.1105 - acc: 0.9592 - val_loss: 0.2471 - val_acc: 0.8125
    Epoch 34/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1022 - acc: 0.9626Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1648 - acc: 0.9375
    Epoch 00034: saving model to /content/data/model/weights.epoch_34.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.1025 - acc: 0.9624 - val_loss: 0.1648 - val_acc: 0.9375
    Epoch 35/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1010 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.3858 - acc: 0.7500
    Epoch 00035: saving model to /content/data/model/weights.epoch_35.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1007 - acc: 0.9684 - val_loss: 0.3858 - val_acc: 0.7500
    Epoch 36/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1065 - acc: 0.9612Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.2015 - acc: 0.8125
    Epoch 00036: saving model to /content/data/model/weights.epoch_36.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.1060 - acc: 0.9615 - val_loss: 0.2015 - val_acc: 0.8125
    Epoch 37/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1044 - acc: 0.9620Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.3078 - acc: 0.8750
    Epoch 00037: saving model to /content/data/model/weights.epoch_37.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.1061 - acc: 0.9615 - val_loss: 0.3078 - val_acc: 0.8750
    Epoch 38/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0994 - acc: 0.9641Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 0.4050 - acc: 0.7500
    Epoch 00038: saving model to /content/data/model/weights.epoch_38.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.1000 - acc: 0.9640 - val_loss: 0.4050 - val_acc: 0.7500
    Epoch 39/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0972 - acc: 0.9645Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.4122 - acc: 0.6875
    Epoch 00039: saving model to /content/data/model/weights.epoch_39.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.0975 - acc: 0.9645 - val_loss: 0.4122 - val_acc: 0.6875
    Epoch 40/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0988 - acc: 0.9643Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2440 - acc: 0.8125
    Epoch 00040: saving model to /content/data/model/weights.epoch_40.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.0986 - acc: 0.9643 - val_loss: 0.2440 - val_acc: 0.8125
    Epoch 41/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1052 - acc: 0.9651Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.3317 - acc: 0.8125
    Epoch 00041: saving model to /content/data/model/weights.epoch_41.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.1055 - acc: 0.9649 - val_loss: 0.3317 - val_acc: 0.8125
    Epoch 42/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1019 - acc: 0.9641Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1347 - acc: 0.9375
    Epoch 00042: saving model to /content/data/model/weights.epoch_42.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.1015 - acc: 0.9641 - val_loss: 0.1347 - val_acc: 0.9375
    Epoch 43/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0979 - acc: 0.9641Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2481 - acc: 0.8750
    Epoch 00043: saving model to /content/data/model/weights.epoch_43.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.0983 - acc: 0.9640 - val_loss: 0.2481 - val_acc: 0.8750
    Epoch 44/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0962 - acc: 0.9686Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1823 - acc: 0.8750
    Epoch 00044: saving model to /content/data/model/weights.epoch_44.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0957 - acc: 0.9688 - val_loss: 0.1823 - val_acc: 0.8750
    Epoch 45/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0857 - acc: 0.9689Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2455 - acc: 0.8750
    Epoch 00045: saving model to /content/data/model/weights.epoch_45.hdf5
    163/163 [==============================] - 91s 561ms/step - loss: 0.0858 - acc: 0.9688 - val_loss: 0.2455 - val_acc: 0.8750
    Epoch 46/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1046 - acc: 0.9659Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2159 - acc: 0.8750
    Epoch 00046: saving model to /content/data/model/weights.epoch_46.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.1047 - acc: 0.9657 - val_loss: 0.2159 - val_acc: 0.8750
    Epoch 47/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1057 - acc: 0.9626Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.2620 - acc: 0.8125
    Epoch 00047: saving model to /content/data/model/weights.epoch_47.hdf5
    163/163 [==============================] - 91s 558ms/step - loss: 0.1052 - acc: 0.9628 - val_loss: 0.2620 - val_acc: 0.8125
    Epoch 48/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1008 - acc: 0.9662Epoch 1/100
      1/163 [..............................] - ETA: 1:03 - loss: 0.2468 - acc: 0.8750
    Epoch 00048: saving model to /content/data/model/weights.epoch_48.hdf5
    163/163 [==============================] - 91s 560ms/step - loss: 0.1020 - acc: 0.9657 - val_loss: 0.2468 - val_acc: 0.8750
    Epoch 49/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0892 - acc: 0.9680Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.5087 - acc: 0.6875
    Epoch 00049: saving model to /content/data/model/weights.epoch_49.hdf5
    163/163 [==============================] - 91s 561ms/step - loss: 0.0893 - acc: 0.9678 - val_loss: 0.5087 - val_acc: 0.6875
    Epoch 50/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1011 - acc: 0.9676Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2828 - acc: 0.8125
    Epoch 00050: saving model to /content/data/model/weights.epoch_50.hdf5
    163/163 [==============================] - 91s 560ms/step - loss: 0.1010 - acc: 0.9676 - val_loss: 0.2828 - val_acc: 0.8125
    Epoch 51/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0929 - acc: 0.9672Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1985 - acc: 0.8750
    Epoch 00051: saving model to /content/data/model/weights.epoch_51.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.0928 - acc: 0.9672 - val_loss: 0.1985 - val_acc: 0.8750
    Epoch 52/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1020 - acc: 0.9641Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 0.2546 - acc: 0.8125
    Epoch 00052: saving model to /content/data/model/weights.epoch_52.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.1020 - acc: 0.9641 - val_loss: 0.2546 - val_acc: 0.8125
    Epoch 53/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0939 - acc: 0.9674Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.1982 - acc: 0.8750
    Epoch 00053: saving model to /content/data/model/weights.epoch_53.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.0935 - acc: 0.9676 - val_loss: 0.1982 - val_acc: 0.8750
    Epoch 54/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0957 - acc: 0.9645Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.3821 - acc: 0.8125
    Epoch 00054: saving model to /content/data/model/weights.epoch_54.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.0956 - acc: 0.9643 - val_loss: 0.3821 - val_acc: 0.8125
    Epoch 55/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0941 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1872 - acc: 0.9375
    Epoch 00055: saving model to /content/data/model/weights.epoch_55.hdf5
    163/163 [==============================] - 92s 562ms/step - loss: 0.0948 - acc: 0.9680 - val_loss: 0.1872 - val_acc: 0.9375
    Epoch 56/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0983 - acc: 0.9643Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1748 - acc: 0.9375
    Epoch 00056: saving model to /content/data/model/weights.epoch_56.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0979 - acc: 0.9645 - val_loss: 0.1748 - val_acc: 0.9375
    Epoch 57/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0899 - acc: 0.9689Epoch 1/100
      1/163 [..............................] - ETA: 1:02 - loss: 0.1884 - acc: 0.8125
    Epoch 00057: saving model to /content/data/model/weights.epoch_57.hdf5
    163/163 [==============================] - 93s 569ms/step - loss: 0.0915 - acc: 0.9684 - val_loss: 0.1884 - val_acc: 0.8125
    Epoch 58/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0974 - acc: 0.9653Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2093 - acc: 0.8750
    Epoch 00058: saving model to /content/data/model/weights.epoch_58.hdf5
    163/163 [==============================] - 92s 561ms/step - loss: 0.0973 - acc: 0.9651 - val_loss: 0.2093 - val_acc: 0.8750
    Epoch 59/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0914 - acc: 0.9674Epoch 1/100
      1/163 [..............................] - ETA: 1:08 - loss: 0.2185 - acc: 0.8750
    Epoch 00059: saving model to /content/data/model/weights.epoch_59.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.0918 - acc: 0.9674 - val_loss: 0.2185 - val_acc: 0.8750
    Epoch 60/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0880 - acc: 0.9697Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1639 - acc: 0.9375
    Epoch 00060: saving model to /content/data/model/weights.epoch_60.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0877 - acc: 0.9697 - val_loss: 0.1639 - val_acc: 0.9375
    Epoch 61/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0932 - acc: 0.9676Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1833 - acc: 0.8750
    Epoch 00061: saving model to /content/data/model/weights.epoch_61.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0939 - acc: 0.9670 - val_loss: 0.1833 - val_acc: 0.8750
    Epoch 62/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0989 - acc: 0.9637Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.4403 - acc: 0.7500
    Epoch 00062: saving model to /content/data/model/weights.epoch_62.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.0985 - acc: 0.9640 - val_loss: 0.4403 - val_acc: 0.7500
    Epoch 63/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0908 - acc: 0.9705Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2128 - acc: 0.8750
    Epoch 00063: saving model to /content/data/model/weights.epoch_63.hdf5
    163/163 [==============================] - 91s 561ms/step - loss: 0.0915 - acc: 0.9705 - val_loss: 0.2128 - val_acc: 0.8750
    Epoch 64/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0974 - acc: 0.9647Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1410 - acc: 0.9375
    Epoch 00064: saving model to /content/data/model/weights.epoch_64.hdf5
    163/163 [==============================] - 93s 568ms/step - loss: 0.0975 - acc: 0.9647 - val_loss: 0.1410 - val_acc: 0.9375
    Epoch 65/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0910 - acc: 0.9695Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1505 - acc: 0.9375
    Epoch 00065: saving model to /content/data/model/weights.epoch_65.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0910 - acc: 0.9693 - val_loss: 0.1505 - val_acc: 0.9375
    Epoch 66/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0902 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1599 - acc: 0.9375
    Epoch 00066: saving model to /content/data/model/weights.epoch_66.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.0905 - acc: 0.9680 - val_loss: 0.1599 - val_acc: 0.9375
    Epoch 67/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0929 - acc: 0.9674Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 0.1942 - acc: 0.8750
    Epoch 00067: saving model to /content/data/model/weights.epoch_67.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.0927 - acc: 0.9676 - val_loss: 0.1942 - val_acc: 0.8750
    Epoch 68/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0899 - acc: 0.9705Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.2455 - acc: 0.8750
    Epoch 00068: saving model to /content/data/model/weights.epoch_68.hdf5
    163/163 [==============================] - 92s 561ms/step - loss: 0.0897 - acc: 0.9705 - val_loss: 0.2455 - val_acc: 0.8750
    Epoch 69/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0955 - acc: 0.9659Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.1850 - acc: 0.9375
    Epoch 00069: saving model to /content/data/model/weights.epoch_69.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.0955 - acc: 0.9657 - val_loss: 0.1850 - val_acc: 0.9375
    Epoch 70/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0820 - acc: 0.9716Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.3135 - acc: 0.8125
    Epoch 00070: saving model to /content/data/model/weights.epoch_70.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.0819 - acc: 0.9716 - val_loss: 0.3135 - val_acc: 0.8125
    Epoch 71/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0924 - acc: 0.9691Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 0.1686 - acc: 0.8750
    Epoch 00071: saving model to /content/data/model/weights.epoch_71.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.0921 - acc: 0.9693 - val_loss: 0.1686 - val_acc: 0.8750
    Epoch 72/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0958 - acc: 0.9655Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 0.3796 - acc: 0.7500
    Epoch 00072: saving model to /content/data/model/weights.epoch_72.hdf5
    163/163 [==============================] - 93s 570ms/step - loss: 0.0958 - acc: 0.9655 - val_loss: 0.3796 - val_acc: 0.7500
    Epoch 73/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0947 - acc: 0.9686Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1221 - acc: 0.9375
    Epoch 00073: saving model to /content/data/model/weights.epoch_73.hdf5
    163/163 [==============================] - 93s 568ms/step - loss: 0.0944 - acc: 0.9688 - val_loss: 0.1221 - val_acc: 0.9375
    Epoch 74/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0908 - acc: 0.9703Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1438 - acc: 0.9375
    Epoch 00074: saving model to /content/data/model/weights.epoch_74.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.0908 - acc: 0.9703 - val_loss: 0.1438 - val_acc: 0.9375
    Epoch 75/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0881 - acc: 0.9707Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1826 - acc: 0.8750
    Epoch 00075: saving model to /content/data/model/weights.epoch_75.hdf5
    163/163 [==============================] - 91s 560ms/step - loss: 0.0878 - acc: 0.9709 - val_loss: 0.1826 - val_acc: 0.8750
    Epoch 76/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0941 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1326 - acc: 0.9375
    Epoch 00076: saving model to /content/data/model/weights.epoch_76.hdf5
    163/163 [==============================] - 93s 569ms/step - loss: 0.0939 - acc: 0.9680 - val_loss: 0.1326 - val_acc: 0.9375
    Epoch 77/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0877 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1591 - acc: 0.8750
    Epoch 00077: saving model to /content/data/model/weights.epoch_77.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.0875 - acc: 0.9688 - val_loss: 0.1591 - val_acc: 0.8750
    Epoch 78/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0957 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 0.1860 - acc: 0.8750
    Epoch 00078: saving model to /content/data/model/weights.epoch_78.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0966 - acc: 0.9680 - val_loss: 0.1860 - val_acc: 0.8750
    Epoch 79/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1010 - acc: 0.9618Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2063 - acc: 0.8125
    Epoch 00079: saving model to /content/data/model/weights.epoch_79.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.1009 - acc: 0.9618 - val_loss: 0.2063 - val_acc: 0.8125
    Epoch 80/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0910 - acc: 0.9674Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.3628 - acc: 0.7500
    Epoch 00080: saving model to /content/data/model/weights.epoch_80.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0905 - acc: 0.9676 - val_loss: 0.3628 - val_acc: 0.7500
    Epoch 81/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0833 - acc: 0.9705Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1777 - acc: 0.9375
    Epoch 00081: saving model to /content/data/model/weights.epoch_81.hdf5
    163/163 [==============================] - 93s 569ms/step - loss: 0.0836 - acc: 0.9703 - val_loss: 0.1777 - val_acc: 0.9375
    Epoch 82/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0895 - acc: 0.9691Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 0.1777 - acc: 0.9375
    Epoch 00082: saving model to /content/data/model/weights.epoch_82.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0891 - acc: 0.9693 - val_loss: 0.1777 - val_acc: 0.9375
    Epoch 83/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0890 - acc: 0.9680Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2701 - acc: 0.8125
    Epoch 00083: saving model to /content/data/model/weights.epoch_83.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.0887 - acc: 0.9680 - val_loss: 0.2701 - val_acc: 0.8125
    Epoch 84/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0830 - acc: 0.9718Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1363 - acc: 0.9375
    Epoch 00084: saving model to /content/data/model/weights.epoch_84.hdf5
    163/163 [==============================] - 93s 571ms/step - loss: 0.0830 - acc: 0.9716 - val_loss: 0.1363 - val_acc: 0.9375
    Epoch 85/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0862 - acc: 0.9693Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 0.1144 - acc: 1.0000
    Epoch 00085: saving model to /content/data/model/weights.epoch_85.hdf5
    163/163 [==============================] - 93s 572ms/step - loss: 0.0864 - acc: 0.9689 - val_loss: 0.1144 - val_acc: 1.0000
    Epoch 86/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0909 - acc: 0.9659Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2144 - acc: 0.8750
    Epoch 00086: saving model to /content/data/model/weights.epoch_86.hdf5
    163/163 [==============================] - 93s 568ms/step - loss: 0.0906 - acc: 0.9661 - val_loss: 0.2144 - val_acc: 0.8750
    Epoch 87/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0855 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1609 - acc: 0.9375
    Epoch 00087: saving model to /content/data/model/weights.epoch_87.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.0852 - acc: 0.9703 - val_loss: 0.1609 - val_acc: 0.9375
    Epoch 88/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0844 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 1:04 - loss: 0.1537 - acc: 0.9375
    Epoch 00088: saving model to /content/data/model/weights.epoch_88.hdf5
    163/163 [==============================] - 93s 569ms/step - loss: 0.0845 - acc: 0.9697 - val_loss: 0.1537 - val_acc: 0.9375
    Epoch 89/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0876 - acc: 0.9680Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.3423 - acc: 0.7500
    Epoch 00089: saving model to /content/data/model/weights.epoch_89.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.0874 - acc: 0.9680 - val_loss: 0.3423 - val_acc: 0.7500
    Epoch 90/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0867 - acc: 0.9689Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.0911 - acc: 1.0000
    Epoch 00090: saving model to /content/data/model/weights.epoch_90.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0871 - acc: 0.9689 - val_loss: 0.0911 - val_acc: 1.0000
    Epoch 91/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0884 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.1374 - acc: 0.9375
    Epoch 00091: saving model to /content/data/model/weights.epoch_91.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.0884 - acc: 0.9688 - val_loss: 0.1374 - val_acc: 0.9375
    Epoch 92/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0960 - acc: 0.9678Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1169 - acc: 0.9375
    Epoch 00092: saving model to /content/data/model/weights.epoch_92.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.0956 - acc: 0.9680 - val_loss: 0.1169 - val_acc: 0.9375
    Epoch 93/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0969 - acc: 0.9662Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 0.1792 - acc: 0.9375
    Epoch 00093: saving model to /content/data/model/weights.epoch_93.hdf5
    163/163 [==============================] - 92s 562ms/step - loss: 0.0965 - acc: 0.9664 - val_loss: 0.1792 - val_acc: 0.9375
    Epoch 94/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0866 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 0.1591 - acc: 0.9375
    Epoch 00094: saving model to /content/data/model/weights.epoch_94.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.0868 - acc: 0.9697 - val_loss: 0.1591 - val_acc: 0.9375
    Epoch 95/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0883 - acc: 0.9699Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 0.1322 - acc: 0.9375
    Epoch 00095: saving model to /content/data/model/weights.epoch_95.hdf5
    163/163 [==============================] - 92s 567ms/step - loss: 0.0884 - acc: 0.9699 - val_loss: 0.1322 - val_acc: 0.9375
    Epoch 96/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0869 - acc: 0.9670Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.1446 - acc: 0.9375
    Epoch 00096: saving model to /content/data/model/weights.epoch_96.hdf5
    163/163 [==============================] - 93s 568ms/step - loss: 0.0865 - acc: 0.9672 - val_loss: 0.1446 - val_acc: 0.9375
    Epoch 97/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0858 - acc: 0.9678Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1481 - acc: 0.9375
    Epoch 00097: saving model to /content/data/model/weights.epoch_97.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.0855 - acc: 0.9680 - val_loss: 0.1481 - val_acc: 0.9375
    Epoch 98/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0926 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 0.1280 - acc: 0.9375
    Epoch 00098: saving model to /content/data/model/weights.epoch_98.hdf5
    163/163 [==============================] - 92s 565ms/step - loss: 0.0934 - acc: 0.9697 - val_loss: 0.1280 - val_acc: 0.9375
    Epoch 99/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0843 - acc: 0.9722Epoch 1/100
      1/163 [..............................] - ETA: 1:02 - loss: 0.1512 - acc: 0.8750
    Epoch 00099: saving model to /content/data/model/weights.epoch_99.hdf5
    163/163 [==============================] - 92s 566ms/step - loss: 0.0841 - acc: 0.9722 - val_loss: 0.1512 - val_acc: 0.8750
    Epoch 100/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0831 - acc: 0.9730Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.1511 - acc: 0.9375
    Epoch 00100: saving model to /content/data/model/weights.epoch_100.hdf5
    163/163 [==============================] - 92s 562ms/step - loss: 0.0832 - acc: 0.9728 - val_loss: 0.1511 - val_acc: 0.9375
    


```
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
```


![png](VGG16%20Model%202%20Version%202.1.1.0.1_files/VGG16%20Model%202%20Version%202.1.1.0.1_5_0.png)



```
idx = np.argmin(history.history['val_loss']) 
model.load_weights("/content/data/model/weights.epoch_{:02d}.hdf5".format(idx + 1))

print("Loading the best model")
print("epoch: {}, val_loss: {}, val_acc: {}".format(idx + 1, history.history['val_loss'][idx], history.history['val_acc'][idx]))
```

    Loading the best model
    epoch: 90, val_loss: 0.09109506011009216, val_acc: 1.0
    


```
test_loss, test_acc = model.evaluate_generator(generator=test_generator, verbose=1)
```

    20/20 [==============================] - 8s 403ms/step - loss: 0.2567 - acc: 0.9183
    


```
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

    20/20 [==============================] - 8s 408ms/step
    CONFUSION MATRIX ------------------
    [[200  34]
     [ 17 373]]
    
    TEST METRICS ----------------------
    Accuracy: 91.82692307692307%
    Precision: 91.64619164619164%
    Recall: 95.64102564102565%
    F1-score: 93.60100376411543
    
    TRAIN METRIC ----------------------
    Train acc: 97.27760553359985%
    
