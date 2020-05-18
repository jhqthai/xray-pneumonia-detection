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

    mkdir: cannot create directory ‘/content/data/’: File exists
    mkdir: cannot create directory ‘/content/data/model’: File exists
    mkdir: cannot create directory ‘/content/data/graph’: File exists
    --2019-10-29 04:11:13--  https://s3.eu-central-1.amazonaws.com/public.unit8.co/data/chest_xray.tar.gz
    Resolving s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)... 52.219.73.175
    Connecting to s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)|52.219.73.175|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1225393795 (1.1G) [application/x-gzip]
    Saving to: ‘/content/data/chest_xray.tar.gz’
    
    /content/data/chest 100%[===================>]   1.14G  26.2MB/s    in 46s     
    
    2019-10-29 04:11:59 (25.5 MB/s) - ‘/content/data/chest_xray.tar.gz’ saved [1225393795/1225393795]
    
    

Change log:
> training_datagen --> ImageDataGenerator

> trainable layer --> All except base

> 24 layers VGG16 model

> Optimizer = RMSprop(learning_rate = 0.0001)

> loss = categorical_crosscentropy

> callback = [checkpoints]

> epochs = 100

> **class weight balancing**



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
    class_weight=classweight,
    verbose = 1
)

# test_loss, test_acc = model.evaluate_generator(generator=test_generator, verbose=1)
```

    Found 5216 images belonging to 2 classes.
    Found 16 images belonging to 2 classes.
    Found 624 images belonging to 2 classes.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58892288/58889256 [==============================] - 1s 0us/step
    [1.9448173  0.67303226]
    Epoch 1/100
    162/163 [============================>.] - ETA: 0s - loss: 0.3802 - acc: 0.8480Epoch 1/100
      1/163 [..............................] - ETA: 4:49 - loss: 0.4098 - acc: 0.8125
    Epoch 00001: saving model to /content/data/model/weights.epoch_01.hdf5
    163/163 [==============================] - 101s 620ms/step - loss: 0.3808 - acc: 0.8478 - val_loss: 0.4098 - val_acc: 0.8125
    Epoch 2/100
    162/163 [============================>.] - ETA: 0s - loss: 0.2467 - acc: 0.9267Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.3625 - acc: 0.8125
    Epoch 00002: saving model to /content/data/model/weights.epoch_02.hdf5
    163/163 [==============================] - 97s 593ms/step - loss: 0.2467 - acc: 0.9268 - val_loss: 0.3625 - val_acc: 0.8125
    Epoch 3/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1974 - acc: 0.9408Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.3540 - acc: 0.8125
    Epoch 00003: saving model to /content/data/model/weights.epoch_03.hdf5
    163/163 [==============================] - 94s 575ms/step - loss: 0.1969 - acc: 0.9410 - val_loss: 0.3540 - val_acc: 0.8125
    Epoch 4/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1631 - acc: 0.9518Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.3638 - acc: 0.8125
    Epoch 00004: saving model to /content/data/model/weights.epoch_04.hdf5
    163/163 [==============================] - 85s 524ms/step - loss: 0.1632 - acc: 0.9519 - val_loss: 0.3638 - val_acc: 0.8125
    Epoch 5/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1548 - acc: 0.9468Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.4146 - acc: 0.8125
    Epoch 00005: saving model to /content/data/model/weights.epoch_05.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.1548 - acc: 0.9467 - val_loss: 0.4146 - val_acc: 0.8125
    Epoch 6/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1441 - acc: 0.9545Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.3516 - acc: 0.8125
    Epoch 00006: saving model to /content/data/model/weights.epoch_06.hdf5
    163/163 [==============================] - 90s 551ms/step - loss: 0.1443 - acc: 0.9546 - val_loss: 0.3516 - val_acc: 0.8125
    Epoch 7/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1440 - acc: 0.9531Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2880 - acc: 0.8125
    Epoch 00007: saving model to /content/data/model/weights.epoch_07.hdf5
    163/163 [==============================] - 90s 553ms/step - loss: 0.1435 - acc: 0.9534 - val_loss: 0.2880 - val_acc: 0.8125
    Epoch 8/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1331 - acc: 0.9545Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.4213 - acc: 0.7500
    Epoch 00008: saving model to /content/data/model/weights.epoch_08.hdf5
    163/163 [==============================] - 90s 555ms/step - loss: 0.1340 - acc: 0.9542 - val_loss: 0.4213 - val_acc: 0.7500
    Epoch 9/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1265 - acc: 0.9578Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 0.2705 - acc: 0.9375
    Epoch 00009: saving model to /content/data/model/weights.epoch_09.hdf5
    163/163 [==============================] - 90s 552ms/step - loss: 0.1267 - acc: 0.9576 - val_loss: 0.2705 - val_acc: 0.9375
    Epoch 10/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1240 - acc: 0.9560Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2467 - acc: 0.8750
    Epoch 00010: saving model to /content/data/model/weights.epoch_10.hdf5
    163/163 [==============================] - 91s 560ms/step - loss: 0.1242 - acc: 0.9557 - val_loss: 0.2467 - val_acc: 0.8750
    Epoch 11/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1154 - acc: 0.9616Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2680 - acc: 0.8750
    Epoch 00011: saving model to /content/data/model/weights.epoch_11.hdf5
    163/163 [==============================] - 91s 558ms/step - loss: 0.1150 - acc: 0.9618 - val_loss: 0.2680 - val_acc: 0.8750
    Epoch 12/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1187 - acc: 0.9593Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2426 - acc: 0.8125
    Epoch 00012: saving model to /content/data/model/weights.epoch_12.hdf5
    163/163 [==============================] - 91s 557ms/step - loss: 0.1187 - acc: 0.9594 - val_loss: 0.2426 - val_acc: 0.8125
    Epoch 13/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1176 - acc: 0.9579Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.3519 - acc: 0.7500
    Epoch 00013: saving model to /content/data/model/weights.epoch_13.hdf5
    163/163 [==============================] - 91s 555ms/step - loss: 0.1179 - acc: 0.9580 - val_loss: 0.3519 - val_acc: 0.7500
    Epoch 14/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1119 - acc: 0.9589Epoch 1/100
      1/163 [..............................] - ETA: 1:02 - loss: 0.2501 - acc: 0.8125
    Epoch 00014: saving model to /content/data/model/weights.epoch_14.hdf5
    163/163 [==============================] - 91s 557ms/step - loss: 0.1117 - acc: 0.9590 - val_loss: 0.2501 - val_acc: 0.8125
    Epoch 15/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1129 - acc: 0.9601Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2360 - acc: 0.9375
    Epoch 00015: saving model to /content/data/model/weights.epoch_15.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.1141 - acc: 0.9595 - val_loss: 0.2360 - val_acc: 0.9375
    Epoch 16/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1100 - acc: 0.9620Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2514 - acc: 0.8750
    Epoch 00016: saving model to /content/data/model/weights.epoch_16.hdf5
    163/163 [==============================] - 91s 559ms/step - loss: 0.1104 - acc: 0.9618 - val_loss: 0.2514 - val_acc: 0.8750
    Epoch 17/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1072 - acc: 0.9589Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1747 - acc: 0.9375
    Epoch 00017: saving model to /content/data/model/weights.epoch_17.hdf5
    163/163 [==============================] - 92s 561ms/step - loss: 0.1073 - acc: 0.9586 - val_loss: 0.1747 - val_acc: 0.9375
    Epoch 18/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1099 - acc: 0.9601Epoch 1/100
      1/163 [..............................] - ETA: 1:02 - loss: 0.2693 - acc: 0.8125
    Epoch 00018: saving model to /content/data/model/weights.epoch_18.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.1095 - acc: 0.9603 - val_loss: 0.2693 - val_acc: 0.8125
    Epoch 19/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1177 - acc: 0.9597Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.3865 - acc: 0.7500
    Epoch 00019: saving model to /content/data/model/weights.epoch_19.hdf5
    163/163 [==============================] - 91s 557ms/step - loss: 0.1181 - acc: 0.9594 - val_loss: 0.3865 - val_acc: 0.7500
    Epoch 20/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1034 - acc: 0.9624Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.3293 - acc: 0.8125
    Epoch 00020: saving model to /content/data/model/weights.epoch_20.hdf5
    163/163 [==============================] - 91s 560ms/step - loss: 0.1042 - acc: 0.9622 - val_loss: 0.3293 - val_acc: 0.8125
    Epoch 21/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1057 - acc: 0.9612Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1473 - acc: 0.9375
    Epoch 00021: saving model to /content/data/model/weights.epoch_21.hdf5
    163/163 [==============================] - 91s 561ms/step - loss: 0.1054 - acc: 0.9613 - val_loss: 0.1473 - val_acc: 0.9375
    Epoch 22/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1071 - acc: 0.9614Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1585 - acc: 0.9375
    Epoch 00022: saving model to /content/data/model/weights.epoch_22.hdf5
    163/163 [==============================] - 91s 558ms/step - loss: 0.1072 - acc: 0.9615 - val_loss: 0.1585 - val_acc: 0.9375
    Epoch 23/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1027 - acc: 0.9624Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2068 - acc: 0.9375
    Epoch 00023: saving model to /content/data/model/weights.epoch_23.hdf5
    163/163 [==============================] - 92s 562ms/step - loss: 0.1024 - acc: 0.9626 - val_loss: 0.2068 - val_acc: 0.9375
    Epoch 24/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1106 - acc: 0.9624Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 0.2637 - acc: 0.8125
    Epoch 00024: saving model to /content/data/model/weights.epoch_24.hdf5
    163/163 [==============================] - 91s 559ms/step - loss: 0.1104 - acc: 0.9624 - val_loss: 0.2637 - val_acc: 0.8125
    Epoch 25/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0990 - acc: 0.9639Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2560 - acc: 0.8750
    Epoch 00025: saving model to /content/data/model/weights.epoch_25.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.0986 - acc: 0.9641 - val_loss: 0.2560 - val_acc: 0.8750
    Epoch 26/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1015 - acc: 0.9666Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2135 - acc: 0.8125
    Epoch 00026: saving model to /content/data/model/weights.epoch_26.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.1013 - acc: 0.9666 - val_loss: 0.2135 - val_acc: 0.8125
    Epoch 27/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1028 - acc: 0.9633Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2201 - acc: 0.8125
    Epoch 00027: saving model to /content/data/model/weights.epoch_27.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.1024 - acc: 0.9636 - val_loss: 0.2201 - val_acc: 0.8125
    Epoch 28/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0967 - acc: 0.9664Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1996 - acc: 0.8750
    Epoch 00028: saving model to /content/data/model/weights.epoch_28.hdf5
    163/163 [==============================] - 92s 562ms/step - loss: 0.0974 - acc: 0.9663 - val_loss: 0.1996 - val_acc: 0.8750
    Epoch 29/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1130 - acc: 0.9599Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.3590 - acc: 0.8125
    Epoch 00029: saving model to /content/data/model/weights.epoch_29.hdf5
    163/163 [==============================] - 91s 561ms/step - loss: 0.1125 - acc: 0.9601 - val_loss: 0.3590 - val_acc: 0.8125
    Epoch 30/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0935 - acc: 0.9660Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2503 - acc: 0.8125
    Epoch 00030: saving model to /content/data/model/weights.epoch_30.hdf5
    163/163 [==============================] - 91s 560ms/step - loss: 0.0934 - acc: 0.9661 - val_loss: 0.2503 - val_acc: 0.8125
    Epoch 31/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1051 - acc: 0.9630Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2337 - acc: 0.8125
    Epoch 00031: saving model to /content/data/model/weights.epoch_31.hdf5
    163/163 [==============================] - 89s 547ms/step - loss: 0.1047 - acc: 0.9632 - val_loss: 0.2337 - val_acc: 0.8125
    Epoch 32/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0945 - acc: 0.9676Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 0.4280 - acc: 0.7500
    Epoch 00032: saving model to /content/data/model/weights.epoch_32.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0955 - acc: 0.9670 - val_loss: 0.4280 - val_acc: 0.7500
    Epoch 33/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0925 - acc: 0.9662Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2131 - acc: 0.8750
    Epoch 00033: saving model to /content/data/model/weights.epoch_33.hdf5
    163/163 [==============================] - 89s 547ms/step - loss: 0.0923 - acc: 0.9663 - val_loss: 0.2131 - val_acc: 0.8750
    Epoch 34/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0964 - acc: 0.9662Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1665 - acc: 0.8750
    Epoch 00034: saving model to /content/data/model/weights.epoch_34.hdf5
    163/163 [==============================] - 89s 547ms/step - loss: 0.0965 - acc: 0.9663 - val_loss: 0.1665 - val_acc: 0.8750
    Epoch 35/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1082 - acc: 0.9632Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1668 - acc: 0.9375
    Epoch 00035: saving model to /content/data/model/weights.epoch_35.hdf5
    163/163 [==============================] - 90s 549ms/step - loss: 0.1078 - acc: 0.9634 - val_loss: 0.1668 - val_acc: 0.9375
    Epoch 36/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0995 - acc: 0.9659Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1578 - acc: 0.9375
    Epoch 00036: saving model to /content/data/model/weights.epoch_36.hdf5
    163/163 [==============================] - 90s 551ms/step - loss: 0.1000 - acc: 0.9657 - val_loss: 0.1578 - val_acc: 0.9375
    Epoch 37/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0978 - acc: 0.9659Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1709 - acc: 0.8750
    Epoch 00037: saving model to /content/data/model/weights.epoch_37.hdf5
    163/163 [==============================] - 90s 550ms/step - loss: 0.0977 - acc: 0.9661 - val_loss: 0.1709 - val_acc: 0.8750
    Epoch 38/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0869 - acc: 0.9711Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1146 - acc: 0.9375
    Epoch 00038: saving model to /content/data/model/weights.epoch_38.hdf5
    163/163 [==============================] - 89s 546ms/step - loss: 0.0868 - acc: 0.9711 - val_loss: 0.1146 - val_acc: 0.9375
    Epoch 39/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0909 - acc: 0.9678Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1466 - acc: 0.9375
    Epoch 00039: saving model to /content/data/model/weights.epoch_39.hdf5
    163/163 [==============================] - 90s 553ms/step - loss: 0.0911 - acc: 0.9676 - val_loss: 0.1466 - val_acc: 0.9375
    Epoch 40/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0933 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1306 - acc: 1.0000
    Epoch 00040: saving model to /content/data/model/weights.epoch_40.hdf5
    163/163 [==============================] - 89s 546ms/step - loss: 0.0937 - acc: 0.9680 - val_loss: 0.1306 - val_acc: 1.0000
    Epoch 41/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0853 - acc: 0.9686Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.1785 - acc: 0.9375
    Epoch 00041: saving model to /content/data/model/weights.epoch_41.hdf5
    163/163 [==============================] - 90s 554ms/step - loss: 0.0862 - acc: 0.9684 - val_loss: 0.1785 - val_acc: 0.9375
    Epoch 42/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0926 - acc: 0.9680Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1822 - acc: 0.8750
    Epoch 00042: saving model to /content/data/model/weights.epoch_42.hdf5
    163/163 [==============================] - 90s 555ms/step - loss: 0.0923 - acc: 0.9682 - val_loss: 0.1822 - val_acc: 0.8750
    Epoch 43/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0979 - acc: 0.9662Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2030 - acc: 0.8750
    Epoch 00043: saving model to /content/data/model/weights.epoch_43.hdf5
    163/163 [==============================] - 89s 549ms/step - loss: 0.0978 - acc: 0.9663 - val_loss: 0.2030 - val_acc: 0.8750
    Epoch 44/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0952 - acc: 0.9659Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1978 - acc: 0.8750
    Epoch 00044: saving model to /content/data/model/weights.epoch_44.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.0948 - acc: 0.9661 - val_loss: 0.1978 - val_acc: 0.8750
    Epoch 45/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0936 - acc: 0.9678Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1818 - acc: 0.8750
    Epoch 00045: saving model to /content/data/model/weights.epoch_45.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0936 - acc: 0.9678 - val_loss: 0.1818 - val_acc: 0.8750
    Epoch 46/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0861 - acc: 0.9713Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1545 - acc: 0.8750
    Epoch 00046: saving model to /content/data/model/weights.epoch_46.hdf5
    163/163 [==============================] - 90s 549ms/step - loss: 0.0857 - acc: 0.9714 - val_loss: 0.1545 - val_acc: 0.8750
    Epoch 47/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0890 - acc: 0.9709Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1602 - acc: 0.9375
    Epoch 00047: saving model to /content/data/model/weights.epoch_47.hdf5
    163/163 [==============================] - 89s 548ms/step - loss: 0.0890 - acc: 0.9709 - val_loss: 0.1602 - val_acc: 0.9375
    Epoch 48/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0929 - acc: 0.9678Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.0931 - acc: 0.9375
    Epoch 00048: saving model to /content/data/model/weights.epoch_48.hdf5
    163/163 [==============================] - 90s 550ms/step - loss: 0.0927 - acc: 0.9678 - val_loss: 0.0931 - val_acc: 0.9375
    Epoch 49/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0920 - acc: 0.9711Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2514 - acc: 0.8750
    Epoch 00049: saving model to /content/data/model/weights.epoch_49.hdf5
    163/163 [==============================] - 90s 553ms/step - loss: 0.0916 - acc: 0.9712 - val_loss: 0.2514 - val_acc: 0.8750
    Epoch 50/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0924 - acc: 0.9693Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1498 - acc: 1.0000
    Epoch 00050: saving model to /content/data/model/weights.epoch_50.hdf5
    163/163 [==============================] - 90s 550ms/step - loss: 0.0933 - acc: 0.9684 - val_loss: 0.1498 - val_acc: 1.0000
    Epoch 51/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0896 - acc: 0.9662Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.1287 - acc: 0.9375
    Epoch 00051: saving model to /content/data/model/weights.epoch_51.hdf5
    163/163 [==============================] - 89s 548ms/step - loss: 0.0895 - acc: 0.9664 - val_loss: 0.1287 - val_acc: 0.9375
    Epoch 52/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0868 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2034 - acc: 0.8750
    Epoch 00052: saving model to /content/data/model/weights.epoch_52.hdf5
    163/163 [==============================] - 91s 557ms/step - loss: 0.0875 - acc: 0.9678 - val_loss: 0.2034 - val_acc: 0.8750
    Epoch 53/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0898 - acc: 0.9680Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0875 - acc: 1.0000
    Epoch 00053: saving model to /content/data/model/weights.epoch_53.hdf5
    163/163 [==============================] - 91s 555ms/step - loss: 0.0896 - acc: 0.9680 - val_loss: 0.0875 - val_acc: 1.0000
    Epoch 54/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0914 - acc: 0.9697Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2708 - acc: 0.8750
    Epoch 00054: saving model to /content/data/model/weights.epoch_54.hdf5
    163/163 [==============================] - 90s 550ms/step - loss: 0.0919 - acc: 0.9695 - val_loss: 0.2708 - val_acc: 0.8750
    Epoch 55/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0914 - acc: 0.9680Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1194 - acc: 1.0000
    Epoch 00055: saving model to /content/data/model/weights.epoch_55.hdf5
    163/163 [==============================] - 89s 548ms/step - loss: 0.0910 - acc: 0.9682 - val_loss: 0.1194 - val_acc: 1.0000
    Epoch 56/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0851 - acc: 0.9699Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1983 - acc: 0.8750
    Epoch 00056: saving model to /content/data/model/weights.epoch_56.hdf5
    163/163 [==============================] - 90s 550ms/step - loss: 0.0850 - acc: 0.9699 - val_loss: 0.1983 - val_acc: 0.8750
    Epoch 57/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0849 - acc: 0.9732Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1570 - acc: 0.8750
    Epoch 00057: saving model to /content/data/model/weights.epoch_57.hdf5
    163/163 [==============================] - 89s 549ms/step - loss: 0.0845 - acc: 0.9734 - val_loss: 0.1570 - val_acc: 0.8750
    Epoch 58/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0887 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1099 - acc: 1.0000
    Epoch 00058: saving model to /content/data/model/weights.epoch_58.hdf5
    163/163 [==============================] - 89s 546ms/step - loss: 0.0883 - acc: 0.9689 - val_loss: 0.1099 - val_acc: 1.0000
    Epoch 59/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0835 - acc: 0.9705Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1104 - acc: 1.0000
    Epoch 00059: saving model to /content/data/model/weights.epoch_59.hdf5
    163/163 [==============================] - 89s 546ms/step - loss: 0.0837 - acc: 0.9703 - val_loss: 0.1104 - val_acc: 1.0000
    Epoch 60/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0908 - acc: 0.9662Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1089 - acc: 1.0000
    Epoch 00060: saving model to /content/data/model/weights.epoch_60.hdf5
    163/163 [==============================] - 89s 548ms/step - loss: 0.0907 - acc: 0.9663 - val_loss: 0.1089 - val_acc: 1.0000
    Epoch 61/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0908 - acc: 0.9689Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1040 - acc: 1.0000
    Epoch 00061: saving model to /content/data/model/weights.epoch_61.hdf5
    163/163 [==============================] - 90s 549ms/step - loss: 0.0907 - acc: 0.9689 - val_loss: 0.1040 - val_acc: 1.0000
    Epoch 62/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0866 - acc: 0.9686Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0850 - acc: 0.9375
    Epoch 00062: saving model to /content/data/model/weights.epoch_62.hdf5
    163/163 [==============================] - 90s 550ms/step - loss: 0.0872 - acc: 0.9682 - val_loss: 0.0850 - val_acc: 0.9375
    Epoch 63/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0908 - acc: 0.9680Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0773 - acc: 1.0000
    Epoch 00063: saving model to /content/data/model/weights.epoch_63.hdf5
    163/163 [==============================] - 90s 554ms/step - loss: 0.0912 - acc: 0.9680 - val_loss: 0.0773 - val_acc: 1.0000
    Epoch 64/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0943 - acc: 0.9680Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.0843 - acc: 1.0000
    Epoch 00064: saving model to /content/data/model/weights.epoch_64.hdf5
    163/163 [==============================] - 90s 550ms/step - loss: 0.0939 - acc: 0.9682 - val_loss: 0.0843 - val_acc: 1.0000
    Epoch 65/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0948 - acc: 0.9666Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1199 - acc: 1.0000
    Epoch 00065: saving model to /content/data/model/weights.epoch_65.hdf5
    163/163 [==============================] - 90s 551ms/step - loss: 0.0945 - acc: 0.9668 - val_loss: 0.1199 - val_acc: 1.0000
    Epoch 66/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0889 - acc: 0.9664Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2478 - acc: 0.8750
    Epoch 00066: saving model to /content/data/model/weights.epoch_66.hdf5
    163/163 [==============================] - 91s 555ms/step - loss: 0.0914 - acc: 0.9661 - val_loss: 0.2478 - val_acc: 0.8750
    Epoch 67/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0983 - acc: 0.9664Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1133 - acc: 0.9375
    Epoch 00067: saving model to /content/data/model/weights.epoch_67.hdf5
    163/163 [==============================] - 90s 552ms/step - loss: 0.0983 - acc: 0.9664 - val_loss: 0.1133 - val_acc: 0.9375
    Epoch 68/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0903 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1708 - acc: 0.8750
    Epoch 00068: saving model to /content/data/model/weights.epoch_68.hdf5
    163/163 [==============================] - 90s 553ms/step - loss: 0.0905 - acc: 0.9684 - val_loss: 0.1708 - val_acc: 0.8750
    Epoch 69/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0911 - acc: 0.9649Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0929 - acc: 1.0000
    Epoch 00069: saving model to /content/data/model/weights.epoch_69.hdf5
    163/163 [==============================] - 90s 549ms/step - loss: 0.0908 - acc: 0.9649 - val_loss: 0.0929 - val_acc: 1.0000
    Epoch 70/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0886 - acc: 0.9674Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2361 - acc: 0.8750
    Epoch 00070: saving model to /content/data/model/weights.epoch_70.hdf5
    163/163 [==============================] - 90s 549ms/step - loss: 0.0888 - acc: 0.9674 - val_loss: 0.2361 - val_acc: 0.8750
    Epoch 71/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0844 - acc: 0.9713Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1138 - acc: 1.0000
    Epoch 00071: saving model to /content/data/model/weights.epoch_71.hdf5
    163/163 [==============================] - 89s 546ms/step - loss: 0.0845 - acc: 0.9712 - val_loss: 0.1138 - val_acc: 1.0000
    Epoch 72/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0767 - acc: 0.9730Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2393 - acc: 0.9375
    Epoch 00072: saving model to /content/data/model/weights.epoch_72.hdf5
    163/163 [==============================] - 89s 545ms/step - loss: 0.0781 - acc: 0.9726 - val_loss: 0.2393 - val_acc: 0.9375
    Epoch 73/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0840 - acc: 0.9695Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.0807 - acc: 0.9375
    Epoch 00073: saving model to /content/data/model/weights.epoch_73.hdf5
    163/163 [==============================] - 90s 551ms/step - loss: 0.0838 - acc: 0.9695 - val_loss: 0.0807 - val_acc: 0.9375
    Epoch 74/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0964 - acc: 0.9660Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1297 - acc: 0.9375
    Epoch 00074: saving model to /content/data/model/weights.epoch_74.hdf5
    163/163 [==============================] - 89s 549ms/step - loss: 0.0959 - acc: 0.9663 - val_loss: 0.1297 - val_acc: 0.9375
    Epoch 75/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0766 - acc: 0.9730Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1688 - acc: 0.9375
    Epoch 00075: saving model to /content/data/model/weights.epoch_75.hdf5
    163/163 [==============================] - 89s 548ms/step - loss: 0.0765 - acc: 0.9730 - val_loss: 0.1688 - val_acc: 0.9375
    Epoch 76/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0785 - acc: 0.9707Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2829 - acc: 0.7500
    Epoch 00076: saving model to /content/data/model/weights.epoch_76.hdf5
    163/163 [==============================] - 90s 550ms/step - loss: 0.0784 - acc: 0.9707 - val_loss: 0.2829 - val_acc: 0.7500
    Epoch 77/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0839 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0655 - acc: 1.0000
    Epoch 00077: saving model to /content/data/model/weights.epoch_77.hdf5
    163/163 [==============================] - 91s 556ms/step - loss: 0.0836 - acc: 0.9684 - val_loss: 0.0655 - val_acc: 1.0000
    Epoch 78/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0780 - acc: 0.9732Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.0590 - acc: 1.0000
    Epoch 00078: saving model to /content/data/model/weights.epoch_78.hdf5
    163/163 [==============================] - 90s 553ms/step - loss: 0.0777 - acc: 0.9734 - val_loss: 0.0590 - val_acc: 1.0000
    Epoch 79/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0826 - acc: 0.9718Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1054 - acc: 0.9375
    Epoch 00079: saving model to /content/data/model/weights.epoch_79.hdf5
    163/163 [==============================] - 90s 553ms/step - loss: 0.0823 - acc: 0.9720 - val_loss: 0.1054 - val_acc: 0.9375
    Epoch 80/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0765 - acc: 0.9740Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1753 - acc: 0.8750
    Epoch 00080: saving model to /content/data/model/weights.epoch_80.hdf5
    163/163 [==============================] - 90s 553ms/step - loss: 0.0763 - acc: 0.9741 - val_loss: 0.1753 - val_acc: 0.8750
    Epoch 81/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0812 - acc: 0.9699Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2651 - acc: 0.8750
    Epoch 00081: saving model to /content/data/model/weights.epoch_81.hdf5
    163/163 [==============================] - 90s 551ms/step - loss: 0.0809 - acc: 0.9701 - val_loss: 0.2651 - val_acc: 0.8750
    Epoch 82/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0869 - acc: 0.9705Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.3787 - acc: 0.7500
    Epoch 00082: saving model to /content/data/model/weights.epoch_82.hdf5
    163/163 [==============================] - 89s 545ms/step - loss: 0.0865 - acc: 0.9707 - val_loss: 0.3787 - val_acc: 0.7500
    Epoch 83/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0887 - acc: 0.9691Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2015 - acc: 0.8750
    Epoch 00083: saving model to /content/data/model/weights.epoch_83.hdf5
    163/163 [==============================] - 89s 547ms/step - loss: 0.0885 - acc: 0.9691 - val_loss: 0.2015 - val_acc: 0.8750
    Epoch 84/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0846 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.1080 - acc: 1.0000
    Epoch 00084: saving model to /content/data/model/weights.epoch_84.hdf5
    163/163 [==============================] - 89s 548ms/step - loss: 0.0842 - acc: 0.9703 - val_loss: 0.1080 - val_acc: 1.0000
    Epoch 85/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0929 - acc: 0.9684Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.0726 - acc: 1.0000
    Epoch 00085: saving model to /content/data/model/weights.epoch_85.hdf5
    163/163 [==============================] - 89s 548ms/step - loss: 0.0924 - acc: 0.9686 - val_loss: 0.0726 - val_acc: 1.0000
    Epoch 86/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0847 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1016 - acc: 1.0000
    Epoch 00086: saving model to /content/data/model/weights.epoch_86.hdf5
    163/163 [==============================] - 89s 547ms/step - loss: 0.0845 - acc: 0.9701 - val_loss: 0.1016 - val_acc: 1.0000
    Epoch 87/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0818 - acc: 0.9722Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0875 - acc: 0.9375
    Epoch 00087: saving model to /content/data/model/weights.epoch_87.hdf5
    163/163 [==============================] - 89s 548ms/step - loss: 0.0822 - acc: 0.9718 - val_loss: 0.0875 - val_acc: 0.9375
    Epoch 88/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0775 - acc: 0.9751Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1044 - acc: 1.0000
    Epoch 00088: saving model to /content/data/model/weights.epoch_88.hdf5
    163/163 [==============================] - 90s 550ms/step - loss: 0.0777 - acc: 0.9749 - val_loss: 0.1044 - val_acc: 1.0000
    Epoch 89/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0717 - acc: 0.9743Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1166 - acc: 1.0000
    Epoch 00089: saving model to /content/data/model/weights.epoch_89.hdf5
    163/163 [==============================] - 89s 546ms/step - loss: 0.0714 - acc: 0.9745 - val_loss: 0.1166 - val_acc: 1.0000
    Epoch 90/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0875 - acc: 0.9684Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1030 - acc: 0.9375
    Epoch 00090: saving model to /content/data/model/weights.epoch_90.hdf5
    163/163 [==============================] - 89s 544ms/step - loss: 0.0871 - acc: 0.9686 - val_loss: 0.1030 - val_acc: 0.9375
    Epoch 91/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0774 - acc: 0.9718Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1324 - acc: 0.8750
    Epoch 00091: saving model to /content/data/model/weights.epoch_91.hdf5
    163/163 [==============================] - 90s 550ms/step - loss: 0.0770 - acc: 0.9720 - val_loss: 0.1324 - val_acc: 0.8750
    Epoch 92/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0756 - acc: 0.9736Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0777 - acc: 1.0000
    Epoch 00092: saving model to /content/data/model/weights.epoch_92.hdf5
    163/163 [==============================] - 90s 551ms/step - loss: 0.0753 - acc: 0.9737 - val_loss: 0.0777 - val_acc: 1.0000
    Epoch 93/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0778 - acc: 0.9707Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.0962 - acc: 0.9375
    Epoch 00093: saving model to /content/data/model/weights.epoch_93.hdf5
    163/163 [==============================] - 89s 548ms/step - loss: 0.0783 - acc: 0.9701 - val_loss: 0.0962 - val_acc: 0.9375
    Epoch 94/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0835 - acc: 0.9715Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2383 - acc: 0.8750
    Epoch 00094: saving model to /content/data/model/weights.epoch_94.hdf5
    163/163 [==============================] - 90s 550ms/step - loss: 0.0831 - acc: 0.9716 - val_loss: 0.2383 - val_acc: 0.8750
    Epoch 95/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0758 - acc: 0.9757Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.0441 - acc: 1.0000
    Epoch 00095: saving model to /content/data/model/weights.epoch_95.hdf5
    163/163 [==============================] - 90s 552ms/step - loss: 0.0769 - acc: 0.9757 - val_loss: 0.0441 - val_acc: 1.0000
    Epoch 96/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0728 - acc: 0.9734Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1091 - acc: 1.0000
    Epoch 00096: saving model to /content/data/model/weights.epoch_96.hdf5
    163/163 [==============================] - 89s 549ms/step - loss: 0.0725 - acc: 0.9735 - val_loss: 0.1091 - val_acc: 1.0000
    Epoch 97/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0753 - acc: 0.9745Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1151 - acc: 0.9375
    Epoch 00097: saving model to /content/data/model/weights.epoch_97.hdf5
    163/163 [==============================] - 90s 553ms/step - loss: 0.0753 - acc: 0.9745 - val_loss: 0.1151 - val_acc: 0.9375
    Epoch 98/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0771 - acc: 0.9716Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.0734 - acc: 0.9375
    Epoch 00098: saving model to /content/data/model/weights.epoch_98.hdf5
    163/163 [==============================] - 90s 555ms/step - loss: 0.0771 - acc: 0.9716 - val_loss: 0.0734 - val_acc: 0.9375
    Epoch 99/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0748 - acc: 0.9745Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0815 - acc: 1.0000
    Epoch 00099: saving model to /content/data/model/weights.epoch_99.hdf5
    163/163 [==============================] - 93s 571ms/step - loss: 0.0745 - acc: 0.9745 - val_loss: 0.0815 - val_acc: 1.0000
    Epoch 100/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0805 - acc: 0.9720Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0945 - acc: 0.9375
    Epoch 00100: saving model to /content/data/model/weights.epoch_100.hdf5
    163/163 [==============================] - 92s 564ms/step - loss: 0.0813 - acc: 0.9716 - val_loss: 0.0945 - val_acc: 0.9375
    


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


![png](VGG16%20Model%202%20Version%202.1.1.1.0_files/VGG16%20Model%202%20Version%202.1.1.1.0_5_0.png)



```
idx = np.argmin(history.history['val_loss']) 
model.load_weights("/content/data/model/weights.epoch_{:02d}.hdf5".format(idx + 1))

print("Loading the best model")
print("epoch: {}, val_loss: {}, val_acc: {}".format(idx + 1, history.history['val_loss'][idx], history.history['val_acc'][idx]))
```

    Loading the best model
    epoch: 95, val_loss: 0.04409787803888321, val_acc: 1.0
    


```
test_loss, test_acc = model.evaluate_generator(generator=test_generator, verbose=1)
```

    20/20 [==============================] - 8s 419ms/step - loss: 0.2661 - acc: 0.9135
    


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

    20/20 [==============================] - 8s 390ms/step
    CONFUSION MATRIX ------------------
    [[197  37]
     [ 17 373]]
    
    TEST METRICS ----------------------
    Accuracy: 91.34615384615384%
    Precision: 90.97560975609757%
    Recall: 95.64102564102565%
    F1-score: 93.25
    
    TRAIN METRIC ----------------------
    Train acc: 97.16257452964781%
    
