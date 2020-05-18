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

    --2019-10-28 09:05:47--  https://s3.eu-central-1.amazonaws.com/public.unit8.co/data/chest_xray.tar.gz
    Resolving s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)... 52.219.73.147
    Connecting to s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)|52.219.73.147|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1225393795 (1.1G) [application/x-gzip]
    Saving to: ‘/content/data/chest_xray.tar.gz’
    
    /content/data/chest 100%[===================>]   1.14G  11.5MB/s    in 1m 45s  
    
    2019-10-28 09:07:33 (11.1 MB/s) - ‘/content/data/chest_xray.tar.gz’ saved [1225393795/1225393795]
    
    

Change log:
> training_datagen --> ImageDataGenerator

> trainable layer --> All except base

> 20 layers VGG16 model - base, flat, dense

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

base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=train_shape)

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
```

    Found 5216 images belonging to 2 classes.
    Found 16 images belonging to 2 classes.
    Found 624 images belonging to 2 classes.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58892288/58889256 [==============================] - 5s 0us/step
    [1.9448173  0.67303226]
    Epoch 1/100
    162/163 [============================>.] - ETA: 0s - loss: 0.3272 - acc: 0.8611Epoch 1/100
      1/163 [..............................] - ETA: 4:26 - loss: 0.3919 - acc: 0.8125
    Epoch 00001: saving model to /content/data/model/weights.epoch_01.hdf5
    163/163 [==============================] - 89s 546ms/step - loss: 0.3262 - acc: 0.8616 - val_loss: 0.3919 - val_acc: 0.8125
    Epoch 2/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1985 - acc: 0.9306Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.3267 - acc: 0.8750
    Epoch 00002: saving model to /content/data/model/weights.epoch_02.hdf5
    163/163 [==============================] - 85s 523ms/step - loss: 0.1994 - acc: 0.9300 - val_loss: 0.3267 - val_acc: 0.8750
    Epoch 3/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1699 - acc: 0.9392Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.3100 - acc: 0.8750
    Epoch 00003: saving model to /content/data/model/weights.epoch_03.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.1695 - acc: 0.9394 - val_loss: 0.3100 - val_acc: 0.8750
    Epoch 4/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1526 - acc: 0.9441Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.3196 - acc: 0.8750
    Epoch 00004: saving model to /content/data/model/weights.epoch_04.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.1521 - acc: 0.9444 - val_loss: 0.3196 - val_acc: 0.8750
    Epoch 5/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1438 - acc: 0.9443Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.4042 - acc: 0.8125
    Epoch 00005: saving model to /content/data/model/weights.epoch_05.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.1442 - acc: 0.9440 - val_loss: 0.4042 - val_acc: 0.8125
    Epoch 6/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1353 - acc: 0.9493Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2808 - acc: 0.8125
    Epoch 00006: saving model to /content/data/model/weights.epoch_06.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.1363 - acc: 0.9486 - val_loss: 0.2808 - val_acc: 0.8125
    Epoch 7/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1350 - acc: 0.9529Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 0.3641 - acc: 0.8125
    Epoch 00007: saving model to /content/data/model/weights.epoch_07.hdf5
    163/163 [==============================] - 86s 525ms/step - loss: 0.1348 - acc: 0.9528 - val_loss: 0.3641 - val_acc: 0.8125
    Epoch 8/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1300 - acc: 0.9504Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.3363 - acc: 0.8750
    Epoch 00008: saving model to /content/data/model/weights.epoch_08.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.1302 - acc: 0.9503 - val_loss: 0.3363 - val_acc: 0.8750
    Epoch 9/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1273 - acc: 0.9520Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.3568 - acc: 0.8125
    Epoch 00009: saving model to /content/data/model/weights.epoch_09.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.1276 - acc: 0.9519 - val_loss: 0.3568 - val_acc: 0.8125
    Epoch 10/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1233 - acc: 0.9558Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.3402 - acc: 0.8750
    Epoch 00010: saving model to /content/data/model/weights.epoch_10.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.1237 - acc: 0.9553 - val_loss: 0.3402 - val_acc: 0.8750
    Epoch 11/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1201 - acc: 0.9562Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2826 - acc: 0.8125
    Epoch 00011: saving model to /content/data/model/weights.epoch_11.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.1205 - acc: 0.9555 - val_loss: 0.2826 - val_acc: 0.8125
    Epoch 12/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1174 - acc: 0.9566Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2523 - acc: 0.8125
    Epoch 00012: saving model to /content/data/model/weights.epoch_12.hdf5
    163/163 [==============================] - 87s 537ms/step - loss: 0.1175 - acc: 0.9565 - val_loss: 0.2523 - val_acc: 0.8125
    Epoch 13/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1190 - acc: 0.9543Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.3225 - acc: 0.8750
    Epoch 00013: saving model to /content/data/model/weights.epoch_13.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.1186 - acc: 0.9546 - val_loss: 0.3225 - val_acc: 0.8750
    Epoch 14/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1149 - acc: 0.9593Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.3168 - acc: 0.8750
    Epoch 00014: saving model to /content/data/model/weights.epoch_14.hdf5
    163/163 [==============================] - 88s 537ms/step - loss: 0.1154 - acc: 0.9586 - val_loss: 0.3168 - val_acc: 0.8750
    Epoch 15/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1174 - acc: 0.9579Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2474 - acc: 0.8125
    Epoch 00015: saving model to /content/data/model/weights.epoch_15.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.1180 - acc: 0.9574 - val_loss: 0.2474 - val_acc: 0.8125
    Epoch 16/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1118 - acc: 0.9579Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2492 - acc: 0.8125
    Epoch 00016: saving model to /content/data/model/weights.epoch_16.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.1121 - acc: 0.9578 - val_loss: 0.2492 - val_acc: 0.8125
    Epoch 17/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1108 - acc: 0.9616Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2344 - acc: 0.8125
    Epoch 00017: saving model to /content/data/model/weights.epoch_17.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.1103 - acc: 0.9618 - val_loss: 0.2344 - val_acc: 0.8125
    Epoch 18/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1063 - acc: 0.9606Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 0.2972 - acc: 0.8750
    Epoch 00018: saving model to /content/data/model/weights.epoch_18.hdf5
    163/163 [==============================] - 85s 522ms/step - loss: 0.1061 - acc: 0.9607 - val_loss: 0.2972 - val_acc: 0.8750
    Epoch 19/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1107 - acc: 0.9599Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.2626 - acc: 0.8750
    Epoch 00019: saving model to /content/data/model/weights.epoch_19.hdf5
    163/163 [==============================] - 85s 520ms/step - loss: 0.1105 - acc: 0.9599 - val_loss: 0.2626 - val_acc: 0.8750
    Epoch 20/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1058 - acc: 0.9639Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.3010 - acc: 0.8750
    Epoch 00020: saving model to /content/data/model/weights.epoch_20.hdf5
    163/163 [==============================] - 85s 522ms/step - loss: 0.1056 - acc: 0.9641 - val_loss: 0.3010 - val_acc: 0.8750
    Epoch 21/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1055 - acc: 0.9614Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2826 - acc: 0.8750
    Epoch 00021: saving model to /content/data/model/weights.epoch_21.hdf5
    163/163 [==============================] - 83s 509ms/step - loss: 0.1058 - acc: 0.9613 - val_loss: 0.2826 - val_acc: 0.8750
    Epoch 22/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1095 - acc: 0.9605Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2716 - acc: 0.8750
    Epoch 00022: saving model to /content/data/model/weights.epoch_22.hdf5
    163/163 [==============================] - 83s 510ms/step - loss: 0.1092 - acc: 0.9605 - val_loss: 0.2716 - val_acc: 0.8750
    Epoch 23/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1034 - acc: 0.9608Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2642 - acc: 0.8750
    Epoch 00023: saving model to /content/data/model/weights.epoch_23.hdf5
    163/163 [==============================] - 83s 510ms/step - loss: 0.1033 - acc: 0.9609 - val_loss: 0.2642 - val_acc: 0.8750
    Epoch 24/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1060 - acc: 0.9605Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2502 - acc: 0.8750
    Epoch 00024: saving model to /content/data/model/weights.epoch_24.hdf5
    163/163 [==============================] - 82s 504ms/step - loss: 0.1057 - acc: 0.9607 - val_loss: 0.2502 - val_acc: 0.8750
    Epoch 25/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0994 - acc: 0.9622Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2606 - acc: 0.8750
    Epoch 00025: saving model to /content/data/model/weights.epoch_25.hdf5
    163/163 [==============================] - 82s 503ms/step - loss: 0.1000 - acc: 0.9618 - val_loss: 0.2606 - val_acc: 0.8750
    Epoch 26/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0993 - acc: 0.9622Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2393 - acc: 0.8125
    Epoch 00026: saving model to /content/data/model/weights.epoch_26.hdf5
    163/163 [==============================] - 82s 505ms/step - loss: 0.0989 - acc: 0.9624 - val_loss: 0.2393 - val_acc: 0.8125
    Epoch 27/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0994 - acc: 0.9659Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2544 - acc: 0.8750
    Epoch 00027: saving model to /content/data/model/weights.epoch_27.hdf5
    163/163 [==============================] - 84s 518ms/step - loss: 0.0997 - acc: 0.9659 - val_loss: 0.2544 - val_acc: 0.8750
    Epoch 28/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1002 - acc: 0.9635Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2212 - acc: 0.8125
    Epoch 00028: saving model to /content/data/model/weights.epoch_28.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.1002 - acc: 0.9634 - val_loss: 0.2212 - val_acc: 0.8125
    Epoch 29/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1051 - acc: 0.9620Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2403 - acc: 0.8750
    Epoch 00029: saving model to /content/data/model/weights.epoch_29.hdf5
    163/163 [==============================] - 88s 537ms/step - loss: 0.1049 - acc: 0.9622 - val_loss: 0.2403 - val_acc: 0.8750
    Epoch 30/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0953 - acc: 0.9632Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2363 - acc: 0.8125
    Epoch 00030: saving model to /content/data/model/weights.epoch_30.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0958 - acc: 0.9632 - val_loss: 0.2363 - val_acc: 0.8125
    Epoch 31/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1003 - acc: 0.9645Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2638 - acc: 0.8750
    Epoch 00031: saving model to /content/data/model/weights.epoch_31.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.1003 - acc: 0.9645 - val_loss: 0.2638 - val_acc: 0.8750
    Epoch 32/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0993 - acc: 0.9601Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2800 - acc: 0.8750
    Epoch 00032: saving model to /content/data/model/weights.epoch_32.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0995 - acc: 0.9599 - val_loss: 0.2800 - val_acc: 0.8750
    Epoch 33/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0931 - acc: 0.9657Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.1983 - acc: 0.8125
    Epoch 00033: saving model to /content/data/model/weights.epoch_33.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0931 - acc: 0.9657 - val_loss: 0.1983 - val_acc: 0.8125
    Epoch 34/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0919 - acc: 0.9684Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2493 - acc: 0.8750
    Epoch 00034: saving model to /content/data/model/weights.epoch_34.hdf5
    163/163 [==============================] - 87s 537ms/step - loss: 0.0916 - acc: 0.9686 - val_loss: 0.2493 - val_acc: 0.8750
    Epoch 35/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0941 - acc: 0.9655Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2328 - acc: 0.8750
    Epoch 00035: saving model to /content/data/model/weights.epoch_35.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0940 - acc: 0.9655 - val_loss: 0.2328 - val_acc: 0.8750
    Epoch 36/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0957 - acc: 0.9660Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2161 - acc: 0.8750
    Epoch 00036: saving model to /content/data/model/weights.epoch_36.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.0958 - acc: 0.9661 - val_loss: 0.2161 - val_acc: 0.8750
    Epoch 37/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0919 - acc: 0.9660Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1894 - acc: 0.8125
    Epoch 00037: saving model to /content/data/model/weights.epoch_37.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.0918 - acc: 0.9661 - val_loss: 0.1894 - val_acc: 0.8125
    Epoch 38/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0915 - acc: 0.9662Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.2136 - acc: 0.8750
    Epoch 00038: saving model to /content/data/model/weights.epoch_38.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0914 - acc: 0.9663 - val_loss: 0.2136 - val_acc: 0.8750
    Epoch 39/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0902 - acc: 0.9649Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2224 - acc: 0.8750
    Epoch 00039: saving model to /content/data/model/weights.epoch_39.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0901 - acc: 0.9649 - val_loss: 0.2224 - val_acc: 0.8750
    Epoch 40/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0947 - acc: 0.9632Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2040 - acc: 0.8125
    Epoch 00040: saving model to /content/data/model/weights.epoch_40.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0945 - acc: 0.9632 - val_loss: 0.2040 - val_acc: 0.8125
    Epoch 41/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0941 - acc: 0.9616Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2177 - acc: 0.8750
    Epoch 00041: saving model to /content/data/model/weights.epoch_41.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0944 - acc: 0.9615 - val_loss: 0.2177 - val_acc: 0.8750
    Epoch 42/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0937 - acc: 0.9649Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1838 - acc: 0.8750
    Epoch 00042: saving model to /content/data/model/weights.epoch_42.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0937 - acc: 0.9649 - val_loss: 0.1838 - val_acc: 0.8750
    Epoch 43/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0897 - acc: 0.9668Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1971 - acc: 0.8750
    Epoch 00043: saving model to /content/data/model/weights.epoch_43.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0892 - acc: 0.9670 - val_loss: 0.1971 - val_acc: 0.8750
    Epoch 44/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0882 - acc: 0.9689Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1768 - acc: 0.9375
    Epoch 00044: saving model to /content/data/model/weights.epoch_44.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.0882 - acc: 0.9689 - val_loss: 0.1768 - val_acc: 0.9375
    Epoch 45/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0877 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2417 - acc: 0.8750
    Epoch 00045: saving model to /content/data/model/weights.epoch_45.hdf5
    163/163 [==============================] - 89s 544ms/step - loss: 0.0877 - acc: 0.9686 - val_loss: 0.2417 - val_acc: 0.8750
    Epoch 46/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0915 - acc: 0.9676Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1968 - acc: 0.8125
    Epoch 00046: saving model to /content/data/model/weights.epoch_46.hdf5
    163/163 [==============================] - 89s 545ms/step - loss: 0.0923 - acc: 0.9672 - val_loss: 0.1968 - val_acc: 0.8125
    Epoch 47/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0865 - acc: 0.9670Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1954 - acc: 0.8125
    Epoch 00047: saving model to /content/data/model/weights.epoch_47.hdf5
    163/163 [==============================] - 89s 543ms/step - loss: 0.0874 - acc: 0.9664 - val_loss: 0.1954 - val_acc: 0.8125
    Epoch 48/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0910 - acc: 0.9676Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1813 - acc: 0.8750
    Epoch 00048: saving model to /content/data/model/weights.epoch_48.hdf5
    163/163 [==============================] - 89s 544ms/step - loss: 0.0906 - acc: 0.9678 - val_loss: 0.1813 - val_acc: 0.8750
    Epoch 49/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0864 - acc: 0.9691Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1764 - acc: 0.9375
    Epoch 00049: saving model to /content/data/model/weights.epoch_49.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.0867 - acc: 0.9689 - val_loss: 0.1764 - val_acc: 0.9375
    Epoch 50/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0837 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.1720 - acc: 0.9375
    Epoch 00050: saving model to /content/data/model/weights.epoch_50.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0835 - acc: 0.9703 - val_loss: 0.1720 - val_acc: 0.9375
    Epoch 51/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0838 - acc: 0.9693Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1691 - acc: 0.8750
    Epoch 00051: saving model to /content/data/model/weights.epoch_51.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0834 - acc: 0.9695 - val_loss: 0.1691 - val_acc: 0.8750
    Epoch 52/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0875 - acc: 0.9664Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1598 - acc: 0.9375
    Epoch 00052: saving model to /content/data/model/weights.epoch_52.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0879 - acc: 0.9663 - val_loss: 0.1598 - val_acc: 0.9375
    Epoch 53/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0833 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2053 - acc: 0.8750
    Epoch 00053: saving model to /content/data/model/weights.epoch_53.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.0832 - acc: 0.9689 - val_loss: 0.2053 - val_acc: 0.8750
    Epoch 54/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0880 - acc: 0.9691Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2352 - acc: 0.8750
    Epoch 00054: saving model to /content/data/model/weights.epoch_54.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.0878 - acc: 0.9691 - val_loss: 0.2352 - val_acc: 0.8750
    Epoch 55/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0878 - acc: 0.9664Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.1857 - acc: 0.8125
    Epoch 00055: saving model to /content/data/model/weights.epoch_55.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.0877 - acc: 0.9664 - val_loss: 0.1857 - val_acc: 0.8125
    Epoch 56/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0896 - acc: 0.9666Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1655 - acc: 0.9375
    Epoch 00056: saving model to /content/data/model/weights.epoch_56.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0900 - acc: 0.9664 - val_loss: 0.1655 - val_acc: 0.9375
    Epoch 57/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0815 - acc: 0.9734Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.2047 - acc: 0.8750
    Epoch 00057: saving model to /content/data/model/weights.epoch_57.hdf5
    163/163 [==============================] - 89s 543ms/step - loss: 0.0819 - acc: 0.9730 - val_loss: 0.2047 - val_acc: 0.8750
    Epoch 58/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0831 - acc: 0.9711Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1619 - acc: 0.9375
    Epoch 00058: saving model to /content/data/model/weights.epoch_58.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0827 - acc: 0.9712 - val_loss: 0.1619 - val_acc: 0.9375
    Epoch 59/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0893 - acc: 0.9670Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2043 - acc: 0.8750
    Epoch 00059: saving model to /content/data/model/weights.epoch_59.hdf5
    163/163 [==============================] - 89s 544ms/step - loss: 0.0890 - acc: 0.9672 - val_loss: 0.2043 - val_acc: 0.8750
    Epoch 60/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0845 - acc: 0.9703Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1591 - acc: 0.9375
    Epoch 00060: saving model to /content/data/model/weights.epoch_60.hdf5
    163/163 [==============================] - 88s 543ms/step - loss: 0.0841 - acc: 0.9705 - val_loss: 0.1591 - val_acc: 0.9375
    Epoch 61/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0831 - acc: 0.9693Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1507 - acc: 1.0000
    Epoch 00061: saving model to /content/data/model/weights.epoch_61.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0833 - acc: 0.9693 - val_loss: 0.1507 - val_acc: 1.0000
    Epoch 62/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0841 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1635 - acc: 1.0000
    Epoch 00062: saving model to /content/data/model/weights.epoch_62.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.0839 - acc: 0.9682 - val_loss: 0.1635 - val_acc: 1.0000
    Epoch 63/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0807 - acc: 0.9705Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1537 - acc: 1.0000
    Epoch 00063: saving model to /content/data/model/weights.epoch_63.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0802 - acc: 0.9707 - val_loss: 0.1537 - val_acc: 1.0000
    Epoch 64/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0805 - acc: 0.9713Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1437 - acc: 0.9375
    Epoch 00064: saving model to /content/data/model/weights.epoch_64.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.0813 - acc: 0.9711 - val_loss: 0.1437 - val_acc: 0.9375
    Epoch 65/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0826 - acc: 0.9707Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1765 - acc: 0.9375
    Epoch 00065: saving model to /content/data/model/weights.epoch_65.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0824 - acc: 0.9707 - val_loss: 0.1765 - val_acc: 0.9375
    Epoch 66/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0794 - acc: 0.9697Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1714 - acc: 0.9375
    Epoch 00066: saving model to /content/data/model/weights.epoch_66.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.0797 - acc: 0.9697 - val_loss: 0.1714 - val_acc: 0.9375
    Epoch 67/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0801 - acc: 0.9707Epoch 1/100
      1/163 [..............................] - ETA: 1:01 - loss: 0.1510 - acc: 0.9375
    Epoch 00067: saving model to /content/data/model/weights.epoch_67.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.0804 - acc: 0.9705 - val_loss: 0.1510 - val_acc: 0.9375
    Epoch 68/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0768 - acc: 0.9734Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1502 - acc: 0.9375
    Epoch 00068: saving model to /content/data/model/weights.epoch_68.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0768 - acc: 0.9732 - val_loss: 0.1502 - val_acc: 0.9375
    Epoch 69/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0859 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1616 - acc: 1.0000
    Epoch 00069: saving model to /content/data/model/weights.epoch_69.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0855 - acc: 0.9684 - val_loss: 0.1616 - val_acc: 1.0000
    Epoch 70/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0780 - acc: 0.9709Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1661 - acc: 1.0000
    Epoch 00070: saving model to /content/data/model/weights.epoch_70.hdf5
    163/163 [==============================] - 89s 544ms/step - loss: 0.0779 - acc: 0.9709 - val_loss: 0.1661 - val_acc: 1.0000
    Epoch 71/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0783 - acc: 0.9718Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1486 - acc: 1.0000
    Epoch 00071: saving model to /content/data/model/weights.epoch_71.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0803 - acc: 0.9712 - val_loss: 0.1486 - val_acc: 1.0000
    Epoch 72/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0803 - acc: 0.9707Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1702 - acc: 0.8750
    Epoch 00072: saving model to /content/data/model/weights.epoch_72.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0802 - acc: 0.9707 - val_loss: 0.1702 - val_acc: 0.8750
    Epoch 73/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0807 - acc: 0.9709Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1317 - acc: 0.9375
    Epoch 00073: saving model to /content/data/model/weights.epoch_73.hdf5
    163/163 [==============================] - 89s 543ms/step - loss: 0.0804 - acc: 0.9711 - val_loss: 0.1317 - val_acc: 0.9375
    Epoch 74/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0781 - acc: 0.9726Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1768 - acc: 0.8750
    Epoch 00074: saving model to /content/data/model/weights.epoch_74.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.0782 - acc: 0.9724 - val_loss: 0.1768 - val_acc: 0.8750
    Epoch 75/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0805 - acc: 0.9711Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1339 - acc: 0.9375
    Epoch 00075: saving model to /content/data/model/weights.epoch_75.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0810 - acc: 0.9709 - val_loss: 0.1339 - val_acc: 0.9375
    Epoch 76/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0808 - acc: 0.9684Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1425 - acc: 0.9375
    Epoch 00076: saving model to /content/data/model/weights.epoch_76.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.0821 - acc: 0.9680 - val_loss: 0.1425 - val_acc: 0.9375
    Epoch 77/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0756 - acc: 0.9732Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1535 - acc: 1.0000
    Epoch 00077: saving model to /content/data/model/weights.epoch_77.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.0758 - acc: 0.9728 - val_loss: 0.1535 - val_acc: 1.0000
    Epoch 78/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0798 - acc: 0.9715Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.1461 - acc: 0.9375
    Epoch 00078: saving model to /content/data/model/weights.epoch_78.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0796 - acc: 0.9716 - val_loss: 0.1461 - val_acc: 0.9375
    Epoch 79/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0758 - acc: 0.9705Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1623 - acc: 1.0000
    Epoch 00079: saving model to /content/data/model/weights.epoch_79.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0760 - acc: 0.9705 - val_loss: 0.1623 - val_acc: 1.0000
    Epoch 80/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0751 - acc: 0.9726Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1533 - acc: 0.9375
    Epoch 00080: saving model to /content/data/model/weights.epoch_80.hdf5
    163/163 [==============================] - 87s 537ms/step - loss: 0.0749 - acc: 0.9728 - val_loss: 0.1533 - val_acc: 0.9375
    Epoch 81/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0799 - acc: 0.9707Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1477 - acc: 1.0000
    Epoch 00081: saving model to /content/data/model/weights.epoch_81.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0796 - acc: 0.9709 - val_loss: 0.1477 - val_acc: 1.0000
    Epoch 82/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0755 - acc: 0.9747Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1323 - acc: 0.9375
    Epoch 00082: saving model to /content/data/model/weights.epoch_82.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0753 - acc: 0.9747 - val_loss: 0.1323 - val_acc: 0.9375
    Epoch 83/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0730 - acc: 0.9728Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1315 - acc: 0.9375
    Epoch 00083: saving model to /content/data/model/weights.epoch_83.hdf5
    163/163 [==============================] - 88s 537ms/step - loss: 0.0728 - acc: 0.9730 - val_loss: 0.1315 - val_acc: 0.9375
    Epoch 84/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0695 - acc: 0.9732Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1608 - acc: 0.9375
    Epoch 00084: saving model to /content/data/model/weights.epoch_84.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0695 - acc: 0.9732 - val_loss: 0.1608 - val_acc: 0.9375
    Epoch 85/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0805 - acc: 0.9703Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1449 - acc: 0.9375
    Epoch 00085: saving model to /content/data/model/weights.epoch_85.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0808 - acc: 0.9701 - val_loss: 0.1449 - val_acc: 0.9375
    Epoch 86/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0748 - acc: 0.9736Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1439 - acc: 0.9375
    Epoch 00086: saving model to /content/data/model/weights.epoch_86.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0747 - acc: 0.9735 - val_loss: 0.1439 - val_acc: 0.9375
    Epoch 87/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0728 - acc: 0.9718Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 0.1476 - acc: 1.0000
    Epoch 00087: saving model to /content/data/model/weights.epoch_87.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0727 - acc: 0.9718 - val_loss: 0.1476 - val_acc: 1.0000
    Epoch 88/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0773 - acc: 0.9722Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1540 - acc: 1.0000
    Epoch 00088: saving model to /content/data/model/weights.epoch_88.hdf5
    163/163 [==============================] - 84s 516ms/step - loss: 0.0774 - acc: 0.9722 - val_loss: 0.1540 - val_acc: 1.0000
    Epoch 89/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0775 - acc: 0.9711Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 0.1274 - acc: 0.9375
    Epoch 00089: saving model to /content/data/model/weights.epoch_89.hdf5
    163/163 [==============================] - 84s 513ms/step - loss: 0.0776 - acc: 0.9709 - val_loss: 0.1274 - val_acc: 0.9375
    Epoch 90/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0744 - acc: 0.9720Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 0.1485 - acc: 1.0000
    Epoch 00090: saving model to /content/data/model/weights.epoch_90.hdf5
    163/163 [==============================] - 83s 509ms/step - loss: 0.0748 - acc: 0.9720 - val_loss: 0.1485 - val_acc: 1.0000
    Epoch 91/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0737 - acc: 0.9757Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 0.1356 - acc: 0.9375
    Epoch 00091: saving model to /content/data/model/weights.epoch_91.hdf5
    163/163 [==============================] - 83s 511ms/step - loss: 0.0738 - acc: 0.9755 - val_loss: 0.1356 - val_acc: 0.9375
    Epoch 92/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0773 - acc: 0.9724Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1256 - acc: 0.9375
    Epoch 00092: saving model to /content/data/model/weights.epoch_92.hdf5
    163/163 [==============================] - 83s 507ms/step - loss: 0.0771 - acc: 0.9726 - val_loss: 0.1256 - val_acc: 0.9375
    Epoch 93/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0770 - acc: 0.9728Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1291 - acc: 0.9375
    Epoch 00093: saving model to /content/data/model/weights.epoch_93.hdf5
    163/163 [==============================] - 82s 501ms/step - loss: 0.0769 - acc: 0.9728 - val_loss: 0.1291 - val_acc: 0.9375
    Epoch 94/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0738 - acc: 0.9699Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.1270 - acc: 0.9375
    Epoch 00094: saving model to /content/data/model/weights.epoch_94.hdf5
    163/163 [==============================] - 83s 507ms/step - loss: 0.0737 - acc: 0.9699 - val_loss: 0.1270 - val_acc: 0.9375
    Epoch 95/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0780 - acc: 0.9713Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.1212 - acc: 0.9375
    Epoch 00095: saving model to /content/data/model/weights.epoch_95.hdf5
    163/163 [==============================] - 83s 507ms/step - loss: 0.0783 - acc: 0.9709 - val_loss: 0.1212 - val_acc: 0.9375
    Epoch 96/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0724 - acc: 0.9745Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1220 - acc: 0.9375
    Epoch 00096: saving model to /content/data/model/weights.epoch_96.hdf5
    163/163 [==============================] - 84s 516ms/step - loss: 0.0726 - acc: 0.9745 - val_loss: 0.1220 - val_acc: 0.9375
    Epoch 97/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0736 - acc: 0.9720Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1406 - acc: 1.0000
    Epoch 00097: saving model to /content/data/model/weights.epoch_97.hdf5
    163/163 [==============================] - 84s 514ms/step - loss: 0.0736 - acc: 0.9720 - val_loss: 0.1406 - val_acc: 1.0000
    Epoch 98/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0688 - acc: 0.9763Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1175 - acc: 0.9375
    Epoch 00098: saving model to /content/data/model/weights.epoch_98.hdf5
    163/163 [==============================] - 85s 520ms/step - loss: 0.0688 - acc: 0.9762 - val_loss: 0.1175 - val_acc: 0.9375
    Epoch 99/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0705 - acc: 0.9724Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1362 - acc: 1.0000
    Epoch 00099: saving model to /content/data/model/weights.epoch_99.hdf5
    163/163 [==============================] - 86s 525ms/step - loss: 0.0705 - acc: 0.9724 - val_loss: 0.1362 - val_acc: 1.0000
    Epoch 100/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0725 - acc: 0.9738Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 0.1209 - acc: 0.9375
    Epoch 00100: saving model to /content/data/model/weights.epoch_100.hdf5
    163/163 [==============================] - 85s 523ms/step - loss: 0.0722 - acc: 0.9739 - val_loss: 0.1209 - val_acc: 0.9375
    


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


![png](VGG16%20Model%201%20Version%201.1.1.0.0_files/VGG16%20Model%201%20Version%201.1.1.0.0_5_0.png)



```
idx = np.argmin(history.history['val_loss']) 
model.load_weights("/content/data/model/weights.epoch_{:02d}.hdf5".format(idx + 1))

print("Loading the best model")
print("epoch: {}, val_loss: {}, val_acc: {}".format(idx + 1, history.history['val_loss'][idx], history.history['val_acc'][idx]))
```

    Loading the best model
    epoch: 98, val_loss: 0.11746180802583694, val_acc: 0.9375
    


```
test_loss, test_acc = model.evaluate_generator(generator=test_generator, verbose=1)
```

    20/20 [==============================] - 6s 318ms/step - loss: 0.2124 - acc: 0.9263
    


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

    20/20 [==============================] - 7s 344ms/step
    CONFUSION MATRIX ------------------
    [[204  30]
     [ 16 374]]
    
    TEST METRICS ----------------------
    Accuracy: 92.62820512820514%
    Precision: 92.57425742574257%
    Recall: 95.8974358974359%
    F1-score: 94.20654911838791
    
    TRAIN METRIC ----------------------
    Train acc: 97.39263653755188%
    


```
from google.colab import drive
drive.mount('/content/drive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive
    


```
!zip -r /content/data/model.zip /content/data/model
```

    updating: content/data/model/ (stored 0%)
    updating: content/data/model/weights.epoch_06.hdf5 (deflated 8%)
    updating: content/data/model/weights.epoch_57.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_36.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_42.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_23.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_98.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_80.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_77.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_20.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_44.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_81.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_46.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_26.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_100.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_97.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_51.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_93.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_68.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_34.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_54.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_59.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_83.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_94.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_66.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_14.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_25.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_61.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_58.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_12.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_27.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_72.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_29.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_55.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_89.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_63.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_37.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_10.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_22.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_49.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_41.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_53.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_48.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_73.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_62.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_18.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_78.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_32.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_71.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_67.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_92.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_19.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_11.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_13.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_70.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_28.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_15.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_24.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_50.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_47.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_84.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_86.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_75.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_21.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_87.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_04.hdf5 (deflated 8%)
    updating: content/data/model/weights.epoch_40.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_56.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_52.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_43.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_02.hdf5 (deflated 8%)
    updating: content/data/model/weights.epoch_45.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_69.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_05.hdf5 (deflated 8%)
    updating: content/data/model/weights.epoch_30.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_33.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_16.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_74.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_31.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_38.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_88.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_09.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_60.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_08.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_85.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_17.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_96.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_07.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_01.hdf5 (deflated 8%)
    updating: content/data/model/weights.epoch_76.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_35.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_99.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_64.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_90.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_79.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_82.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_39.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_95.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_91.hdf5 (deflated 9%)
    updating: content/data/model/weights.epoch_03.hdf5 (deflated 8%)
    updating: content/data/model/weights.epoch_65.hdf5 (deflated 9%)
    


    ---------------------------------------------------------------------------

    MessageError                              Traceback (most recent call last)

    <ipython-input-18-f6da413c0a17> in <module>()
          2 
          3 from google.colab import files
    ----> 4 files.download("/content/data/model.zip")
    

    /usr/local/lib/python3.6/dist-packages/google/colab/files.py in download(filename)
        176       'port': port,
        177       'path': _os.path.abspath(filename),
    --> 178       'name': _os.path.basename(filename),
        179   })
    

    /usr/local/lib/python3.6/dist-packages/google/colab/output/_js.py in eval_js(script, ignore_result)
         37   if ignore_result:
         38     return
    ---> 39   return _message.read_reply_from_input(request_id)
         40 
         41 
    

    /usr/local/lib/python3.6/dist-packages/google/colab/_message.py in read_reply_from_input(message_id, timeout_sec)
        104         reply.get('colab_msg_id') == message_id):
        105       if 'error' in reply:
    --> 106         raise MessageError(reply['error'])
        107       return reply.get('data', None)
        108 
    

    MessageError: TypeError: Failed to fetch

