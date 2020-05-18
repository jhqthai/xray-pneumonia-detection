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

    --2019-10-27 05:14:25--  https://s3.eu-central-1.amazonaws.com/public.unit8.co/data/chest_xray.tar.gz
    Resolving s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)... 52.219.72.4
    Connecting to s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)|52.219.72.4|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1225393795 (1.1G) [application/x-gzip]
    Saving to: ‘/content/data/chest_xray.tar.gz’
    
    /content/data/chest 100%[===================>]   1.14G  23.4MB/s    in 51s     
    
    2019-10-27 05:15:17 (22.8 MB/s) - ‘/content/data/chest_xray.tar.gz’ saved [1225393795/1225393795]
    
    

Change log:
> training_datagen --> ImageDataGenerator

> trainable layer --> All except base

> 24 layers VGG16 model

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
    162/163 [============================>.] - ETA: 0s - loss: 0.4086 - acc: 0.8316Epoch 1/100
      1/163 [..............................] - ETA: 4:26 - loss: 0.3991 - acc: 0.8125
    Epoch 00001: saving model to /content/data/model/weights.epoch_01.hdf5
    163/163 [==============================] - 92s 563ms/step - loss: 0.4070 - acc: 0.8322 - val_loss: 0.3991 - val_acc: 0.8125
    Epoch 2/100
    162/163 [============================>.] - ETA: 0s - loss: 0.2507 - acc: 0.9271Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.3287 - acc: 0.8125
    Epoch 00002: saving model to /content/data/model/weights.epoch_02.hdf5
    163/163 [==============================] - 85s 524ms/step - loss: 0.2505 - acc: 0.9271 - val_loss: 0.3287 - val_acc: 0.8125
    Epoch 3/100
    162/163 [============================>.] - ETA: 0s - loss: 0.2063 - acc: 0.9373Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2651 - acc: 0.8750
    Epoch 00003: saving model to /content/data/model/weights.epoch_03.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.2058 - acc: 0.9375 - val_loss: 0.2651 - val_acc: 0.8750
    Epoch 4/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1740 - acc: 0.9460Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.3186 - acc: 0.8125
    Epoch 00004: saving model to /content/data/model/weights.epoch_04.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.1734 - acc: 0.9463 - val_loss: 0.3186 - val_acc: 0.8125
    Epoch 5/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1643 - acc: 0.9431Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.3525 - acc: 0.8125
    Epoch 00005: saving model to /content/data/model/weights.epoch_05.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.1642 - acc: 0.9429 - val_loss: 0.3525 - val_acc: 0.8125
    Epoch 6/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1421 - acc: 0.9498Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.3826 - acc: 0.8125
    Epoch 00006: saving model to /content/data/model/weights.epoch_06.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.1426 - acc: 0.9496 - val_loss: 0.3826 - val_acc: 0.8125
    Epoch 7/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1485 - acc: 0.9504Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.3499 - acc: 0.8125
    Epoch 00007: saving model to /content/data/model/weights.epoch_07.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.1478 - acc: 0.9507 - val_loss: 0.3499 - val_acc: 0.8125
    Epoch 8/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1403 - acc: 0.9510Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.3228 - acc: 0.8750
    Epoch 00008: saving model to /content/data/model/weights.epoch_08.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.1412 - acc: 0.9507 - val_loss: 0.3228 - val_acc: 0.8750
    Epoch 9/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1461 - acc: 0.9473Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2416 - acc: 0.8125
    Epoch 00009: saving model to /content/data/model/weights.epoch_09.hdf5
    163/163 [==============================] - 88s 537ms/step - loss: 0.1460 - acc: 0.9473 - val_loss: 0.2416 - val_acc: 0.8125
    Epoch 10/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1290 - acc: 0.9527Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2749 - acc: 0.8750
    Epoch 00010: saving model to /content/data/model/weights.epoch_10.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.1288 - acc: 0.9528 - val_loss: 0.2749 - val_acc: 0.8750
    Epoch 11/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1317 - acc: 0.9549Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.4315 - acc: 0.7500
    Epoch 00011: saving model to /content/data/model/weights.epoch_11.hdf5
    163/163 [==============================] - 89s 547ms/step - loss: 0.1318 - acc: 0.9548 - val_loss: 0.4315 - val_acc: 0.7500
    Epoch 12/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1192 - acc: 0.9581Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.3953 - acc: 0.8125
    Epoch 00012: saving model to /content/data/model/weights.epoch_12.hdf5
    163/163 [==============================] - 89s 547ms/step - loss: 0.1189 - acc: 0.9582 - val_loss: 0.3953 - val_acc: 0.8125
    Epoch 13/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1210 - acc: 0.9576Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2080 - acc: 0.8750
    Epoch 00013: saving model to /content/data/model/weights.epoch_13.hdf5
    163/163 [==============================] - 89s 545ms/step - loss: 0.1209 - acc: 0.9576 - val_loss: 0.2080 - val_acc: 0.8750
    Epoch 14/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1239 - acc: 0.9545Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2700 - acc: 0.8750
    Epoch 00014: saving model to /content/data/model/weights.epoch_14.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.1240 - acc: 0.9546 - val_loss: 0.2700 - val_acc: 0.8750
    Epoch 15/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1191 - acc: 0.9587Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.4464 - acc: 0.7500
    Epoch 00015: saving model to /content/data/model/weights.epoch_15.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.1192 - acc: 0.9586 - val_loss: 0.4464 - val_acc: 0.7500
    Epoch 16/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1164 - acc: 0.9591Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.3760 - acc: 0.8125
    Epoch 00016: saving model to /content/data/model/weights.epoch_16.hdf5
    163/163 [==============================] - 85s 524ms/step - loss: 0.1174 - acc: 0.9588 - val_loss: 0.3760 - val_acc: 0.8125
    Epoch 17/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1126 - acc: 0.9610Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.3419 - acc: 0.8750
    Epoch 00017: saving model to /content/data/model/weights.epoch_17.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.1133 - acc: 0.9605 - val_loss: 0.3419 - val_acc: 0.8750
    Epoch 18/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1161 - acc: 0.9585Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.3383 - acc: 0.8750
    Epoch 00018: saving model to /content/data/model/weights.epoch_18.hdf5
    163/163 [==============================] - 88s 537ms/step - loss: 0.1158 - acc: 0.9588 - val_loss: 0.3383 - val_acc: 0.8750
    Epoch 19/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1165 - acc: 0.9595Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2217 - acc: 0.8750
    Epoch 00019: saving model to /content/data/model/weights.epoch_19.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.1163 - acc: 0.9595 - val_loss: 0.2217 - val_acc: 0.8750
    Epoch 20/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1012 - acc: 0.9633Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2163 - acc: 0.8750
    Epoch 00020: saving model to /content/data/model/weights.epoch_20.hdf5
    163/163 [==============================] - 86s 525ms/step - loss: 0.1023 - acc: 0.9630 - val_loss: 0.2163 - val_acc: 0.8750
    Epoch 21/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1058 - acc: 0.9603Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1534 - acc: 0.9375
    Epoch 00021: saving model to /content/data/model/weights.epoch_21.hdf5
    163/163 [==============================] - 86s 525ms/step - loss: 0.1061 - acc: 0.9599 - val_loss: 0.1534 - val_acc: 0.9375
    Epoch 22/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1041 - acc: 0.9628Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1989 - acc: 0.8750
    Epoch 00022: saving model to /content/data/model/weights.epoch_22.hdf5
    163/163 [==============================] - 88s 537ms/step - loss: 0.1038 - acc: 0.9628 - val_loss: 0.1989 - val_acc: 0.8750
    Epoch 23/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1014 - acc: 0.9633Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1429 - acc: 0.9375
    Epoch 00023: saving model to /content/data/model/weights.epoch_23.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.1013 - acc: 0.9632 - val_loss: 0.1429 - val_acc: 0.9375
    Epoch 24/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1144 - acc: 0.9608Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2555 - acc: 0.8750
    Epoch 00024: saving model to /content/data/model/weights.epoch_24.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.1139 - acc: 0.9611 - val_loss: 0.2555 - val_acc: 0.8750
    Epoch 25/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1158 - acc: 0.9587Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.3426 - acc: 0.8125
    Epoch 00025: saving model to /content/data/model/weights.epoch_25.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.1154 - acc: 0.9588 - val_loss: 0.3426 - val_acc: 0.8125
    Epoch 26/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1019 - acc: 0.9657Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2778 - acc: 0.8750
    Epoch 00026: saving model to /content/data/model/weights.epoch_26.hdf5
    163/163 [==============================] - 89s 543ms/step - loss: 0.1030 - acc: 0.9657 - val_loss: 0.2778 - val_acc: 0.8750
    Epoch 27/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1105 - acc: 0.9605Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.3853 - acc: 0.8750
    Epoch 00027: saving model to /content/data/model/weights.epoch_27.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.1099 - acc: 0.9607 - val_loss: 0.3853 - val_acc: 0.8750
    Epoch 28/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1103 - acc: 0.9610Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.1571 - acc: 0.8750
    Epoch 00028: saving model to /content/data/model/weights.epoch_28.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.1098 - acc: 0.9613 - val_loss: 0.1571 - val_acc: 0.8750
    Epoch 29/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1073 - acc: 0.9653Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2325 - acc: 0.8750
    Epoch 00029: saving model to /content/data/model/weights.epoch_29.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.1070 - acc: 0.9655 - val_loss: 0.2325 - val_acc: 0.8750
    Epoch 30/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0944 - acc: 0.9670Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1494 - acc: 0.9375
    Epoch 00030: saving model to /content/data/model/weights.epoch_30.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.0947 - acc: 0.9666 - val_loss: 0.1494 - val_acc: 0.9375
    Epoch 31/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0893 - acc: 0.9678Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1171 - acc: 0.9375
    Epoch 00031: saving model to /content/data/model/weights.epoch_31.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.0893 - acc: 0.9678 - val_loss: 0.1171 - val_acc: 0.9375
    Epoch 32/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0974 - acc: 0.9632Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1659 - acc: 0.8750
    Epoch 00032: saving model to /content/data/model/weights.epoch_32.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0970 - acc: 0.9634 - val_loss: 0.1659 - val_acc: 0.8750
    Epoch 33/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0941 - acc: 0.9668Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1246 - acc: 1.0000
    Epoch 00033: saving model to /content/data/model/weights.epoch_33.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0942 - acc: 0.9668 - val_loss: 0.1246 - val_acc: 1.0000
    Epoch 34/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0914 - acc: 0.9676Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.3142 - acc: 0.8750
    Epoch 00034: saving model to /content/data/model/weights.epoch_34.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0911 - acc: 0.9678 - val_loss: 0.3142 - val_acc: 0.8750
    Epoch 35/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0960 - acc: 0.9653Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1748 - acc: 0.8750
    Epoch 00035: saving model to /content/data/model/weights.epoch_35.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.0964 - acc: 0.9651 - val_loss: 0.1748 - val_acc: 0.8750
    Epoch 36/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1011 - acc: 0.9643Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2288 - acc: 0.8125
    Epoch 00036: saving model to /content/data/model/weights.epoch_36.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.1021 - acc: 0.9641 - val_loss: 0.2288 - val_acc: 0.8125
    Epoch 37/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0920 - acc: 0.9699Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1580 - acc: 0.8750
    Epoch 00037: saving model to /content/data/model/weights.epoch_37.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0934 - acc: 0.9695 - val_loss: 0.1580 - val_acc: 0.8750
    Epoch 38/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0952 - acc: 0.9674Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2501 - acc: 0.8750
    Epoch 00038: saving model to /content/data/model/weights.epoch_38.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0961 - acc: 0.9670 - val_loss: 0.2501 - val_acc: 0.8750
    Epoch 39/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1001 - acc: 0.9659Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1291 - acc: 0.9375
    Epoch 00039: saving model to /content/data/model/weights.epoch_39.hdf5
    163/163 [==============================] - 86s 531ms/step - loss: 0.1008 - acc: 0.9657 - val_loss: 0.1291 - val_acc: 0.9375
    Epoch 40/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1015 - acc: 0.9637Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2680 - acc: 0.8750
    Epoch 00040: saving model to /content/data/model/weights.epoch_40.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.1012 - acc: 0.9638 - val_loss: 0.2680 - val_acc: 0.8750
    Epoch 41/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0939 - acc: 0.9655Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1955 - acc: 0.8750
    Epoch 00041: saving model to /content/data/model/weights.epoch_41.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0937 - acc: 0.9657 - val_loss: 0.1955 - val_acc: 0.8750
    Epoch 42/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0870 - acc: 0.9693Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.3026 - acc: 0.8125
    Epoch 00042: saving model to /content/data/model/weights.epoch_42.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0874 - acc: 0.9689 - val_loss: 0.3026 - val_acc: 0.8125
    Epoch 43/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0911 - acc: 0.9668Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1595 - acc: 0.8750
    Epoch 00043: saving model to /content/data/model/weights.epoch_43.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0917 - acc: 0.9664 - val_loss: 0.1595 - val_acc: 0.8750
    Epoch 44/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0959 - acc: 0.9674Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.3788 - acc: 0.7500
    Epoch 00044: saving model to /content/data/model/weights.epoch_44.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0955 - acc: 0.9676 - val_loss: 0.3788 - val_acc: 0.7500
    Epoch 45/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0936 - acc: 0.9676Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1382 - acc: 0.9375
    Epoch 00045: saving model to /content/data/model/weights.epoch_45.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0938 - acc: 0.9674 - val_loss: 0.1382 - val_acc: 0.9375
    Epoch 46/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0894 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1350 - acc: 0.9375
    Epoch 00046: saving model to /content/data/model/weights.epoch_46.hdf5
    163/163 [==============================] - 87s 537ms/step - loss: 0.0894 - acc: 0.9688 - val_loss: 0.1350 - val_acc: 0.9375
    Epoch 47/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1030 - acc: 0.9630Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1182 - acc: 1.0000
    Epoch 00047: saving model to /content/data/model/weights.epoch_47.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.1034 - acc: 0.9630 - val_loss: 0.1182 - val_acc: 1.0000
    Epoch 48/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0949 - acc: 0.9678Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1833 - acc: 0.8750
    Epoch 00048: saving model to /content/data/model/weights.epoch_48.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0950 - acc: 0.9678 - val_loss: 0.1833 - val_acc: 0.8750
    Epoch 49/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0936 - acc: 0.9659Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1593 - acc: 0.8750
    Epoch 00049: saving model to /content/data/model/weights.epoch_49.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0931 - acc: 0.9661 - val_loss: 0.1593 - val_acc: 0.8750
    Epoch 50/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0848 - acc: 0.9680Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1079 - acc: 0.9375
    Epoch 00050: saving model to /content/data/model/weights.epoch_50.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.0846 - acc: 0.9682 - val_loss: 0.1079 - val_acc: 0.9375
    Epoch 51/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0881 - acc: 0.9697Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.1095 - acc: 1.0000
    Epoch 00051: saving model to /content/data/model/weights.epoch_51.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0883 - acc: 0.9695 - val_loss: 0.1095 - val_acc: 1.0000
    Epoch 52/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0947 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.1149 - acc: 1.0000
    Epoch 00052: saving model to /content/data/model/weights.epoch_52.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0944 - acc: 0.9688 - val_loss: 0.1149 - val_acc: 1.0000
    Epoch 53/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0895 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.2385 - acc: 0.8125
    Epoch 00053: saving model to /content/data/model/weights.epoch_53.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0895 - acc: 0.9701 - val_loss: 0.2385 - val_acc: 0.8125
    Epoch 54/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0848 - acc: 0.9703Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.6372 - acc: 0.7500
    Epoch 00054: saving model to /content/data/model/weights.epoch_54.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0870 - acc: 0.9699 - val_loss: 0.6372 - val_acc: 0.7500
    Epoch 55/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0879 - acc: 0.9689Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1005 - acc: 1.0000
    Epoch 00055: saving model to /content/data/model/weights.epoch_55.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0890 - acc: 0.9686 - val_loss: 0.1005 - val_acc: 1.0000
    Epoch 56/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0884 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1344 - acc: 0.8750
    Epoch 00056: saving model to /content/data/model/weights.epoch_56.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.0884 - acc: 0.9688 - val_loss: 0.1344 - val_acc: 0.8750
    Epoch 57/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0875 - acc: 0.9697Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2307 - acc: 0.8750
    Epoch 00057: saving model to /content/data/model/weights.epoch_57.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0883 - acc: 0.9693 - val_loss: 0.2307 - val_acc: 0.8750
    Epoch 58/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0849 - acc: 0.9703Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1128 - acc: 1.0000
    Epoch 00058: saving model to /content/data/model/weights.epoch_58.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0850 - acc: 0.9703 - val_loss: 0.1128 - val_acc: 1.0000
    Epoch 59/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0823 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0822 - acc: 1.0000
    Epoch 00059: saving model to /content/data/model/weights.epoch_59.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.0820 - acc: 0.9703 - val_loss: 0.0822 - val_acc: 1.0000
    Epoch 60/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0902 - acc: 0.9664Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2329 - acc: 0.8750
    Epoch 00060: saving model to /content/data/model/weights.epoch_60.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0898 - acc: 0.9666 - val_loss: 0.2329 - val_acc: 0.8750
    Epoch 61/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0820 - acc: 0.9711Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0746 - acc: 1.0000
    Epoch 00061: saving model to /content/data/model/weights.epoch_61.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.0826 - acc: 0.9707 - val_loss: 0.0746 - val_acc: 1.0000
    Epoch 62/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0909 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1717 - acc: 0.8750
    Epoch 00062: saving model to /content/data/model/weights.epoch_62.hdf5
    163/163 [==============================] - 87s 537ms/step - loss: 0.0908 - acc: 0.9688 - val_loss: 0.1717 - val_acc: 0.8750
    Epoch 63/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0859 - acc: 0.9697Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1260 - acc: 1.0000
    Epoch 00063: saving model to /content/data/model/weights.epoch_63.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0862 - acc: 0.9693 - val_loss: 0.1260 - val_acc: 1.0000
    Epoch 64/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0954 - acc: 0.9676Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0732 - acc: 1.0000
    Epoch 00064: saving model to /content/data/model/weights.epoch_64.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.0957 - acc: 0.9676 - val_loss: 0.0732 - val_acc: 1.0000
    Epoch 65/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0829 - acc: 0.9718Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1568 - acc: 0.8750
    Epoch 00065: saving model to /content/data/model/weights.epoch_65.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0827 - acc: 0.9720 - val_loss: 0.1568 - val_acc: 0.8750
    Epoch 66/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0852 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.0780 - acc: 1.0000
    Epoch 00066: saving model to /content/data/model/weights.epoch_66.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0852 - acc: 0.9699 - val_loss: 0.0780 - val_acc: 1.0000
    Epoch 67/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0906 - acc: 0.9668Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.0974 - acc: 0.9375
    Epoch 00067: saving model to /content/data/model/weights.epoch_67.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0905 - acc: 0.9668 - val_loss: 0.0974 - val_acc: 0.9375
    Epoch 68/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0875 - acc: 0.9662Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0688 - acc: 1.0000
    Epoch 00068: saving model to /content/data/model/weights.epoch_68.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.0873 - acc: 0.9664 - val_loss: 0.0688 - val_acc: 1.0000
    Epoch 69/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0850 - acc: 0.9697Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2010 - acc: 0.8750
    Epoch 00069: saving model to /content/data/model/weights.epoch_69.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0849 - acc: 0.9697 - val_loss: 0.2010 - val_acc: 0.8750
    Epoch 70/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0818 - acc: 0.9699Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0961 - acc: 1.0000
    Epoch 00070: saving model to /content/data/model/weights.epoch_70.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0819 - acc: 0.9699 - val_loss: 0.0961 - val_acc: 1.0000
    Epoch 71/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0798 - acc: 0.9724Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1345 - acc: 0.8750
    Epoch 00071: saving model to /content/data/model/weights.epoch_71.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0795 - acc: 0.9726 - val_loss: 0.1345 - val_acc: 0.8750
    Epoch 72/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0745 - acc: 0.9722Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0583 - acc: 1.0000
    Epoch 00072: saving model to /content/data/model/weights.epoch_72.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0744 - acc: 0.9722 - val_loss: 0.0583 - val_acc: 1.0000
    Epoch 73/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0826 - acc: 0.9715Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.2599 - acc: 0.8125
    Epoch 00073: saving model to /content/data/model/weights.epoch_73.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0825 - acc: 0.9714 - val_loss: 0.2599 - val_acc: 0.8125
    Epoch 74/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0801 - acc: 0.9726Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.0894 - acc: 1.0000
    Epoch 00074: saving model to /content/data/model/weights.epoch_74.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0800 - acc: 0.9726 - val_loss: 0.0894 - val_acc: 1.0000
    Epoch 75/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0753 - acc: 0.9751Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.1020 - acc: 1.0000
    Epoch 00075: saving model to /content/data/model/weights.epoch_75.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0752 - acc: 0.9753 - val_loss: 0.1020 - val_acc: 1.0000
    Epoch 76/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0833 - acc: 0.9674Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1947 - acc: 0.8750
    Epoch 00076: saving model to /content/data/model/weights.epoch_76.hdf5
    163/163 [==============================] - 87s 537ms/step - loss: 0.0831 - acc: 0.9674 - val_loss: 0.1947 - val_acc: 0.8750
    Epoch 77/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0694 - acc: 0.9761Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1262 - acc: 0.9375
    Epoch 00077: saving model to /content/data/model/weights.epoch_77.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0701 - acc: 0.9758 - val_loss: 0.1262 - val_acc: 0.9375
    Epoch 78/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0814 - acc: 0.9730Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.0767 - acc: 1.0000
    Epoch 00078: saving model to /content/data/model/weights.epoch_78.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0812 - acc: 0.9730 - val_loss: 0.0767 - val_acc: 1.0000
    Epoch 79/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0819 - acc: 0.9728Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0978 - acc: 1.0000
    Epoch 00079: saving model to /content/data/model/weights.epoch_79.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0816 - acc: 0.9730 - val_loss: 0.0978 - val_acc: 1.0000
    Epoch 80/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0837 - acc: 0.9718Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1157 - acc: 1.0000
    Epoch 00080: saving model to /content/data/model/weights.epoch_80.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0836 - acc: 0.9718 - val_loss: 0.1157 - val_acc: 1.0000
    Epoch 81/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0783 - acc: 0.9728Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.3497 - acc: 0.8125
    Epoch 00081: saving model to /content/data/model/weights.epoch_81.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.0780 - acc: 0.9730 - val_loss: 0.3497 - val_acc: 0.8125
    Epoch 82/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0825 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.0968 - acc: 1.0000
    Epoch 00082: saving model to /content/data/model/weights.epoch_82.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.0822 - acc: 0.9689 - val_loss: 0.0968 - val_acc: 1.0000
    Epoch 83/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0865 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1103 - acc: 0.9375
    Epoch 00083: saving model to /content/data/model/weights.epoch_83.hdf5
    163/163 [==============================] - 89s 547ms/step - loss: 0.0870 - acc: 0.9701 - val_loss: 0.1103 - val_acc: 0.9375
    Epoch 84/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0820 - acc: 0.9730Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0963 - acc: 1.0000
    Epoch 00084: saving model to /content/data/model/weights.epoch_84.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0817 - acc: 0.9732 - val_loss: 0.0963 - val_acc: 1.0000
    Epoch 85/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0771 - acc: 0.9728Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0892 - acc: 1.0000
    Epoch 00085: saving model to /content/data/model/weights.epoch_85.hdf5
    163/163 [==============================] - 89s 545ms/step - loss: 0.0768 - acc: 0.9728 - val_loss: 0.0892 - val_acc: 1.0000
    Epoch 86/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0817 - acc: 0.9716Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1064 - acc: 0.9375
    Epoch 00086: saving model to /content/data/model/weights.epoch_86.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0819 - acc: 0.9716 - val_loss: 0.1064 - val_acc: 0.9375
    Epoch 87/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0824 - acc: 0.9703Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1038 - acc: 1.0000
    Epoch 00087: saving model to /content/data/model/weights.epoch_87.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.0824 - acc: 0.9703 - val_loss: 0.1038 - val_acc: 1.0000
    Epoch 88/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0869 - acc: 0.9716Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0846 - acc: 0.9375
    Epoch 00088: saving model to /content/data/model/weights.epoch_88.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0868 - acc: 0.9716 - val_loss: 0.0846 - val_acc: 0.9375
    Epoch 89/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0782 - acc: 0.9726Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1482 - acc: 0.9375
    Epoch 00089: saving model to /content/data/model/weights.epoch_89.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.0790 - acc: 0.9722 - val_loss: 0.1482 - val_acc: 0.9375
    Epoch 90/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0796 - acc: 0.9713Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.2667 - acc: 0.8125
    Epoch 00090: saving model to /content/data/model/weights.epoch_90.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0799 - acc: 0.9712 - val_loss: 0.2667 - val_acc: 0.8125
    Epoch 91/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0806 - acc: 0.9711Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1422 - acc: 0.9375
    Epoch 00091: saving model to /content/data/model/weights.epoch_91.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.0807 - acc: 0.9711 - val_loss: 0.1422 - val_acc: 0.9375
    Epoch 92/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0730 - acc: 0.9743Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0867 - acc: 1.0000
    Epoch 00092: saving model to /content/data/model/weights.epoch_92.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.0734 - acc: 0.9743 - val_loss: 0.0867 - val_acc: 1.0000
    Epoch 93/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0810 - acc: 0.9693Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0704 - acc: 0.9375
    Epoch 00093: saving model to /content/data/model/weights.epoch_93.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0809 - acc: 0.9693 - val_loss: 0.0704 - val_acc: 0.9375
    Epoch 94/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0775 - acc: 0.9713Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0916 - acc: 1.0000
    Epoch 00094: saving model to /content/data/model/weights.epoch_94.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.0776 - acc: 0.9711 - val_loss: 0.0916 - val_acc: 1.0000
    Epoch 95/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0790 - acc: 0.9747Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0681 - acc: 1.0000
    Epoch 00095: saving model to /content/data/model/weights.epoch_95.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0786 - acc: 0.9749 - val_loss: 0.0681 - val_acc: 1.0000
    Epoch 96/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0731 - acc: 0.9724Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.0682 - acc: 1.0000
    Epoch 00096: saving model to /content/data/model/weights.epoch_96.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0731 - acc: 0.9724 - val_loss: 0.0682 - val_acc: 1.0000
    Epoch 97/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0777 - acc: 0.9720Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0799 - acc: 1.0000
    Epoch 00097: saving model to /content/data/model/weights.epoch_97.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0777 - acc: 0.9718 - val_loss: 0.0799 - val_acc: 1.0000
    Epoch 98/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0769 - acc: 0.9743Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0810 - acc: 1.0000
    Epoch 00098: saving model to /content/data/model/weights.epoch_98.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.0767 - acc: 0.9745 - val_loss: 0.0810 - val_acc: 1.0000
    Epoch 99/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0834 - acc: 0.9672Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1029 - acc: 1.0000
    Epoch 00099: saving model to /content/data/model/weights.epoch_99.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.0831 - acc: 0.9674 - val_loss: 0.1029 - val_acc: 1.0000
    Epoch 100/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0826 - acc: 0.9709Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1546 - acc: 0.9375
    Epoch 00100: saving model to /content/data/model/weights.epoch_100.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.0838 - acc: 0.9705 - val_loss: 0.1546 - val_acc: 0.9375
    


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


![png](VGG16%20Model%202%20Version%202.1.1.0.0_files/VGG16%20Model%202%20Version%202.1.1.0.0_5_0.png)



```
idx = np.argmin(history.history['val_loss']) 
model.load_weights("/content/data/model/weights.epoch_{:02d}.hdf5".format(idx + 1))

print("Loading the best model")
print("epoch: {}, val_loss: {}, val_acc: {}".format(idx + 1, history.history['val_loss'][idx], history.history['val_acc'][idx]))
```

    Loading the best model
    epoch: 72, val_loss: 0.05833631008863449, val_acc: 1.0
    


```
test_loss, test_acc = model.evaluate_generator(generator=test_generator, verbose=1)
```

    20/20 [==============================] - 7s 330ms/step - loss: 0.2280 - acc: 0.9215
    


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

    20/20 [==============================] - 7s 355ms/step
    CONFUSION MATRIX ------------------
    [[203  31]
     [ 18 372]]
    
    TEST METRICS ----------------------
    Accuracy: 92.1474358974359%
    Precision: 92.3076923076923%
    Recall: 95.38461538461539%
    F1-score: 93.82093316519547
    
    TRAIN METRIC ----------------------
    Train acc: 97.0475435256958%
    


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

