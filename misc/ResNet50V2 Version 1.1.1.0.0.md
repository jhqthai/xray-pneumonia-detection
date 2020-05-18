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

    --2019-11-08 04:14:02--  https://s3.eu-central-1.amazonaws.com/public.unit8.co/data/chest_xray.tar.gz
    Resolving s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)... 52.219.72.8
    Connecting to s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)|52.219.72.8|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1225393795 (1.1G) [application/x-gzip]
    Saving to: ‘/content/data/chest_xray.tar.gz’
    
    /content/data/chest 100%[===================>]   1.14G  85.7MB/s    in 14s     
    
    2019-11-08 04:14:17 (83.8 MB/s) - ‘/content/data/chest_xray.tar.gz’ saved [1225393795/1225393795]
    
    

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

base_model = tf.keras.applications.resnet_v2.ResNet50V2(weights='imagenet', include_top=False, input_shape=train_shape)

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
    Downloading data from https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
    94674944/94668760 [==============================] - 3s 0us/step
    [1.9448173  0.67303226]
    Epoch 1/100
    162/163 [============================>.] - ETA: 0s - loss: 0.2824 - acc: 0.8966Epoch 1/100
      1/163 [..............................] - ETA: 6:11 - loss: 0.8304 - acc: 0.8125
    Epoch 00001: saving model to /content/data/model/weights.epoch_01.hdf5
    163/163 [==============================] - 95s 581ms/step - loss: 0.2818 - acc: 0.8967 - val_loss: 0.8304 - val_acc: 0.8125
    Epoch 2/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1982 - acc: 0.9356Epoch 1/100
      1/163 [..............................] - ETA: 50s - loss: 2.0862 - acc: 0.6250
    Epoch 00002: saving model to /content/data/model/weights.epoch_02.hdf5
    163/163 [==============================] - 77s 472ms/step - loss: 0.1973 - acc: 0.9358 - val_loss: 2.0862 - val_acc: 0.6250
    Epoch 3/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1548 - acc: 0.9483Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.9625 - acc: 0.7500
    Epoch 00003: saving model to /content/data/model/weights.epoch_03.hdf5
    163/163 [==============================] - 81s 495ms/step - loss: 0.1557 - acc: 0.9479 - val_loss: 0.9625 - val_acc: 0.7500
    Epoch 4/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1416 - acc: 0.9502Epoch 1/100
      1/163 [..............................] - ETA: 50s - loss: 3.3152 - acc: 0.6250
    Epoch 00004: saving model to /content/data/model/weights.epoch_04.hdf5
    163/163 [==============================] - 81s 494ms/step - loss: 0.1425 - acc: 0.9496 - val_loss: 3.3152 - val_acc: 0.6250
    Epoch 5/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1407 - acc: 0.9539Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 4.5362 - acc: 0.5000
    Epoch 00005: saving model to /content/data/model/weights.epoch_05.hdf5
    163/163 [==============================] - 81s 496ms/step - loss: 0.1399 - acc: 0.9542 - val_loss: 4.5362 - val_acc: 0.5000
    Epoch 6/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1265 - acc: 0.9551Epoch 1/100
      1/163 [..............................] - ETA: 50s - loss: 2.8676 - acc: 0.5625
    Epoch 00006: saving model to /content/data/model/weights.epoch_06.hdf5
    163/163 [==============================] - 80s 494ms/step - loss: 0.1259 - acc: 0.9553 - val_loss: 2.8676 - val_acc: 0.5625
    Epoch 7/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1267 - acc: 0.9576Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 2.9686 - acc: 0.6250
    Epoch 00007: saving model to /content/data/model/weights.epoch_07.hdf5
    163/163 [==============================] - 80s 491ms/step - loss: 0.1261 - acc: 0.9578 - val_loss: 2.9686 - val_acc: 0.6250
    Epoch 8/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1099 - acc: 0.9657Epoch 1/100
      1/163 [..............................] - ETA: 50s - loss: 4.3251 - acc: 0.5625
    Epoch 00008: saving model to /content/data/model/weights.epoch_08.hdf5
    163/163 [==============================] - 80s 493ms/step - loss: 0.1104 - acc: 0.9655 - val_loss: 4.3251 - val_acc: 0.5625
    Epoch 9/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0994 - acc: 0.9660Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 5.9945 - acc: 0.5000
    Epoch 00009: saving model to /content/data/model/weights.epoch_09.hdf5
    163/163 [==============================] - 80s 489ms/step - loss: 0.0999 - acc: 0.9655 - val_loss: 5.9945 - val_acc: 0.5000
    Epoch 10/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1189 - acc: 0.9618Epoch 1/100
      1/163 [..............................] - ETA: 50s - loss: 3.3174 - acc: 0.6250
    Epoch 00010: saving model to /content/data/model/weights.epoch_10.hdf5
    163/163 [==============================] - 80s 491ms/step - loss: 0.1182 - acc: 0.9620 - val_loss: 3.3174 - val_acc: 0.6250
    Epoch 11/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1122 - acc: 0.9651Epoch 1/100
      1/163 [..............................] - ETA: 50s - loss: 6.8161 - acc: 0.5000
    Epoch 00011: saving model to /content/data/model/weights.epoch_11.hdf5
    163/163 [==============================] - 80s 491ms/step - loss: 0.1119 - acc: 0.9651 - val_loss: 6.8161 - val_acc: 0.5000
    Epoch 12/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1041 - acc: 0.9662Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 6.4790 - acc: 0.5000
    Epoch 00012: saving model to /content/data/model/weights.epoch_12.hdf5
    163/163 [==============================] - 80s 488ms/step - loss: 0.1036 - acc: 0.9663 - val_loss: 6.4790 - val_acc: 0.5000
    Epoch 13/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0991 - acc: 0.9678Epoch 1/100
      1/163 [..............................] - ETA: 50s - loss: 6.4378 - acc: 0.5000
    Epoch 00013: saving model to /content/data/model/weights.epoch_13.hdf5
    163/163 [==============================] - 83s 507ms/step - loss: 0.1002 - acc: 0.9674 - val_loss: 6.4378 - val_acc: 0.5000
    Epoch 14/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0935 - acc: 0.9670Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 6.1984 - acc: 0.5000
    Epoch 00014: saving model to /content/data/model/weights.epoch_14.hdf5
    163/163 [==============================] - 83s 511ms/step - loss: 0.0941 - acc: 0.9668 - val_loss: 6.1984 - val_acc: 0.5000
    Epoch 15/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0984 - acc: 0.9686Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 6.0110 - acc: 0.5000
    Epoch 00015: saving model to /content/data/model/weights.epoch_15.hdf5
    163/163 [==============================] - 85s 520ms/step - loss: 0.0978 - acc: 0.9688 - val_loss: 6.0110 - val_acc: 0.5000
    Epoch 16/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1009 - acc: 0.9699Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 6.4342 - acc: 0.5000
    Epoch 00016: saving model to /content/data/model/weights.epoch_16.hdf5
    163/163 [==============================] - 85s 521ms/step - loss: 0.1006 - acc: 0.9699 - val_loss: 6.4342 - val_acc: 0.5000
    Epoch 17/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0859 - acc: 0.9716Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 6.4695 - acc: 0.5000
    Epoch 00017: saving model to /content/data/model/weights.epoch_17.hdf5
    163/163 [==============================] - 84s 516ms/step - loss: 0.0856 - acc: 0.9716 - val_loss: 6.4695 - val_acc: 0.5000
    Epoch 18/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0976 - acc: 0.9693Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 5.3972 - acc: 0.5625
    Epoch 00018: saving model to /content/data/model/weights.epoch_18.hdf5
    163/163 [==============================] - 84s 515ms/step - loss: 0.0971 - acc: 0.9695 - val_loss: 5.3972 - val_acc: 0.5625
    Epoch 19/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0888 - acc: 0.9726Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 4.0055 - acc: 0.5625
    Epoch 00019: saving model to /content/data/model/weights.epoch_19.hdf5
    163/163 [==============================] - 84s 517ms/step - loss: 0.0895 - acc: 0.9722 - val_loss: 4.0055 - val_acc: 0.5625
    Epoch 20/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0988 - acc: 0.9713Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 5.8819 - acc: 0.5625
    Epoch 00020: saving model to /content/data/model/weights.epoch_20.hdf5
    163/163 [==============================] - 85s 519ms/step - loss: 0.0986 - acc: 0.9712 - val_loss: 5.8819 - val_acc: 0.5625
    Epoch 21/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1007 - acc: 0.9705Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 6.4408 - acc: 0.5625
    Epoch 00021: saving model to /content/data/model/weights.epoch_21.hdf5
    163/163 [==============================] - 85s 519ms/step - loss: 0.1003 - acc: 0.9707 - val_loss: 6.4408 - val_acc: 0.5625
    Epoch 22/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0846 - acc: 0.9732Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 6.7749 - acc: 0.5000
    Epoch 00022: saving model to /content/data/model/weights.epoch_22.hdf5
    163/163 [==============================] - 85s 519ms/step - loss: 0.0847 - acc: 0.9732 - val_loss: 6.7749 - val_acc: 0.5000
    Epoch 23/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1020 - acc: 0.9707Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 7.9478 - acc: 0.5000
    Epoch 00023: saving model to /content/data/model/weights.epoch_23.hdf5
    163/163 [==============================] - 85s 518ms/step - loss: 0.1026 - acc: 0.9703 - val_loss: 7.9478 - val_acc: 0.5000
    Epoch 24/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0776 - acc: 0.9740Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 8.0591 - acc: 0.5000
    Epoch 00024: saving model to /content/data/model/weights.epoch_24.hdf5
    163/163 [==============================] - 85s 523ms/step - loss: 0.0772 - acc: 0.9741 - val_loss: 8.0591 - val_acc: 0.5000
    Epoch 25/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0890 - acc: 0.9707Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 6.3816 - acc: 0.5000
    Epoch 00025: saving model to /content/data/model/weights.epoch_25.hdf5
    163/163 [==============================] - 85s 523ms/step - loss: 0.0887 - acc: 0.9707 - val_loss: 6.3816 - val_acc: 0.5000
    Epoch 26/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0863 - acc: 0.9740Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 6.5415 - acc: 0.5000
    Epoch 00026: saving model to /content/data/model/weights.epoch_26.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.0862 - acc: 0.9739 - val_loss: 6.5415 - val_acc: 0.5000
    Epoch 27/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0889 - acc: 0.9726Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 8.1368 - acc: 0.5000
    Epoch 00027: saving model to /content/data/model/weights.epoch_27.hdf5
    163/163 [==============================] - 86s 525ms/step - loss: 0.0883 - acc: 0.9728 - val_loss: 8.1368 - val_acc: 0.5000
    Epoch 28/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1028 - acc: 0.9720Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 8.3632 - acc: 0.5000
    Epoch 00028: saving model to /content/data/model/weights.epoch_28.hdf5
    163/163 [==============================] - 85s 523ms/step - loss: 0.1026 - acc: 0.9720 - val_loss: 8.3632 - val_acc: 0.5000
    Epoch 29/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0850 - acc: 0.9724Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 8.6129 - acc: 0.5000
    Epoch 00029: saving model to /content/data/model/weights.epoch_29.hdf5
    163/163 [==============================] - 86s 525ms/step - loss: 0.0849 - acc: 0.9724 - val_loss: 8.6129 - val_acc: 0.5000
    Epoch 30/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1020 - acc: 0.9738Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 10.9667 - acc: 0.5000
    Epoch 00030: saving model to /content/data/model/weights.epoch_30.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.1014 - acc: 0.9739 - val_loss: 10.9667 - val_acc: 0.5000
    Epoch 31/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0753 - acc: 0.9761Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 7.7622 - acc: 0.5000
    Epoch 00031: saving model to /content/data/model/weights.epoch_31.hdf5
    163/163 [==============================] - 85s 524ms/step - loss: 0.0752 - acc: 0.9760 - val_loss: 7.7622 - val_acc: 0.5000
    Epoch 32/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0852 - acc: 0.9745Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 6.6926 - acc: 0.5000
    Epoch 00032: saving model to /content/data/model/weights.epoch_32.hdf5
    163/163 [==============================] - 85s 521ms/step - loss: 0.0847 - acc: 0.9747 - val_loss: 6.6926 - val_acc: 0.5000
    Epoch 33/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0782 - acc: 0.9769Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 4.1305 - acc: 0.6875
    Epoch 00033: saving model to /content/data/model/weights.epoch_33.hdf5
    163/163 [==============================] - 85s 524ms/step - loss: 0.0778 - acc: 0.9770 - val_loss: 4.1305 - val_acc: 0.6875
    Epoch 34/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0758 - acc: 0.9774Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 6.4277 - acc: 0.5000
    Epoch 00034: saving model to /content/data/model/weights.epoch_34.hdf5
    163/163 [==============================] - 85s 523ms/step - loss: 0.0758 - acc: 0.9772 - val_loss: 6.4277 - val_acc: 0.5000
    Epoch 35/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0752 - acc: 0.9782Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 8.0318 - acc: 0.5000
    Epoch 00035: saving model to /content/data/model/weights.epoch_35.hdf5
    163/163 [==============================] - 85s 524ms/step - loss: 0.0749 - acc: 0.9781 - val_loss: 8.0318 - val_acc: 0.5000
    Epoch 36/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0716 - acc: 0.9767Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 7.8507 - acc: 0.5000
    Epoch 00036: saving model to /content/data/model/weights.epoch_36.hdf5
    163/163 [==============================] - 85s 522ms/step - loss: 0.0712 - acc: 0.9768 - val_loss: 7.8507 - val_acc: 0.5000
    Epoch 37/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0738 - acc: 0.9776Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 7.4273 - acc: 0.5625
    Epoch 00037: saving model to /content/data/model/weights.epoch_37.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.0737 - acc: 0.9776 - val_loss: 7.4273 - val_acc: 0.5625
    Epoch 38/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0796 - acc: 0.9730Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 6.2294 - acc: 0.5625
    Epoch 00038: saving model to /content/data/model/weights.epoch_38.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0796 - acc: 0.9730 - val_loss: 6.2294 - val_acc: 0.5625
    Epoch 39/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0788 - acc: 0.9757Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 6.5632 - acc: 0.5000
    Epoch 00039: saving model to /content/data/model/weights.epoch_39.hdf5
    163/163 [==============================] - 85s 523ms/step - loss: 0.0784 - acc: 0.9758 - val_loss: 6.5632 - val_acc: 0.5000
    Epoch 40/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0843 - acc: 0.9761Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 6.6225 - acc: 0.5625
    Epoch 00040: saving model to /content/data/model/weights.epoch_40.hdf5
    163/163 [==============================] - 86s 525ms/step - loss: 0.0848 - acc: 0.9760 - val_loss: 6.6225 - val_acc: 0.5625
    Epoch 41/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0926 - acc: 0.9711Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 10.5288 - acc: 0.5000
    Epoch 00041: saving model to /content/data/model/weights.epoch_41.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.0921 - acc: 0.9712 - val_loss: 10.5288 - val_acc: 0.5000
    Epoch 42/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0812 - acc: 0.9759Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 7.3827 - acc: 0.5000
    Epoch 00042: saving model to /content/data/model/weights.epoch_42.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0807 - acc: 0.9760 - val_loss: 7.3827 - val_acc: 0.5000
    Epoch 43/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0590 - acc: 0.9792Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 10.0128 - acc: 0.5000
    Epoch 00043: saving model to /content/data/model/weights.epoch_43.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.0587 - acc: 0.9793 - val_loss: 10.0128 - val_acc: 0.5000
    Epoch 44/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0906 - acc: 0.9755Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 10.3990 - acc: 0.5000
    Epoch 00044: saving model to /content/data/model/weights.epoch_44.hdf5
    163/163 [==============================] - 86s 525ms/step - loss: 0.0906 - acc: 0.9755 - val_loss: 10.3990 - val_acc: 0.5000
    Epoch 45/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0788 - acc: 0.9769Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 11.6837 - acc: 0.5000
    Epoch 00045: saving model to /content/data/model/weights.epoch_45.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.0787 - acc: 0.9768 - val_loss: 11.6837 - val_acc: 0.5000
    Epoch 46/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0935 - acc: 0.9738Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 9.3021 - acc: 0.5000
    Epoch 00046: saving model to /content/data/model/weights.epoch_46.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0930 - acc: 0.9739 - val_loss: 9.3021 - val_acc: 0.5000
    Epoch 47/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0700 - acc: 0.9799Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 8.2738 - acc: 0.5000
    Epoch 00047: saving model to /content/data/model/weights.epoch_47.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.0696 - acc: 0.9801 - val_loss: 8.2738 - val_acc: 0.5000
    Epoch 48/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0755 - acc: 0.9778Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 9.4213 - acc: 0.5000
    Epoch 00048: saving model to /content/data/model/weights.epoch_48.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.0751 - acc: 0.9780 - val_loss: 9.4213 - val_acc: 0.5000
    Epoch 49/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0786 - acc: 0.9767Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 7.9777 - acc: 0.5000
    Epoch 00049: saving model to /content/data/model/weights.epoch_49.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.0783 - acc: 0.9766 - val_loss: 7.9777 - val_acc: 0.5000
    Epoch 50/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0655 - acc: 0.9772Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 6.6981 - acc: 0.5000
    Epoch 00050: saving model to /content/data/model/weights.epoch_50.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0668 - acc: 0.9772 - val_loss: 6.6981 - val_acc: 0.5000
    Epoch 51/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0696 - acc: 0.9790Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 7.3452 - acc: 0.5000
    Epoch 00051: saving model to /content/data/model/weights.epoch_51.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0692 - acc: 0.9791 - val_loss: 7.3452 - val_acc: 0.5000
    Epoch 52/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0764 - acc: 0.9780Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 7.5369 - acc: 0.5000
    Epoch 00052: saving model to /content/data/model/weights.epoch_52.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.0763 - acc: 0.9780 - val_loss: 7.5369 - val_acc: 0.5000
    Epoch 53/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0817 - acc: 0.9774Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 5.7383 - acc: 0.5000
    Epoch 00053: saving model to /content/data/model/weights.epoch_53.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0818 - acc: 0.9772 - val_loss: 5.7383 - val_acc: 0.5000
    Epoch 54/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0707 - acc: 0.9815Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 6.5155 - acc: 0.5000
    Epoch 00054: saving model to /content/data/model/weights.epoch_54.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.0708 - acc: 0.9812 - val_loss: 6.5155 - val_acc: 0.5000
    Epoch 55/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0776 - acc: 0.9782Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 7.7717 - acc: 0.5000
    Epoch 00055: saving model to /content/data/model/weights.epoch_55.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.0786 - acc: 0.9780 - val_loss: 7.7717 - val_acc: 0.5000
    Epoch 56/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0676 - acc: 0.9807Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 6.3639 - acc: 0.5000
    Epoch 00056: saving model to /content/data/model/weights.epoch_56.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.0675 - acc: 0.9806 - val_loss: 6.3639 - val_acc: 0.5000
    Epoch 57/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0586 - acc: 0.9834Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 8.9210 - acc: 0.5000
    Epoch 00057: saving model to /content/data/model/weights.epoch_57.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.0586 - acc: 0.9833 - val_loss: 8.9210 - val_acc: 0.5000
    Epoch 58/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0818 - acc: 0.9778Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 5.4594 - acc: 0.6250
    Epoch 00058: saving model to /content/data/model/weights.epoch_58.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.0814 - acc: 0.9778 - val_loss: 5.4594 - val_acc: 0.6250
    Epoch 59/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0711 - acc: 0.9788Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 7.3799 - acc: 0.5000
    Epoch 00059: saving model to /content/data/model/weights.epoch_59.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0713 - acc: 0.9787 - val_loss: 7.3799 - val_acc: 0.5000
    Epoch 60/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0739 - acc: 0.9784Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 5.6847 - acc: 0.5625
    Epoch 00060: saving model to /content/data/model/weights.epoch_60.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0735 - acc: 0.9785 - val_loss: 5.6847 - val_acc: 0.5625
    Epoch 61/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0718 - acc: 0.9776Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 6.9876 - acc: 0.5000
    Epoch 00061: saving model to /content/data/model/weights.epoch_61.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0716 - acc: 0.9776 - val_loss: 6.9876 - val_acc: 0.5000
    Epoch 62/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0822 - acc: 0.9755Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 5.7479 - acc: 0.5625
    Epoch 00062: saving model to /content/data/model/weights.epoch_62.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.0818 - acc: 0.9757 - val_loss: 5.7479 - val_acc: 0.5625
    Epoch 63/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0706 - acc: 0.9782Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 9.4446 - acc: 0.5000
    Epoch 00063: saving model to /content/data/model/weights.epoch_63.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0704 - acc: 0.9781 - val_loss: 9.4446 - val_acc: 0.5000
    Epoch 64/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0735 - acc: 0.9794Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 8.7786 - acc: 0.5000
    Epoch 00064: saving model to /content/data/model/weights.epoch_64.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0731 - acc: 0.9795 - val_loss: 8.7786 - val_acc: 0.5000
    Epoch 65/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0723 - acc: 0.9780Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 9.3511 - acc: 0.5000
    Epoch 00065: saving model to /content/data/model/weights.epoch_65.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.0719 - acc: 0.9781 - val_loss: 9.3511 - val_acc: 0.5000
    Epoch 66/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0813 - acc: 0.9780Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 9.4141 - acc: 0.5000
    Epoch 00066: saving model to /content/data/model/weights.epoch_66.hdf5
    163/163 [==============================] - 86s 525ms/step - loss: 0.0808 - acc: 0.9781 - val_loss: 9.4141 - val_acc: 0.5000
    Epoch 67/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0584 - acc: 0.9809Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 9.0598 - acc: 0.5000
    Epoch 00067: saving model to /content/data/model/weights.epoch_67.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.0593 - acc: 0.9806 - val_loss: 9.0598 - val_acc: 0.5000
    Epoch 68/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0586 - acc: 0.9838Epoch 1/100
      1/163 [..............................] - ETA: 49s - loss: 9.7076 - acc: 0.5000
    Epoch 00068: saving model to /content/data/model/weights.epoch_68.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.0588 - acc: 0.9837 - val_loss: 9.7076 - val_acc: 0.5000
    Epoch 69/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0726 - acc: 0.9778Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 10.0596 - acc: 0.5000
    Epoch 00069: saving model to /content/data/model/weights.epoch_69.hdf5
    163/163 [==============================] - 85s 523ms/step - loss: 0.0723 - acc: 0.9780 - val_loss: 10.0596 - val_acc: 0.5000
    Epoch 70/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0686 - acc: 0.9807Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 7.9896 - acc: 0.5000
    Epoch 00070: saving model to /content/data/model/weights.epoch_70.hdf5
    163/163 [==============================] - 85s 524ms/step - loss: 0.0689 - acc: 0.9804 - val_loss: 7.9896 - val_acc: 0.5000
    Epoch 71/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0736 - acc: 0.9774Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 8.1873 - acc: 0.5000
    Epoch 00071: saving model to /content/data/model/weights.epoch_71.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0755 - acc: 0.9772 - val_loss: 8.1873 - val_acc: 0.5000
    Epoch 72/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0751 - acc: 0.9794Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 10.4079 - acc: 0.5000
    Epoch 00072: saving model to /content/data/model/weights.epoch_72.hdf5
    163/163 [==============================] - 85s 521ms/step - loss: 0.0746 - acc: 0.9795 - val_loss: 10.4079 - val_acc: 0.5000
    Epoch 73/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0746 - acc: 0.9797Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 8.7054 - acc: 0.5625
    Epoch 00073: saving model to /content/data/model/weights.epoch_73.hdf5
    163/163 [==============================] - 85s 522ms/step - loss: 0.0743 - acc: 0.9797 - val_loss: 8.7054 - val_acc: 0.5625
    Epoch 74/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0640 - acc: 0.9796Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 9.3623 - acc: 0.5000
    Epoch 00074: saving model to /content/data/model/weights.epoch_74.hdf5
    163/163 [==============================] - 85s 522ms/step - loss: 0.0636 - acc: 0.9797 - val_loss: 9.3623 - val_acc: 0.5000
    Epoch 75/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0666 - acc: 0.9826Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 7.5627 - acc: 0.5625
    Epoch 00075: saving model to /content/data/model/weights.epoch_75.hdf5
    163/163 [==============================] - 85s 521ms/step - loss: 0.0663 - acc: 0.9827 - val_loss: 7.5627 - val_acc: 0.5625
    Epoch 76/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0741 - acc: 0.9778Epoch 1/100
      1/163 [..............................] - ETA: 50s - loss: 8.9220 - acc: 0.5000
    Epoch 00076: saving model to /content/data/model/weights.epoch_76.hdf5
    163/163 [==============================] - 85s 520ms/step - loss: 0.0737 - acc: 0.9780 - val_loss: 8.9220 - val_acc: 0.5000
    Epoch 77/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0706 - acc: 0.9786Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 4.8831 - acc: 0.6250
    Epoch 00077: saving model to /content/data/model/weights.epoch_77.hdf5
    163/163 [==============================] - 84s 518ms/step - loss: 0.0709 - acc: 0.9785 - val_loss: 4.8831 - val_acc: 0.6250
    Epoch 78/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0722 - acc: 0.9799Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 5.9518 - acc: 0.5625
    Epoch 00078: saving model to /content/data/model/weights.epoch_78.hdf5
    163/163 [==============================] - 85s 522ms/step - loss: 0.0731 - acc: 0.9799 - val_loss: 5.9518 - val_acc: 0.5625
    Epoch 79/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0661 - acc: 0.9799Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 6.6153 - acc: 0.5625
    Epoch 00079: saving model to /content/data/model/weights.epoch_79.hdf5
    163/163 [==============================] - 85s 520ms/step - loss: 0.0671 - acc: 0.9797 - val_loss: 6.6153 - val_acc: 0.5625
    Epoch 80/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0534 - acc: 0.9834Epoch 1/100
      1/163 [..............................] - ETA: 51s - loss: 8.5519 - acc: 0.5000
    Epoch 00080: saving model to /content/data/model/weights.epoch_80.hdf5
    163/163 [==============================] - 84s 517ms/step - loss: 0.0532 - acc: 0.9835 - val_loss: 8.5519 - val_acc: 0.5000
    Epoch 81/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0729 - acc: 0.9792Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 6.9769 - acc: 0.5625
    Epoch 00081: saving model to /content/data/model/weights.epoch_81.hdf5
    163/163 [==============================] - 84s 517ms/step - loss: 0.0724 - acc: 0.9793 - val_loss: 6.9769 - val_acc: 0.5625
    Epoch 82/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0646 - acc: 0.9797Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 7.0855 - acc: 0.5000
    Epoch 00082: saving model to /content/data/model/weights.epoch_82.hdf5
    163/163 [==============================] - 85s 522ms/step - loss: 0.0652 - acc: 0.9795 - val_loss: 7.0855 - val_acc: 0.5000
    Epoch 83/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0713 - acc: 0.9813Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 4.9015 - acc: 0.6250
    Epoch 00083: saving model to /content/data/model/weights.epoch_83.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.0728 - acc: 0.9812 - val_loss: 4.9015 - val_acc: 0.6250
    Epoch 84/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0585 - acc: 0.9832Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 6.8088 - acc: 0.5000
    Epoch 00084: saving model to /content/data/model/weights.epoch_84.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0583 - acc: 0.9831 - val_loss: 6.8088 - val_acc: 0.5000
    Epoch 85/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0718 - acc: 0.9794Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 7.9591 - acc: 0.5000
    Epoch 00085: saving model to /content/data/model/weights.epoch_85.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0715 - acc: 0.9795 - val_loss: 7.9591 - val_acc: 0.5000
    Epoch 86/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0670 - acc: 0.9801Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 5.3045 - acc: 0.5625
    Epoch 00086: saving model to /content/data/model/weights.epoch_86.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.0666 - acc: 0.9803 - val_loss: 5.3045 - val_acc: 0.5625
    Epoch 87/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0646 - acc: 0.9797Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 6.9546 - acc: 0.5625
    Epoch 00087: saving model to /content/data/model/weights.epoch_87.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0651 - acc: 0.9797 - val_loss: 6.9546 - val_acc: 0.5625
    Epoch 88/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0834 - acc: 0.9767Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 6.5918 - acc: 0.5625
    Epoch 00088: saving model to /content/data/model/weights.epoch_88.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0829 - acc: 0.9768 - val_loss: 6.5918 - val_acc: 0.5625
    Epoch 89/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0662 - acc: 0.9828Epoch 1/100
      1/163 [..............................] - ETA: 52s - loss: 9.3918 - acc: 0.5000
    Epoch 00089: saving model to /content/data/model/weights.epoch_89.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0662 - acc: 0.9827 - val_loss: 9.3918 - val_acc: 0.5000
    Epoch 90/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0635 - acc: 0.9826Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 10.3696 - acc: 0.5000
    Epoch 00090: saving model to /content/data/model/weights.epoch_90.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0645 - acc: 0.9826 - val_loss: 10.3696 - val_acc: 0.5000
    Epoch 91/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0709 - acc: 0.9803Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 10.2275 - acc: 0.5000
    Epoch 00091: saving model to /content/data/model/weights.epoch_91.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.0705 - acc: 0.9804 - val_loss: 10.2275 - val_acc: 0.5000
    Epoch 92/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0770 - acc: 0.9807Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 11.1581 - acc: 0.5000
    Epoch 00092: saving model to /content/data/model/weights.epoch_92.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0768 - acc: 0.9806 - val_loss: 11.1581 - val_acc: 0.5000
    Epoch 93/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0675 - acc: 0.9815Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 9.9042 - acc: 0.5000
    Epoch 00093: saving model to /content/data/model/weights.epoch_93.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0671 - acc: 0.9816 - val_loss: 9.9042 - val_acc: 0.5000
    Epoch 94/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0676 - acc: 0.9819Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 12.0309 - acc: 0.5000
    Epoch 00094: saving model to /content/data/model/weights.epoch_94.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.0676 - acc: 0.9818 - val_loss: 12.0309 - val_acc: 0.5000
    Epoch 95/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0725 - acc: 0.9805Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 6.1905 - acc: 0.5625
    Epoch 00095: saving model to /content/data/model/weights.epoch_95.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.0732 - acc: 0.9803 - val_loss: 6.1905 - val_acc: 0.5625
    Epoch 96/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0692 - acc: 0.9807Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 6.8690 - acc: 0.5625
    Epoch 00096: saving model to /content/data/model/weights.epoch_96.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0688 - acc: 0.9808 - val_loss: 6.8690 - val_acc: 0.5625
    Epoch 97/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0640 - acc: 0.9794Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 8.5963 - acc: 0.5625
    Epoch 00097: saving model to /content/data/model/weights.epoch_97.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.0651 - acc: 0.9791 - val_loss: 8.5963 - val_acc: 0.5625
    Epoch 98/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0763 - acc: 0.9784Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 8.3895 - acc: 0.5000
    Epoch 00098: saving model to /content/data/model/weights.epoch_98.hdf5
    163/163 [==============================] - 85s 521ms/step - loss: 0.0763 - acc: 0.9783 - val_loss: 8.3895 - val_acc: 0.5000
    Epoch 99/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0727 - acc: 0.9805Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 7.5499 - acc: 0.5000
    Epoch 00099: saving model to /content/data/model/weights.epoch_99.hdf5
    163/163 [==============================] - 85s 522ms/step - loss: 0.0727 - acc: 0.9804 - val_loss: 7.5499 - val_acc: 0.5000
    Epoch 100/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0643 - acc: 0.9824Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 8.4232 - acc: 0.5000
    Epoch 00100: saving model to /content/data/model/weights.epoch_100.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0639 - acc: 0.9826 - val_loss: 8.4232 - val_acc: 0.5000
    Loading the best model
    epoch: 1, val_loss: 0.8304075598716736, val_acc: 0.8125
    20/20 [==============================] - 8s 387ms/step - loss: 0.6455 - acc: 0.8093
    20/20 [==============================] - 7s 358ms/step
    CONFUSION MATRIX ------------------
    [[136  98]
     [ 21 369]]
    
    TEST METRICS ----------------------
    Accuracy: 80.92948717948718%
    Precision: 79.01498929336188%
    Recall: 94.61538461538461%
    F1-score: 86.11435239206534
    
    TRAIN METRIC ----------------------
    Train acc: 98.25536608695984%
    


![png](ResNet50V2%20Version%201.1.1.0.0_files/ResNet50V2%20Version%201.1.1.0.0_4_1.png)

