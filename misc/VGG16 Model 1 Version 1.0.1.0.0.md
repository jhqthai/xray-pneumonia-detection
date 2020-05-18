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

    --2019-10-30 03:46:42--  https://s3.eu-central-1.amazonaws.com/public.unit8.co/data/chest_xray.tar.gz
    Resolving s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)... 52.219.74.60
    Connecting to s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)|52.219.74.60|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1225393795 (1.1G) [application/x-gzip]
    Saving to: ‘/content/data/chest_xray.tar.gz’
    
    /content/data/chest 100%[===================>]   1.14G  82.2MB/s    in 13s     
    
    2019-10-30 03:46:56 (87.6 MB/s) - ‘/content/data/chest_xray.tar.gz’ saved [1225393795/1225393795]
    
    

Change log:
> **training_datagen --> ImageDataGenerator(NOTHING)**

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
    # rescale = 1./255,
)

validation_datagen = ImageDataGenerator(
    # rescale = 1./255
)

test_datagen = ImageDataGenerator(
    # rescale = 1./255
)

# Create training data batch
# TODO: Try grayscaling the image to see what will happen
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
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
    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58892288/58889256 [==============================] - 3s 0us/step
    [1.9448173  0.67303226]
    Epoch 1/100
    162/163 [============================>.] - ETA: 0s - loss: 0.6576 - acc: 0.9302Epoch 1/100
      1/163 [..............................] - ETA: 7:06 - loss: 3.4501e-04 - acc: 1.0000
    Epoch 00001: saving model to /content/data/model/weights.epoch_01.hdf5
    163/163 [==============================] - 82s 502ms/step - loss: 0.6575 - acc: 0.9304 - val_loss: 3.4501e-04 - val_acc: 1.0000
    Epoch 2/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1611 - acc: 0.9770Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0131 - acc: 1.0000
    Epoch 00002: saving model to /content/data/model/weights.epoch_02.hdf5
    163/163 [==============================] - 74s 455ms/step - loss: 0.1609 - acc: 0.9770 - val_loss: 0.0131 - val_acc: 1.0000
    Epoch 3/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0772 - acc: 0.9880Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.1195 - acc: 0.9375
    Epoch 00003: saving model to /content/data/model/weights.epoch_03.hdf5
    163/163 [==============================] - 76s 466ms/step - loss: 0.0767 - acc: 0.9881 - val_loss: 0.1195 - val_acc: 0.9375
    Epoch 4/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0499 - acc: 0.9929Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0029 - acc: 1.0000
    Epoch 00004: saving model to /content/data/model/weights.epoch_04.hdf5
    163/163 [==============================] - 76s 464ms/step - loss: 0.0496 - acc: 0.9929 - val_loss: 0.0029 - val_acc: 1.0000
    Epoch 5/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0090 - acc: 0.9977Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0301 - acc: 1.0000
    Epoch 00005: saving model to /content/data/model/weights.epoch_05.hdf5
    163/163 [==============================] - 76s 465ms/step - loss: 0.0090 - acc: 0.9977 - val_loss: 0.0301 - val_acc: 1.0000
    Epoch 6/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0182 - acc: 0.9979Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0280 - acc: 1.0000
    Epoch 00006: saving model to /content/data/model/weights.epoch_06.hdf5
    163/163 [==============================] - 75s 460ms/step - loss: 0.0181 - acc: 0.9979 - val_loss: 0.0280 - val_acc: 1.0000
    Epoch 7/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0011 - acc: 0.9998Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0024 - acc: 1.0000
    Epoch 00007: saving model to /content/data/model/weights.epoch_07.hdf5
    163/163 [==============================] - 76s 466ms/step - loss: 0.0011 - acc: 0.9998 - val_loss: 0.0024 - val_acc: 1.0000
    Epoch 8/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0048 - acc: 0.9983Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.1892 - acc: 0.9375
    Epoch 00008: saving model to /content/data/model/weights.epoch_08.hdf5
    163/163 [==============================] - 76s 464ms/step - loss: 0.0047 - acc: 0.9983 - val_loss: 0.1892 - val_acc: 0.9375
    Epoch 9/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0029 - acc: 0.9988Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0028 - acc: 1.0000
    Epoch 00009: saving model to /content/data/model/weights.epoch_09.hdf5
    163/163 [==============================] - 76s 464ms/step - loss: 0.0029 - acc: 0.9988 - val_loss: 0.0028 - val_acc: 1.0000
    Epoch 10/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0046 - acc: 0.9994Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0072 - acc: 1.0000
    Epoch 00010: saving model to /content/data/model/weights.epoch_10.hdf5
    163/163 [==============================] - 75s 463ms/step - loss: 0.0046 - acc: 0.9994 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 11/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0012 - acc: 0.9994Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 3.2180e-04 - acc: 1.0000
    Epoch 00011: saving model to /content/data/model/weights.epoch_11.hdf5
    163/163 [==============================] - 76s 464ms/step - loss: 0.0012 - acc: 0.9994 - val_loss: 3.2180e-04 - val_acc: 1.0000
    Epoch 12/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0025 - acc: 0.9990Epoch 1/100
      1/163 [..............................] - ETA: 1:19 - loss: 5.0857e-04 - acc: 1.0000
    Epoch 00012: saving model to /content/data/model/weights.epoch_12.hdf5
    163/163 [==============================] - 76s 469ms/step - loss: 0.0027 - acc: 0.9988 - val_loss: 5.0857e-04 - val_acc: 1.0000
    Epoch 13/100
    162/163 [============================>.] - ETA: 0s - loss: 1.0987e-04 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:12 - loss: 2.7596e-04 - acc: 1.0000
    Epoch 00013: saving model to /content/data/model/weights.epoch_13.hdf5
    163/163 [==============================] - 76s 469ms/step - loss: 1.0920e-04 - acc: 1.0000 - val_loss: 2.7596e-04 - val_acc: 1.0000
    Epoch 14/100
    162/163 [============================>.] - ETA: 0s - loss: 2.6691e-07 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 6.7055e-08 - acc: 1.0000
    Epoch 00014: saving model to /content/data/model/weights.epoch_14.hdf5
    163/163 [==============================] - 76s 468ms/step - loss: 2.6527e-07 - acc: 1.0000 - val_loss: 6.7055e-08 - val_acc: 1.0000
    Epoch 15/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0047 - acc: 0.9992Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 5.7648e-05 - acc: 1.0000
    Epoch 00015: saving model to /content/data/model/weights.epoch_15.hdf5
    163/163 [==============================] - 77s 471ms/step - loss: 0.0046 - acc: 0.9992 - val_loss: 5.7648e-05 - val_acc: 1.0000
    Epoch 16/100
    162/163 [============================>.] - ETA: 0s - loss: 1.1142e-05 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0043 - acc: 1.0000
    Epoch 00016: saving model to /content/data/model/weights.epoch_16.hdf5
    163/163 [==============================] - 76s 469ms/step - loss: 1.1073e-05 - acc: 1.0000 - val_loss: 0.0043 - val_acc: 1.0000
    Epoch 17/100
    162/163 [============================>.] - ETA: 0s - loss: 6.5034e-04 - acc: 0.9996Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 2.5331e-06 - acc: 1.0000
    Epoch 00017: saving model to /content/data/model/weights.epoch_17.hdf5
    163/163 [==============================] - 76s 468ms/step - loss: 6.4640e-04 - acc: 0.9996 - val_loss: 2.5331e-06 - val_acc: 1.0000
    Epoch 18/100
    162/163 [============================>.] - ETA: 0s - loss: 2.0865e-04 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:12 - loss: 0.0014 - acc: 1.0000
    Epoch 00018: saving model to /content/data/model/weights.epoch_18.hdf5
    163/163 [==============================] - 76s 464ms/step - loss: 2.0737e-04 - acc: 1.0000 - val_loss: 0.0014 - val_acc: 1.0000
    Epoch 19/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0021 - acc: 0.9998Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 9.7460e-05 - acc: 1.0000
    Epoch 00019: saving model to /content/data/model/weights.epoch_19.hdf5
    163/163 [==============================] - 75s 461ms/step - loss: 0.0021 - acc: 0.9998 - val_loss: 9.7460e-05 - val_acc: 1.0000
    Epoch 20/100
    162/163 [============================>.] - ETA: 0s - loss: 1.8856e-09 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:17 - loss: 2.2273e-04 - acc: 1.0000
    Epoch 00020: saving model to /content/data/model/weights.epoch_20.hdf5
    163/163 [==============================] - 77s 474ms/step - loss: 1.8741e-09 - acc: 1.0000 - val_loss: 2.2273e-04 - val_acc: 1.0000
    Epoch 21/100
    162/163 [============================>.] - ETA: 0s - loss: 6.7991e-05 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:16 - loss: 0.0095 - acc: 1.0000
    Epoch 00021: saving model to /content/data/model/weights.epoch_21.hdf5
    163/163 [==============================] - 76s 468ms/step - loss: 6.7574e-05 - acc: 1.0000 - val_loss: 0.0095 - val_acc: 1.0000
    Epoch 22/100
    162/163 [============================>.] - ETA: 0s - loss: 1.2153e-05 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:12 - loss: 0.0028 - acc: 1.0000
    Epoch 00022: saving model to /content/data/model/weights.epoch_22.hdf5
    163/163 [==============================] - 76s 468ms/step - loss: 1.2078e-05 - acc: 1.0000 - val_loss: 0.0028 - val_acc: 1.0000
    Epoch 23/100
    162/163 [============================>.] - ETA: 0s - loss: 1.5637e-09 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 8.6583e-04 - acc: 1.0000
    Epoch 00023: saving model to /content/data/model/weights.epoch_23.hdf5
    163/163 [==============================] - 76s 464ms/step - loss: 1.5541e-09 - acc: 1.0000 - val_loss: 8.6583e-04 - val_acc: 1.0000
    Epoch 24/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:11 - loss: 0.0017 - acc: 1.0000
    Epoch 00024: saving model to /content/data/model/weights.epoch_24.hdf5
    163/163 [==============================] - 76s 463ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0017 - val_acc: 1.0000
    Epoch 25/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0028 - acc: 1.0000
    Epoch 00025: saving model to /content/data/model/weights.epoch_25.hdf5
    163/163 [==============================] - 75s 460ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0028 - val_acc: 1.0000
    Epoch 26/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0036 - acc: 1.0000
    Epoch 00026: saving model to /content/data/model/weights.epoch_26.hdf5
    163/163 [==============================] - 77s 471ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0036 - val_acc: 1.0000
    Epoch 27/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0032 - acc: 1.0000
    Epoch 00027: saving model to /content/data/model/weights.epoch_27.hdf5
    163/163 [==============================] - 77s 473ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0032 - val_acc: 1.0000
    Epoch 28/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:10 - loss: 0.0041 - acc: 1.0000
    Epoch 00028: saving model to /content/data/model/weights.epoch_28.hdf5
    163/163 [==============================] - 78s 478ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0041 - val_acc: 1.0000
    Epoch 29/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0046 - acc: 1.0000
    Epoch 00029: saving model to /content/data/model/weights.epoch_29.hdf5
    163/163 [==============================] - 77s 472ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0046 - val_acc: 1.0000
    Epoch 30/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0060 - acc: 1.0000
    Epoch 00030: saving model to /content/data/model/weights.epoch_30.hdf5
    163/163 [==============================] - 76s 465ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0060 - val_acc: 1.0000
    Epoch 31/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0047 - acc: 1.0000
    Epoch 00031: saving model to /content/data/model/weights.epoch_31.hdf5
    163/163 [==============================] - 76s 464ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0047 - val_acc: 1.0000
    Epoch 32/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0040 - acc: 1.0000
    Epoch 00032: saving model to /content/data/model/weights.epoch_32.hdf5
    163/163 [==============================] - 78s 481ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0040 - val_acc: 1.0000
    Epoch 33/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0039 - acc: 1.0000
    Epoch 00033: saving model to /content/data/model/weights.epoch_33.hdf5
    163/163 [==============================] - 77s 470ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0039 - val_acc: 1.0000
    Epoch 34/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:16 - loss: 0.0041 - acc: 1.0000
    Epoch 00034: saving model to /content/data/model/weights.epoch_34.hdf5
    163/163 [==============================] - 76s 465ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0041 - val_acc: 1.0000
    Epoch 35/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:12 - loss: 0.0056 - acc: 1.0000
    Epoch 00035: saving model to /content/data/model/weights.epoch_35.hdf5
    163/163 [==============================] - 75s 460ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0056 - val_acc: 1.0000
    Epoch 36/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:12 - loss: 0.0046 - acc: 1.0000
    Epoch 00036: saving model to /content/data/model/weights.epoch_36.hdf5
    163/163 [==============================] - 76s 464ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0046 - val_acc: 1.0000
    Epoch 37/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0052 - acc: 1.0000
    Epoch 00037: saving model to /content/data/model/weights.epoch_37.hdf5
    163/163 [==============================] - 76s 465ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0052 - val_acc: 1.0000
    Epoch 38/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:18 - loss: 0.0049 - acc: 1.0000
    Epoch 00038: saving model to /content/data/model/weights.epoch_38.hdf5
    163/163 [==============================] - 78s 476ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0049 - val_acc: 1.0000
    Epoch 39/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:17 - loss: 0.0045 - acc: 1.0000
    Epoch 00039: saving model to /content/data/model/weights.epoch_39.hdf5
    163/163 [==============================] - 75s 463ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0045 - val_acc: 1.0000
    Epoch 40/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0055 - acc: 1.0000
    Epoch 00040: saving model to /content/data/model/weights.epoch_40.hdf5
    163/163 [==============================] - 78s 478ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0055 - val_acc: 1.0000
    Epoch 41/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0055 - acc: 1.0000
    Epoch 00041: saving model to /content/data/model/weights.epoch_41.hdf5
    163/163 [==============================] - 76s 466ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0055 - val_acc: 1.0000
    Epoch 42/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:17 - loss: 0.0042 - acc: 1.0000
    Epoch 00042: saving model to /content/data/model/weights.epoch_42.hdf5
    163/163 [==============================] - 78s 481ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0042 - val_acc: 1.0000
    Epoch 43/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0044 - acc: 1.0000
    Epoch 00043: saving model to /content/data/model/weights.epoch_43.hdf5
    163/163 [==============================] - 78s 480ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0044 - val_acc: 1.0000
    Epoch 44/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:10 - loss: 0.0048 - acc: 1.0000
    Epoch 00044: saving model to /content/data/model/weights.epoch_44.hdf5
    163/163 [==============================] - 77s 470ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0048 - val_acc: 1.0000
    Epoch 45/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0047 - acc: 1.0000
    Epoch 00045: saving model to /content/data/model/weights.epoch_45.hdf5
    163/163 [==============================] - 76s 463ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0047 - val_acc: 1.0000
    Epoch 46/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0048 - acc: 1.0000
    Epoch 00046: saving model to /content/data/model/weights.epoch_46.hdf5
    163/163 [==============================] - 76s 468ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0048 - val_acc: 1.0000
    Epoch 47/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:16 - loss: 0.0050 - acc: 1.0000
    Epoch 00047: saving model to /content/data/model/weights.epoch_47.hdf5
    163/163 [==============================] - 78s 479ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0050 - val_acc: 1.0000
    Epoch 48/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0052 - acc: 1.0000
    Epoch 00048: saving model to /content/data/model/weights.epoch_48.hdf5
    163/163 [==============================] - 78s 481ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0052 - val_acc: 1.0000
    Epoch 49/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0046 - acc: 1.0000
    Epoch 00049: saving model to /content/data/model/weights.epoch_49.hdf5
    163/163 [==============================] - 76s 468ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0046 - val_acc: 1.0000
    Epoch 50/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:17 - loss: 0.0047 - acc: 1.0000
    Epoch 00050: saving model to /content/data/model/weights.epoch_50.hdf5
    163/163 [==============================] - 76s 464ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0047 - val_acc: 1.0000
    Epoch 51/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:10 - loss: 0.0048 - acc: 1.0000
    Epoch 00051: saving model to /content/data/model/weights.epoch_51.hdf5
    163/163 [==============================] - 75s 461ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0048 - val_acc: 1.0000
    Epoch 52/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0045 - acc: 1.0000
    Epoch 00052: saving model to /content/data/model/weights.epoch_52.hdf5
    163/163 [==============================] - 76s 466ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0045 - val_acc: 1.0000
    Epoch 53/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0054 - acc: 1.0000
    Epoch 00053: saving model to /content/data/model/weights.epoch_53.hdf5
    163/163 [==============================] - 77s 472ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0054 - val_acc: 1.0000
    Epoch 54/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0049 - acc: 1.0000
    Epoch 00054: saving model to /content/data/model/weights.epoch_54.hdf5
    163/163 [==============================] - 77s 473ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0049 - val_acc: 1.0000
    Epoch 55/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:16 - loss: 0.0047 - acc: 1.0000
    Epoch 00055: saving model to /content/data/model/weights.epoch_55.hdf5
    163/163 [==============================] - 78s 478ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0047 - val_acc: 1.0000
    Epoch 56/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0050 - acc: 1.0000
    Epoch 00056: saving model to /content/data/model/weights.epoch_56.hdf5
    163/163 [==============================] - 78s 478ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0050 - val_acc: 1.0000
    Epoch 57/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0052 - acc: 1.0000
    Epoch 00057: saving model to /content/data/model/weights.epoch_57.hdf5
    163/163 [==============================] - 76s 466ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0052 - val_acc: 1.0000
    Epoch 58/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0045 - acc: 1.0000
    Epoch 00058: saving model to /content/data/model/weights.epoch_58.hdf5
    163/163 [==============================] - 75s 461ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0045 - val_acc: 1.0000
    Epoch 59/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0055 - acc: 1.0000
    Epoch 00059: saving model to /content/data/model/weights.epoch_59.hdf5
    163/163 [==============================] - 75s 461ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0055 - val_acc: 1.0000
    Epoch 60/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0052 - acc: 1.0000
    Epoch 00060: saving model to /content/data/model/weights.epoch_60.hdf5
    163/163 [==============================] - 75s 463ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0052 - val_acc: 1.0000
    Epoch 61/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0047 - acc: 1.0000
    Epoch 00061: saving model to /content/data/model/weights.epoch_61.hdf5
    163/163 [==============================] - 76s 465ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0047 - val_acc: 1.0000
    Epoch 62/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:11 - loss: 0.0047 - acc: 1.0000
    Epoch 00062: saving model to /content/data/model/weights.epoch_62.hdf5
    163/163 [==============================] - 75s 460ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0047 - val_acc: 1.0000
    Epoch 63/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0049 - acc: 1.0000
    Epoch 00063: saving model to /content/data/model/weights.epoch_63.hdf5
    163/163 [==============================] - 75s 463ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0049 - val_acc: 1.0000
    Epoch 64/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0047 - acc: 1.0000
    Epoch 00064: saving model to /content/data/model/weights.epoch_64.hdf5
    163/163 [==============================] - 75s 459ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0047 - val_acc: 1.0000
    Epoch 65/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0052 - acc: 1.0000
    Epoch 00065: saving model to /content/data/model/weights.epoch_65.hdf5
    163/163 [==============================] - 75s 461ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0052 - val_acc: 1.0000
    Epoch 66/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0046 - acc: 1.0000
    Epoch 00066: saving model to /content/data/model/weights.epoch_66.hdf5
    163/163 [==============================] - 75s 463ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0046 - val_acc: 1.0000
    Epoch 67/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:11 - loss: 0.0046 - acc: 1.0000
    Epoch 00067: saving model to /content/data/model/weights.epoch_67.hdf5
    163/163 [==============================] - 76s 468ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0046 - val_acc: 1.0000
    Epoch 68/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:12 - loss: 0.0050 - acc: 1.0000
    Epoch 00068: saving model to /content/data/model/weights.epoch_68.hdf5
    163/163 [==============================] - 77s 471ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0050 - val_acc: 1.0000
    Epoch 69/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0050 - acc: 1.0000
    Epoch 00069: saving model to /content/data/model/weights.epoch_69.hdf5
    163/163 [==============================] - 76s 464ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0050 - val_acc: 1.0000
    Epoch 70/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0052 - acc: 1.0000
    Epoch 00070: saving model to /content/data/model/weights.epoch_70.hdf5
    163/163 [==============================] - 77s 475ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0052 - val_acc: 1.0000
    Epoch 71/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0049 - acc: 1.0000
    Epoch 00071: saving model to /content/data/model/weights.epoch_71.hdf5
    163/163 [==============================] - 77s 469ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0049 - val_acc: 1.0000
    Epoch 72/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0048 - acc: 1.0000
    Epoch 00072: saving model to /content/data/model/weights.epoch_72.hdf5
    163/163 [==============================] - 77s 473ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0048 - val_acc: 1.0000
    Epoch 73/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0049 - acc: 1.0000
    Epoch 00073: saving model to /content/data/model/weights.epoch_73.hdf5
    163/163 [==============================] - 76s 467ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0049 - val_acc: 1.0000
    Epoch 74/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:16 - loss: 0.0050 - acc: 1.0000
    Epoch 00074: saving model to /content/data/model/weights.epoch_74.hdf5
    163/163 [==============================] - 77s 471ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0050 - val_acc: 1.0000
    Epoch 75/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:16 - loss: 0.0053 - acc: 1.0000
    Epoch 00075: saving model to /content/data/model/weights.epoch_75.hdf5
    163/163 [==============================] - 78s 480ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0053 - val_acc: 1.0000
    Epoch 76/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:17 - loss: 0.0052 - acc: 1.0000
    Epoch 00076: saving model to /content/data/model/weights.epoch_76.hdf5
    163/163 [==============================] - 78s 479ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0052 - val_acc: 1.0000
    Epoch 77/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0049 - acc: 1.0000
    Epoch 00077: saving model to /content/data/model/weights.epoch_77.hdf5
    163/163 [==============================] - 76s 464ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0049 - val_acc: 1.0000
    Epoch 78/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0047 - acc: 1.0000
    Epoch 00078: saving model to /content/data/model/weights.epoch_78.hdf5
    163/163 [==============================] - 78s 476ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0047 - val_acc: 1.0000
    Epoch 79/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0051 - acc: 1.0000
    Epoch 00079: saving model to /content/data/model/weights.epoch_79.hdf5
    163/163 [==============================] - 78s 476ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0051 - val_acc: 1.0000
    Epoch 80/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:16 - loss: 0.0049 - acc: 1.0000
    Epoch 00080: saving model to /content/data/model/weights.epoch_80.hdf5
    163/163 [==============================] - 77s 472ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0049 - val_acc: 1.0000
    Epoch 81/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0050 - acc: 1.0000
    Epoch 00081: saving model to /content/data/model/weights.epoch_81.hdf5
    163/163 [==============================] - 76s 468ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0050 - val_acc: 1.0000
    Epoch 82/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:16 - loss: 0.0049 - acc: 1.0000
    Epoch 00082: saving model to /content/data/model/weights.epoch_82.hdf5
    163/163 [==============================] - 77s 471ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0049 - val_acc: 1.0000
    Epoch 83/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:16 - loss: 0.0053 - acc: 1.0000
    Epoch 00083: saving model to /content/data/model/weights.epoch_83.hdf5
    163/163 [==============================] - 78s 476ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0053 - val_acc: 1.0000
    Epoch 84/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0052 - acc: 1.0000
    Epoch 00084: saving model to /content/data/model/weights.epoch_84.hdf5
    163/163 [==============================] - 77s 470ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0052 - val_acc: 1.0000
    Epoch 85/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0047 - acc: 1.0000
    Epoch 00085: saving model to /content/data/model/weights.epoch_85.hdf5
    163/163 [==============================] - 76s 467ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0047 - val_acc: 1.0000
    Epoch 86/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:17 - loss: 0.0052 - acc: 1.0000
    Epoch 00086: saving model to /content/data/model/weights.epoch_86.hdf5
    163/163 [==============================] - 77s 472ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0052 - val_acc: 1.0000
    Epoch 87/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0051 - acc: 1.0000
    Epoch 00087: saving model to /content/data/model/weights.epoch_87.hdf5
    163/163 [==============================] - 78s 478ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0051 - val_acc: 1.0000
    Epoch 88/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:16 - loss: 0.0052 - acc: 1.0000
    Epoch 00088: saving model to /content/data/model/weights.epoch_88.hdf5
    163/163 [==============================] - 78s 479ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0052 - val_acc: 1.0000
    Epoch 89/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:16 - loss: 0.0055 - acc: 1.0000
    Epoch 00089: saving model to /content/data/model/weights.epoch_89.hdf5
    163/163 [==============================] - 76s 469ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0055 - val_acc: 1.0000
    Epoch 90/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0049 - acc: 1.0000
    Epoch 00090: saving model to /content/data/model/weights.epoch_90.hdf5
    163/163 [==============================] - 75s 463ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0049 - val_acc: 1.0000
    Epoch 91/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:18 - loss: 0.0050 - acc: 1.0000
    Epoch 00091: saving model to /content/data/model/weights.epoch_91.hdf5
    163/163 [==============================] - 77s 473ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0050 - val_acc: 1.0000
    Epoch 92/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0049 - acc: 1.0000
    Epoch 00092: saving model to /content/data/model/weights.epoch_92.hdf5
    163/163 [==============================] - 78s 476ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0049 - val_acc: 1.0000
    Epoch 93/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0052 - acc: 1.0000
    Epoch 00093: saving model to /content/data/model/weights.epoch_93.hdf5
    163/163 [==============================] - 78s 476ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0052 - val_acc: 1.0000
    Epoch 94/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0053 - acc: 1.0000
    Epoch 00094: saving model to /content/data/model/weights.epoch_94.hdf5
    163/163 [==============================] - 77s 474ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0053 - val_acc: 1.0000
    Epoch 95/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0054 - acc: 1.0000
    Epoch 00095: saving model to /content/data/model/weights.epoch_95.hdf5
    163/163 [==============================] - 78s 477ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0054 - val_acc: 1.0000
    Epoch 96/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:16 - loss: 0.0050 - acc: 1.0000
    Epoch 00096: saving model to /content/data/model/weights.epoch_96.hdf5
    163/163 [==============================] - 78s 480ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0050 - val_acc: 1.0000
    Epoch 97/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:14 - loss: 0.0053 - acc: 1.0000
    Epoch 00097: saving model to /content/data/model/weights.epoch_97.hdf5
    163/163 [==============================] - 78s 480ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0053 - val_acc: 1.0000
    Epoch 98/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:17 - loss: 0.0051 - acc: 1.0000
    Epoch 00098: saving model to /content/data/model/weights.epoch_98.hdf5
    163/163 [==============================] - 78s 478ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0051 - val_acc: 1.0000
    Epoch 99/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:13 - loss: 0.0051 - acc: 1.0000
    Epoch 00099: saving model to /content/data/model/weights.epoch_99.hdf5
    163/163 [==============================] - 77s 473ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0051 - val_acc: 1.0000
    Epoch 100/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0000e+00 - acc: 1.0000Epoch 1/100
      1/163 [..............................] - ETA: 1:15 - loss: 0.0053 - acc: 1.0000
    Epoch 00100: saving model to /content/data/model/weights.epoch_100.hdf5
    163/163 [==============================] - 76s 464ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0053 - val_acc: 1.0000
    Loading the best model
    epoch: 14, val_loss: 6.705518984517767e-08, val_acc: 1.0
    20/20 [==============================] - 10s 480ms/step - loss: 4.3357 - acc: 0.8157
    20/20 [==============================] - 10s 494ms/step
    CONFUSION MATRIX ------------------
    [[120 114]
     [  1 389]]
    
    TEST METRICS ----------------------
    Accuracy: 81.57051282051282%
    Precision: 77.33598409542743%
    Recall: 99.74358974358975%
    F1-score: 87.12206047032474
    
    TRAIN METRIC ----------------------
    Train acc: 100.0%
    


![png](VGG16%20Model%201%20Version%201.0.1.0.0_files/VGG16%20Model%201%20Version%201.0.1.0.0_4_1.png)

