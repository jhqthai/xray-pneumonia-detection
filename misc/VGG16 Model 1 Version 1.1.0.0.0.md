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

    --2019-10-31 04:30:59--  https://s3.eu-central-1.amazonaws.com/public.unit8.co/data/chest_xray.tar.gz
    Resolving s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)... 52.219.74.151
    Connecting to s3.eu-central-1.amazonaws.com (s3.eu-central-1.amazonaws.com)|52.219.74.151|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1225393795 (1.1G) [application/x-gzip]
    Saving to: ‘/content/data/chest_xray.tar.gz’
    
    /content/data/chest 100%[===================>]   1.14G  30.0MB/s    in 40s     
    
    2019-10-31 04:31:39 (29.2 MB/s) - ‘/content/data/chest_xray.tar.gz’ saved [1225393795/1225393795]
    
    

Change log:
> training_datagen --> ImageDataGenerator

> trainable layer --> All except base

> 20 layers VGG16 model - base, flat, dense

> **Optimizer = RMSprop(learning_rate = 0.001)**

> loss = categorical_crosscentropy

> callback = [checkpoints]

> epochs = 100

> no class weight balancing



```
TRAINING_DIR = "/content/data/chest_xray/train"
VALIDATION_DIR = "/content/data/chest_xray/val"
TEST_DIR = "/content/data/chest_xray/test"

# Data preprocessing
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

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

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

## Graph loss and acc
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


## Evaualate
test_loss, test_acc = model.evaluate_generator(generator=test_generator, verbose=1)

## Load best weight
idx = np.argmin(history.history['val_loss']) 
model.load_weights("/content/data/model/weights.epoch_{:02d}.hdf5".format(idx + 1))

print("Loading the best model")
print("epoch: {}, val_loss: {}, val_acc: {}".format(idx + 1, history.history['val_loss'][idx], history.history['val_acc'][idx]))

## Evaluate the best weight
test_loss, test_acc = model.evaluate_generator(generator=test_generator, verbose=1)

## Test analytics
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
    162/163 [============================>.] - ETA: 0s - loss: 0.2429 - acc: 0.8978Epoch 1/100
      1/163 [..............................] - ETA: 4:07 - loss: 0.5661 - acc: 0.7500
    Epoch 00001: saving model to /content/data/model/weights.epoch_01.hdf5
    163/163 [==============================] - 89s 547ms/step - loss: 0.2419 - acc: 0.8980 - val_loss: 0.5661 - val_acc: 0.7500
    Epoch 2/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1500 - acc: 0.9417Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2780 - acc: 0.8125
    Epoch 00002: saving model to /content/data/model/weights.epoch_02.hdf5
    163/163 [==============================] - 88s 537ms/step - loss: 0.1495 - acc: 0.9419 - val_loss: 0.2780 - val_acc: 0.8125
    Epoch 3/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1334 - acc: 0.9466Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.8554 - acc: 0.7500
    Epoch 00003: saving model to /content/data/model/weights.epoch_03.hdf5
    163/163 [==============================] - 87s 537ms/step - loss: 0.1343 - acc: 0.9465 - val_loss: 0.8554 - val_acc: 0.7500
    Epoch 4/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1295 - acc: 0.9508Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.4039 - acc: 0.8750
    Epoch 00004: saving model to /content/data/model/weights.epoch_04.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.1301 - acc: 0.9505 - val_loss: 0.4039 - val_acc: 0.8750
    Epoch 5/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1242 - acc: 0.9564Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.4727 - acc: 0.8125
    Epoch 00005: saving model to /content/data/model/weights.epoch_05.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.1237 - acc: 0.9567 - val_loss: 0.4727 - val_acc: 0.8125
    Epoch 6/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1231 - acc: 0.9574Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2148 - acc: 0.8750
    Epoch 00006: saving model to /content/data/model/weights.epoch_06.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.1231 - acc: 0.9572 - val_loss: 0.2148 - val_acc: 0.8750
    Epoch 7/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1223 - acc: 0.9535Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1770 - acc: 0.9375
    Epoch 00007: saving model to /content/data/model/weights.epoch_07.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.1216 - acc: 0.9538 - val_loss: 0.1770 - val_acc: 0.9375
    Epoch 8/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1123 - acc: 0.9576Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.1544 - acc: 0.9375
    Epoch 00008: saving model to /content/data/model/weights.epoch_08.hdf5
    163/163 [==============================] - 86s 527ms/step - loss: 0.1119 - acc: 0.9576 - val_loss: 0.1544 - val_acc: 0.9375
    Epoch 9/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1011 - acc: 0.9585Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.1367 - acc: 0.9375
    Epoch 00009: saving model to /content/data/model/weights.epoch_09.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.1007 - acc: 0.9588 - val_loss: 0.1367 - val_acc: 0.9375
    Epoch 10/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0992 - acc: 0.9626Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1243 - acc: 0.9375
    Epoch 00010: saving model to /content/data/model/weights.epoch_10.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.1000 - acc: 0.9624 - val_loss: 0.1243 - val_acc: 0.9375
    Epoch 11/100
    162/163 [============================>.] - ETA: 0s - loss: 0.1025 - acc: 0.9616Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1493 - acc: 0.9375
    Epoch 00011: saving model to /content/data/model/weights.epoch_11.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.1022 - acc: 0.9617 - val_loss: 0.1493 - val_acc: 0.9375
    Epoch 12/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0990 - acc: 0.9601Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1211 - acc: 0.8750
    Epoch 00012: saving model to /content/data/model/weights.epoch_12.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0993 - acc: 0.9601 - val_loss: 0.1211 - val_acc: 0.8750
    Epoch 13/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0991 - acc: 0.9612Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1547 - acc: 0.9375
    Epoch 00013: saving model to /content/data/model/weights.epoch_13.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0995 - acc: 0.9613 - val_loss: 0.1547 - val_acc: 0.9375
    Epoch 14/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0935 - acc: 0.9651Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1807 - acc: 0.9375
    Epoch 00014: saving model to /content/data/model/weights.epoch_14.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0935 - acc: 0.9651 - val_loss: 0.1807 - val_acc: 0.9375
    Epoch 15/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0903 - acc: 0.9637Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1229 - acc: 0.9375
    Epoch 00015: saving model to /content/data/model/weights.epoch_15.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.0902 - acc: 0.9636 - val_loss: 0.1229 - val_acc: 0.9375
    Epoch 16/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0904 - acc: 0.9664Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.1215 - acc: 0.9375
    Epoch 00016: saving model to /content/data/model/weights.epoch_16.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0899 - acc: 0.9666 - val_loss: 0.1215 - val_acc: 0.9375
    Epoch 17/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0922 - acc: 0.9637Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.3688 - acc: 0.8125
    Epoch 00017: saving model to /content/data/model/weights.epoch_17.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0922 - acc: 0.9636 - val_loss: 0.3688 - val_acc: 0.8125
    Epoch 18/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0894 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1606 - acc: 0.9375
    Epoch 00018: saving model to /content/data/model/weights.epoch_18.hdf5
    163/163 [==============================] - 85s 524ms/step - loss: 0.0890 - acc: 0.9684 - val_loss: 0.1606 - val_acc: 0.9375
    Epoch 19/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0851 - acc: 0.9695Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1894 - acc: 0.8750
    Epoch 00019: saving model to /content/data/model/weights.epoch_19.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0850 - acc: 0.9695 - val_loss: 0.1894 - val_acc: 0.8750
    Epoch 20/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0860 - acc: 0.9688Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2548 - acc: 0.8750
    Epoch 00020: saving model to /content/data/model/weights.epoch_20.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.0860 - acc: 0.9686 - val_loss: 0.2548 - val_acc: 0.8750
    Epoch 21/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0882 - acc: 0.9662Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.3508 - acc: 0.7500
    Epoch 00021: saving model to /content/data/model/weights.epoch_21.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0877 - acc: 0.9664 - val_loss: 0.3508 - val_acc: 0.7500
    Epoch 22/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0787 - acc: 0.9709Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.1240 - acc: 0.9375
    Epoch 00022: saving model to /content/data/model/weights.epoch_22.hdf5
    163/163 [==============================] - 86s 529ms/step - loss: 0.0788 - acc: 0.9709 - val_loss: 0.1240 - val_acc: 0.9375
    Epoch 23/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0871 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1897 - acc: 0.9375
    Epoch 00023: saving model to /content/data/model/weights.epoch_23.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0867 - acc: 0.9684 - val_loss: 0.1897 - val_acc: 0.9375
    Epoch 24/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0806 - acc: 0.9720Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.1861 - acc: 0.9375
    Epoch 00024: saving model to /content/data/model/weights.epoch_24.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0806 - acc: 0.9718 - val_loss: 0.1861 - val_acc: 0.9375
    Epoch 25/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0777 - acc: 0.9709Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.0779 - acc: 0.9375
    Epoch 00025: saving model to /content/data/model/weights.epoch_25.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0786 - acc: 0.9707 - val_loss: 0.0779 - val_acc: 0.9375
    Epoch 26/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0848 - acc: 0.9674Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1038 - acc: 0.9375
    Epoch 00026: saving model to /content/data/model/weights.epoch_26.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0855 - acc: 0.9668 - val_loss: 0.1038 - val_acc: 0.9375
    Epoch 27/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0792 - acc: 0.9724Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1531 - acc: 0.8750
    Epoch 00027: saving model to /content/data/model/weights.epoch_27.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0788 - acc: 0.9726 - val_loss: 0.1531 - val_acc: 0.8750
    Epoch 28/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0796 - acc: 0.9703Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.2089 - acc: 0.9375
    Epoch 00028: saving model to /content/data/model/weights.epoch_28.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0806 - acc: 0.9701 - val_loss: 0.2089 - val_acc: 0.9375
    Epoch 29/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0831 - acc: 0.9682Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.0790 - acc: 1.0000
    Epoch 00029: saving model to /content/data/model/weights.epoch_29.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0826 - acc: 0.9684 - val_loss: 0.0790 - val_acc: 1.0000
    Epoch 30/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0822 - acc: 0.9701Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0625 - acc: 1.0000
    Epoch 00030: saving model to /content/data/model/weights.epoch_30.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0818 - acc: 0.9703 - val_loss: 0.0625 - val_acc: 1.0000
    Epoch 31/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0761 - acc: 0.9726Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1074 - acc: 1.0000
    Epoch 00031: saving model to /content/data/model/weights.epoch_31.hdf5
    163/163 [==============================] - 87s 537ms/step - loss: 0.0757 - acc: 0.9728 - val_loss: 0.1074 - val_acc: 1.0000
    Epoch 32/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0763 - acc: 0.9703Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.0581 - acc: 1.0000
    Epoch 00032: saving model to /content/data/model/weights.epoch_32.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0763 - acc: 0.9701 - val_loss: 0.0581 - val_acc: 1.0000
    Epoch 33/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0768 - acc: 0.9707Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1973 - acc: 0.8750
    Epoch 00033: saving model to /content/data/model/weights.epoch_33.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0766 - acc: 0.9707 - val_loss: 0.1973 - val_acc: 0.8750
    Epoch 34/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0793 - acc: 0.9722Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0914 - acc: 1.0000
    Epoch 00034: saving model to /content/data/model/weights.epoch_34.hdf5
    163/163 [==============================] - 87s 537ms/step - loss: 0.0792 - acc: 0.9722 - val_loss: 0.0914 - val_acc: 1.0000
    Epoch 35/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0749 - acc: 0.9709Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1380 - acc: 0.9375
    Epoch 00035: saving model to /content/data/model/weights.epoch_35.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0749 - acc: 0.9709 - val_loss: 0.1380 - val_acc: 0.9375
    Epoch 36/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0703 - acc: 0.9718Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1734 - acc: 0.8750
    Epoch 00036: saving model to /content/data/model/weights.epoch_36.hdf5
    163/163 [==============================] - 87s 537ms/step - loss: 0.0700 - acc: 0.9720 - val_loss: 0.1734 - val_acc: 0.8750
    Epoch 37/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0755 - acc: 0.9715Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.1433 - acc: 0.9375
    Epoch 00037: saving model to /content/data/model/weights.epoch_37.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.0757 - acc: 0.9712 - val_loss: 0.1433 - val_acc: 0.9375
    Epoch 38/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0751 - acc: 0.9732Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1892 - acc: 0.9375
    Epoch 00038: saving model to /content/data/model/weights.epoch_38.hdf5
    163/163 [==============================] - 88s 537ms/step - loss: 0.0754 - acc: 0.9732 - val_loss: 0.1892 - val_acc: 0.9375
    Epoch 39/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0752 - acc: 0.9722Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.0646 - acc: 1.0000
    Epoch 00039: saving model to /content/data/model/weights.epoch_39.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0748 - acc: 0.9724 - val_loss: 0.0646 - val_acc: 1.0000
    Epoch 40/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0698 - acc: 0.9745Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.4798 - acc: 0.7500
    Epoch 00040: saving model to /content/data/model/weights.epoch_40.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0700 - acc: 0.9745 - val_loss: 0.4798 - val_acc: 0.7500
    Epoch 41/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0742 - acc: 0.9718Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.0673 - acc: 0.9375
    Epoch 00041: saving model to /content/data/model/weights.epoch_41.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0738 - acc: 0.9720 - val_loss: 0.0673 - val_acc: 0.9375
    Epoch 42/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0669 - acc: 0.9730Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.0433 - acc: 1.0000
    Epoch 00042: saving model to /content/data/model/weights.epoch_42.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0669 - acc: 0.9730 - val_loss: 0.0433 - val_acc: 1.0000
    Epoch 43/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0704 - acc: 0.9732Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0838 - acc: 1.0000
    Epoch 00043: saving model to /content/data/model/weights.epoch_43.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0701 - acc: 0.9734 - val_loss: 0.0838 - val_acc: 1.0000
    Epoch 44/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0802 - acc: 0.9699Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0449 - acc: 1.0000
    Epoch 00044: saving model to /content/data/model/weights.epoch_44.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0798 - acc: 0.9701 - val_loss: 0.0449 - val_acc: 1.0000
    Epoch 45/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0697 - acc: 0.9734Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0499 - acc: 1.0000
    Epoch 00045: saving model to /content/data/model/weights.epoch_45.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0695 - acc: 0.9735 - val_loss: 0.0499 - val_acc: 1.0000
    Epoch 46/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0701 - acc: 0.9720Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1592 - acc: 0.8750
    Epoch 00046: saving model to /content/data/model/weights.epoch_46.hdf5
    163/163 [==============================] - 88s 541ms/step - loss: 0.0698 - acc: 0.9720 - val_loss: 0.1592 - val_acc: 0.8750
    Epoch 47/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0718 - acc: 0.9715Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.0679 - acc: 1.0000
    Epoch 00047: saving model to /content/data/model/weights.epoch_47.hdf5
    163/163 [==============================] - 89s 544ms/step - loss: 0.0714 - acc: 0.9716 - val_loss: 0.0679 - val_acc: 1.0000
    Epoch 48/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0682 - acc: 0.9728Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1295 - acc: 0.8750
    Epoch 00048: saving model to /content/data/model/weights.epoch_48.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.0689 - acc: 0.9726 - val_loss: 0.1295 - val_acc: 0.8750
    Epoch 49/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0733 - acc: 0.9726Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0895 - acc: 0.9375
    Epoch 00049: saving model to /content/data/model/weights.epoch_49.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0731 - acc: 0.9726 - val_loss: 0.0895 - val_acc: 0.9375
    Epoch 50/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0686 - acc: 0.9743Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.1431 - acc: 0.9375
    Epoch 00050: saving model to /content/data/model/weights.epoch_50.hdf5
    163/163 [==============================] - 86s 531ms/step - loss: 0.0686 - acc: 0.9743 - val_loss: 0.1431 - val_acc: 0.9375
    Epoch 51/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0688 - acc: 0.9734Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.1358 - acc: 0.9375
    Epoch 00051: saving model to /content/data/model/weights.epoch_51.hdf5
    163/163 [==============================] - 86s 526ms/step - loss: 0.0687 - acc: 0.9732 - val_loss: 0.1358 - val_acc: 0.9375
    Epoch 52/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0702 - acc: 0.9740Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.2695 - acc: 0.8125
    Epoch 00052: saving model to /content/data/model/weights.epoch_52.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0705 - acc: 0.9737 - val_loss: 0.2695 - val_acc: 0.8125
    Epoch 53/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0664 - acc: 0.9751Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0418 - acc: 1.0000
    Epoch 00053: saving model to /content/data/model/weights.epoch_53.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0661 - acc: 0.9753 - val_loss: 0.0418 - val_acc: 1.0000
    Epoch 54/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0659 - acc: 0.9755Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.0923 - acc: 0.9375
    Epoch 00054: saving model to /content/data/model/weights.epoch_54.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0659 - acc: 0.9755 - val_loss: 0.0923 - val_acc: 0.9375
    Epoch 55/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0629 - acc: 0.9778Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.1823 - acc: 0.8750
    Epoch 00055: saving model to /content/data/model/weights.epoch_55.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.0633 - acc: 0.9778 - val_loss: 0.1823 - val_acc: 0.8750
    Epoch 56/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0696 - acc: 0.9738Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.2149 - acc: 0.9375
    Epoch 00056: saving model to /content/data/model/weights.epoch_56.hdf5
    163/163 [==============================] - 88s 540ms/step - loss: 0.0699 - acc: 0.9735 - val_loss: 0.2149 - val_acc: 0.9375
    Epoch 57/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0674 - acc: 0.9755Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.0913 - acc: 1.0000
    Epoch 00057: saving model to /content/data/model/weights.epoch_57.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0674 - acc: 0.9755 - val_loss: 0.0913 - val_acc: 1.0000
    Epoch 58/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0683 - acc: 0.9765Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.0494 - acc: 1.0000
    Epoch 00058: saving model to /content/data/model/weights.epoch_58.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0694 - acc: 0.9762 - val_loss: 0.0494 - val_acc: 1.0000
    Epoch 59/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0671 - acc: 0.9755Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0989 - acc: 1.0000
    Epoch 00059: saving model to /content/data/model/weights.epoch_59.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0669 - acc: 0.9757 - val_loss: 0.0989 - val_acc: 1.0000
    Epoch 60/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0672 - acc: 0.9738Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0486 - acc: 1.0000
    Epoch 00060: saving model to /content/data/model/weights.epoch_60.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0669 - acc: 0.9739 - val_loss: 0.0486 - val_acc: 1.0000
    Epoch 61/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0659 - acc: 0.9770Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0884 - acc: 1.0000
    Epoch 00061: saving model to /content/data/model/weights.epoch_61.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0656 - acc: 0.9772 - val_loss: 0.0884 - val_acc: 1.0000
    Epoch 62/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0616 - acc: 0.9753Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0403 - acc: 1.0000
    Epoch 00062: saving model to /content/data/model/weights.epoch_62.hdf5
    163/163 [==============================] - 88s 537ms/step - loss: 0.0612 - acc: 0.9755 - val_loss: 0.0403 - val_acc: 1.0000
    Epoch 63/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0672 - acc: 0.9747Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0395 - acc: 1.0000
    Epoch 00063: saving model to /content/data/model/weights.epoch_63.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0669 - acc: 0.9749 - val_loss: 0.0395 - val_acc: 1.0000
    Epoch 64/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0650 - acc: 0.9759Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0335 - acc: 1.0000
    Epoch 00064: saving model to /content/data/model/weights.epoch_64.hdf5
    163/163 [==============================] - 86s 531ms/step - loss: 0.0646 - acc: 0.9760 - val_loss: 0.0335 - val_acc: 1.0000
    Epoch 65/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0661 - acc: 0.9749Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0741 - acc: 1.0000
    Epoch 00065: saving model to /content/data/model/weights.epoch_65.hdf5
    163/163 [==============================] - 86s 528ms/step - loss: 0.0658 - acc: 0.9751 - val_loss: 0.0741 - val_acc: 1.0000
    Epoch 66/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0625 - acc: 0.9751Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0522 - acc: 1.0000
    Epoch 00066: saving model to /content/data/model/weights.epoch_66.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0625 - acc: 0.9751 - val_loss: 0.0522 - val_acc: 1.0000
    Epoch 67/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0681 - acc: 0.9778Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0501 - acc: 1.0000
    Epoch 00067: saving model to /content/data/model/weights.epoch_67.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.0678 - acc: 0.9780 - val_loss: 0.0501 - val_acc: 1.0000
    Epoch 68/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0651 - acc: 0.9778Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0337 - acc: 1.0000
    Epoch 00068: saving model to /content/data/model/weights.epoch_68.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0648 - acc: 0.9780 - val_loss: 0.0337 - val_acc: 1.0000
    Epoch 69/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0629 - acc: 0.9765Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0396 - acc: 1.0000
    Epoch 00069: saving model to /content/data/model/weights.epoch_69.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0626 - acc: 0.9766 - val_loss: 0.0396 - val_acc: 1.0000
    Epoch 70/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0592 - acc: 0.9757Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0529 - acc: 1.0000
    Epoch 00070: saving model to /content/data/model/weights.epoch_70.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0589 - acc: 0.9758 - val_loss: 0.0529 - val_acc: 1.0000
    Epoch 71/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0620 - acc: 0.9780Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0662 - acc: 1.0000
    Epoch 00071: saving model to /content/data/model/weights.epoch_71.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0617 - acc: 0.9781 - val_loss: 0.0662 - val_acc: 1.0000
    Epoch 72/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0620 - acc: 0.9786Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0431 - acc: 1.0000
    Epoch 00072: saving model to /content/data/model/weights.epoch_72.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0618 - acc: 0.9787 - val_loss: 0.0431 - val_acc: 1.0000
    Epoch 73/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0648 - acc: 0.9765Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0710 - acc: 1.0000
    Epoch 00073: saving model to /content/data/model/weights.epoch_73.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0648 - acc: 0.9764 - val_loss: 0.0710 - val_acc: 1.0000
    Epoch 74/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0655 - acc: 0.9763Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0354 - acc: 1.0000
    Epoch 00074: saving model to /content/data/model/weights.epoch_74.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0651 - acc: 0.9764 - val_loss: 0.0354 - val_acc: 1.0000
    Epoch 75/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0682 - acc: 0.9761Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.2922 - acc: 0.8750
    Epoch 00075: saving model to /content/data/model/weights.epoch_75.hdf5
    163/163 [==============================] - 87s 532ms/step - loss: 0.0687 - acc: 0.9758 - val_loss: 0.2922 - val_acc: 0.8750
    Epoch 76/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0711 - acc: 0.9734Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0363 - acc: 1.0000
    Epoch 00076: saving model to /content/data/model/weights.epoch_76.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0707 - acc: 0.9735 - val_loss: 0.0363 - val_acc: 1.0000
    Epoch 77/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0657 - acc: 0.9769Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0367 - acc: 1.0000
    Epoch 00077: saving model to /content/data/model/weights.epoch_77.hdf5
    163/163 [==============================] - 87s 533ms/step - loss: 0.0660 - acc: 0.9766 - val_loss: 0.0367 - val_acc: 1.0000
    Epoch 78/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0610 - acc: 0.9811Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0346 - acc: 1.0000
    Epoch 00078: saving model to /content/data/model/weights.epoch_78.hdf5
    163/163 [==============================] - 87s 531ms/step - loss: 0.0608 - acc: 0.9812 - val_loss: 0.0346 - val_acc: 1.0000
    Epoch 79/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0591 - acc: 0.9782Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0830 - acc: 0.9375
    Epoch 00079: saving model to /content/data/model/weights.epoch_79.hdf5
    163/163 [==============================] - 86s 530ms/step - loss: 0.0587 - acc: 0.9783 - val_loss: 0.0830 - val_acc: 0.9375
    Epoch 80/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0641 - acc: 0.9738Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0763 - acc: 1.0000
    Epoch 00080: saving model to /content/data/model/weights.epoch_80.hdf5
    163/163 [==============================] - 87s 536ms/step - loss: 0.0637 - acc: 0.9739 - val_loss: 0.0763 - val_acc: 1.0000
    Epoch 81/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0575 - acc: 0.9786Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0510 - acc: 1.0000
    Epoch 00081: saving model to /content/data/model/weights.epoch_81.hdf5
    163/163 [==============================] - 87s 537ms/step - loss: 0.0578 - acc: 0.9785 - val_loss: 0.0510 - val_acc: 1.0000
    Epoch 82/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0547 - acc: 0.9788Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0347 - acc: 1.0000
    Epoch 00082: saving model to /content/data/model/weights.epoch_82.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0548 - acc: 0.9787 - val_loss: 0.0347 - val_acc: 1.0000
    Epoch 83/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0550 - acc: 0.9801Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.0282 - acc: 1.0000
    Epoch 00083: saving model to /content/data/model/weights.epoch_83.hdf5
    163/163 [==============================] - 87s 535ms/step - loss: 0.0548 - acc: 0.9803 - val_loss: 0.0282 - val_acc: 1.0000
    Epoch 84/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0636 - acc: 0.9772Epoch 1/100
      1/163 [..............................] - ETA: 58s - loss: 0.0374 - acc: 1.0000
    Epoch 00084: saving model to /content/data/model/weights.epoch_84.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0637 - acc: 0.9772 - val_loss: 0.0374 - val_acc: 1.0000
    Epoch 85/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0544 - acc: 0.9788Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0228 - acc: 1.0000
    Epoch 00085: saving model to /content/data/model/weights.epoch_85.hdf5
    163/163 [==============================] - 88s 539ms/step - loss: 0.0542 - acc: 0.9789 - val_loss: 0.0228 - val_acc: 1.0000
    Epoch 86/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0602 - acc: 0.9776Epoch 1/100
      1/163 [..............................] - ETA: 53s - loss: 0.0334 - acc: 1.0000
    Epoch 00086: saving model to /content/data/model/weights.epoch_86.hdf5
    163/163 [==============================] - 87s 534ms/step - loss: 0.0599 - acc: 0.9778 - val_loss: 0.0334 - val_acc: 1.0000
    Epoch 87/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0678 - acc: 0.9726Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0555 - acc: 1.0000
    Epoch 00087: saving model to /content/data/model/weights.epoch_87.hdf5
    163/163 [==============================] - 88s 538ms/step - loss: 0.0689 - acc: 0.9720 - val_loss: 0.0555 - val_acc: 1.0000
    Epoch 88/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0592 - acc: 0.9790Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0570 - acc: 1.0000
    Epoch 00088: saving model to /content/data/model/weights.epoch_88.hdf5
    163/163 [==============================] - 89s 547ms/step - loss: 0.0589 - acc: 0.9791 - val_loss: 0.0570 - val_acc: 1.0000
    Epoch 89/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0624 - acc: 0.9767Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0185 - acc: 1.0000
    Epoch 00089: saving model to /content/data/model/weights.epoch_89.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0622 - acc: 0.9768 - val_loss: 0.0185 - val_acc: 1.0000
    Epoch 90/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0569 - acc: 0.9801Epoch 1/100
      1/163 [..............................] - ETA: 54s - loss: 0.0194 - acc: 1.0000
    Epoch 00090: saving model to /content/data/model/weights.epoch_90.hdf5
    163/163 [==============================] - 88s 542ms/step - loss: 0.0566 - acc: 0.9803 - val_loss: 0.0194 - val_acc: 1.0000
    Epoch 91/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0576 - acc: 0.9792Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0355 - acc: 1.0000
    Epoch 00091: saving model to /content/data/model/weights.epoch_91.hdf5
    163/163 [==============================] - 88s 543ms/step - loss: 0.0575 - acc: 0.9793 - val_loss: 0.0355 - val_acc: 1.0000
    Epoch 92/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0651 - acc: 0.9761Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0418 - acc: 1.0000
    Epoch 00092: saving model to /content/data/model/weights.epoch_92.hdf5
    163/163 [==============================] - 89s 544ms/step - loss: 0.0648 - acc: 0.9762 - val_loss: 0.0418 - val_acc: 1.0000
    Epoch 93/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0603 - acc: 0.9765Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0679 - acc: 1.0000
    Epoch 00093: saving model to /content/data/model/weights.epoch_93.hdf5
    163/163 [==============================] - 89s 548ms/step - loss: 0.0601 - acc: 0.9766 - val_loss: 0.0679 - val_acc: 1.0000
    Epoch 94/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0567 - acc: 0.9799Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0181 - acc: 1.0000
    Epoch 00094: saving model to /content/data/model/weights.epoch_94.hdf5
    163/163 [==============================] - 89s 546ms/step - loss: 0.0568 - acc: 0.9799 - val_loss: 0.0181 - val_acc: 1.0000
    Epoch 95/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0600 - acc: 0.9784Epoch 1/100
      1/163 [..............................] - ETA: 1:00 - loss: 0.0219 - acc: 1.0000
    Epoch 00095: saving model to /content/data/model/weights.epoch_95.hdf5
    163/163 [==============================] - 89s 547ms/step - loss: 0.0599 - acc: 0.9783 - val_loss: 0.0219 - val_acc: 1.0000
    Epoch 96/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0541 - acc: 0.9794Epoch 1/100
      1/163 [..............................] - ETA: 57s - loss: 0.0304 - acc: 1.0000
    Epoch 00096: saving model to /content/data/model/weights.epoch_96.hdf5
    163/163 [==============================] - 89s 545ms/step - loss: 0.0540 - acc: 0.9795 - val_loss: 0.0304 - val_acc: 1.0000
    Epoch 97/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0550 - acc: 0.9807Epoch 1/100
      1/163 [..............................] - ETA: 59s - loss: 0.0389 - acc: 1.0000
    Epoch 00097: saving model to /content/data/model/weights.epoch_97.hdf5
    163/163 [==============================] - 89s 547ms/step - loss: 0.0547 - acc: 0.9808 - val_loss: 0.0389 - val_acc: 1.0000
    Epoch 98/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0595 - acc: 0.9780Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0282 - acc: 1.0000
    Epoch 00098: saving model to /content/data/model/weights.epoch_98.hdf5
    163/163 [==============================] - 89s 548ms/step - loss: 0.0592 - acc: 0.9781 - val_loss: 0.0282 - val_acc: 1.0000
    Epoch 99/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0593 - acc: 0.9759Epoch 1/100
      1/163 [..............................] - ETA: 55s - loss: 0.0730 - acc: 0.9375
    Epoch 00099: saving model to /content/data/model/weights.epoch_99.hdf5
    163/163 [==============================] - 90s 551ms/step - loss: 0.0591 - acc: 0.9760 - val_loss: 0.0730 - val_acc: 0.9375
    Epoch 100/100
    162/163 [============================>.] - ETA: 0s - loss: 0.0603 - acc: 0.9794Epoch 1/100
      1/163 [..............................] - ETA: 56s - loss: 0.0189 - acc: 1.0000
    Epoch 00100: saving model to /content/data/model/weights.epoch_100.hdf5
    163/163 [==============================] - 90s 550ms/step - loss: 0.0601 - acc: 0.9795 - val_loss: 0.0189 - val_acc: 1.0000
    20/20 [==============================] - 8s 383ms/step - loss: 0.3198 - acc: 0.9087
    Loading the best model
    epoch: 94, val_loss: 0.01814509741961956, val_acc: 1.0
    20/20 [==============================] - 7s 363ms/step - loss: 0.3109 - acc: 0.9167
    20/20 [==============================] - 8s 394ms/step
    CONFUSION MATRIX ------------------
    [[201  33]
     [ 19 371]]
    
    TEST METRICS ----------------------
    Accuracy: 91.66666666666666%
    Precision: 91.83168316831683%
    Recall: 95.12820512820512%
    F1-score: 93.45088161209067
    
    TRAIN METRIC ----------------------
    Train acc: 97.94861674308775%
    


![png](VGG16%20Model%201%20Version%201.1.0.0.0_files/VGG16%20Model%201%20Version%201.1.0.0.0_4_1.png)



```
## Load best weight
idx = np.argmin(history.history['acc']) 
model.load_weights("/content/data/model/weights.epoch_{:02d}.hdf5".format(idx + 1))

print("Loading the best model")
print("epoch: {}, val_loss: {}, val_acc: {}".format(idx + 1, history.history['val_loss'][idx], history.history['val_acc'][idx]))

## Evaluate the best weight
test_loss, test_acc = model.evaluate_generator(generator=test_generator, verbose=1)

## Test analytics
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

    Loading the best model
    epoch: 1, val_loss: 0.5660557150840759, val_acc: 0.75
    20/20 [==============================] - 7s 349ms/step - loss: 0.3333 - acc: 0.8622
    20/20 [==============================] - 7s 365ms/step
    CONFUSION MATRIX ------------------
    [[154  80]
     [  6 384]]
    
    TEST METRICS ----------------------
    Accuracy: 86.21794871794873%
    Precision: 82.75862068965517%
    Recall: 98.46153846153847%
    F1-score: 89.9297423887588
    
    TRAIN METRIC ----------------------
    Train acc: 97.94861674308775%
    
