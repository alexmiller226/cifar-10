#!/usr/bin/env python
# coding: utf-8

# # CIFAR-10 CNN Model

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import csv,sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, preprocessing


# ## Config for notebook

# In[2]:


config = {'model_save_path': '/Users/alexmiller/Documents/cifar-10/cifar-10-cnn-model-dropout',
          'checkpoint_path': 'cifar-10-cnn/cp.ckpt',
          'class_names_list': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 
                               'horse', 'ship', 'truck'],
          'optimizer': 'adam',
          'loss': 'sparse_categorical_crossentropy',
          'metrics': 'sparse_categorical_accuracy',
          'epochs': 10,
          'batch_size': 64
          }


# ## Functions

# In[3]:


def plot_history(r, accuracy_metric):
    plt.plot(r.history[accuracy_metric], label = 'accuracy', color = 'red')
    plt.plot(r.history['val_'+accuracy_metric], label = 'val_acc', color = 'green')
    plt.legend()
    plt.show()


# ## Load dataset and scale training data

# In[4]:


cifar10 = tf.keras.datasets.cifar10
 
# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# In[5]:


x_train = x_train/255.0
x_test = x_test/255.0

print(f"Train data min: {np.min(x_train)} and max: {np.max(x_train)}")
print(f"Test data min: {np.min(x_test)} and max: {np.max(x_test)}")
print(f"Train labels shape: {x_train.shape}")
print(f"Test labels shape: {x_test.shape}")


# ## Visualize images

# In[6]:


plt.imshow(x_train[0], aspect = 'auto')


# In[7]:


rows = 5
cols = 5
k = 0

fig, ax = plt.subplots(nrows = rows, ncols = cols)

for i in range(rows):
    for j in range(cols):
        image = np.reshape(x_train[k], (32, 32, 3))
        ax[i][j].imshow(image, aspect = 'auto')
        k += 1

plt.show()


# ## CNN Model 1

# In[8]:


K = len(np.unique(x_train))
print("Number of classes:", K)

# Build model
model = models.Sequential([
    layers.Conv2D(32, 3, padding = 'same', activation = 'relu',
                  input_shape = [x_train.shape[1], x_train.shape[2], x_train.shape[3]]),
    layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(2, 2, padding = 'valid'),
    layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(2, 2, padding = 'valid'),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(K, activation = 'softmax')
])

# Print model summary
model.summary()


# In[9]:


# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])


# In[10]:


# Fit model
r = model.fit(x_train, y_train, 
              validation_data=(x_test, y_test), epochs=10)


# In[11]:


# Plot accuracy and validation accuracy by epoch
plot_history(r, 'sparse_categorical_accuracy')


# In[12]:


# Accuracy for test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss} Test accuracy: {test_accuracy}")


# In[13]:


# Predict probability on test set
y_pred = model.predict(x_test)
print(f"Test probability predictions shape: {y_pred.shape}")

# Convert probabilities to labels
y_pred_class = np.argmax(y_pred, axis=-1)
print(f"Test class predictions shape: {y_pred_class.shape}")


# In[15]:


from sklearn.metrics import roc_auc_score, roc_curve, classification_report

print(classification_report(y_test, y_pred_class, target_names = config['class_names_list']))


# ## CNN Model 2

# In[16]:


K = len(np.unique(x_train))
print("Number of classes:", K)

model2 = models.Sequential([
    layers.Conv2D(32, 3, padding = 'same', activation = 'relu',
                  input_shape = [x_train.shape[1], x_train.shape[2], x_train.shape[3]]),
    layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(2, 2, padding = 'valid'),
    layers.Dropout(0.2),
    layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(2, 2, padding = 'valid'),
    layers.Dropout(0.2),
    layers.Conv2D(128, 3, padding = 'same', activation = 'relu'),
    layers.Conv2D(128, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(2, 2, padding = 'valid'),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.2),
    layers.Dense(K, activation = 'softmax')
])

model2.summary()


# In[22]:


model2.compile(optimizer=config['optimizer'],
               loss=config['loss'],
               metrics=[config['metrics']])


# ### Save model checkpoints

# In[23]:


checkpoint_path = config['checkpoint_path']
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
r2 = model2.fit(x_train, 
                y_train,  
                epochs=config['epochs'],
                batch_size = config['batch_size'],
                validation_data=(x_test, y_test),
                callbacks=[cp_callback])  # Pass callback to training


# ### Plot history of model

# In[24]:


plot_history(r2, config['metrics'])


# In[25]:


# Accuracy on test set
test_loss, test_accuracy = model2.evaluate(x_test, y_test)
print(f"Test loss: {test_loss} Test accuracy: {test_accuracy}")


# In[26]:


y_pred = model.predict(x_test)
print(f"Test probability predictions shape: {y_pred.shape}")
# y_pred_class = model.predict_classes(x_test)
y_pred_class = np.argmax(y_pred, axis=-1)
print(f"Test class predictions shape: {y_pred_class.shape}")


# In[27]:


print(classification_report(y_test, y_pred_class, target_names = config['class_names_list']))


# ### Save final model

# In[28]:


model_save_path = config['model_save_path']

model2.save(model_save_path)

loaded_model = tf.keras.models.load_model(model_save_path)

# Check its architecture
loaded_model.summary()


# In[29]:


test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
print(f"Test loss {test_loss} Test accuracy {test_acc}")

