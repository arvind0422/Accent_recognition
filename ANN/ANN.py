
# coding: utf-8

# In[1]:


#2#


# In[2]:


from __future__ import division
import numpy as np
import keras
import os
import sklearn.metrics as skm
import pandas as pd
from keras.layers import Dense, Dropout, Input, Activation, Softmax
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, load_model, save_model
from keras.utils import plot_model
from random import shuffle


# In[9]:


# Printing stuff in console for debugging purposes.
import sys
import os
try:
    os.system("rm log.txt")
    sys.stdout = open("log.txt",'w')
except:
    sys.stdout = open("log.txt", 'w')


# In[4]:


# Model architecture is defined here.
def get_model(feat_shape=200,num_accents=5):
    
    input1 = Input(shape=(feat_shape,),name="input")
    
    dense = Dense(feat_shape)(input1)
    dense = BatchNormalization()(dense)
    dense = Activation("relu")(dense)
    
    dense = Dropout(0.5)(dense)
    dense = Dense(feat_shape)(dense)
    dense = BatchNormalization()(dense)
    dense = Activation("relu")(dense)
    
    dense = Dropout(0.25)(dense)
    dense = Dense(int(feat_shape/2))(dense)
    dense = BatchNormalization()(dense)
    dense = Activation("relu")(dense)
    
    dense = Dropout(0)(dense)
    dense = Dense(int(feat_shape/4))(dense)
    dense = BatchNormalization()(dense)
    dense = Activation("relu")(dense)
    
    dense = Dense(num_accents)(dense)
    output = Softmax(axis=0,name="output")(dense)
    print(output.shape)
    
    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# In[5]:


num_accents=5
model = get_model(200,num_accents)


# In[6]:


plot_model(model,"model.png")
model.summary()


# In[7]:


batch_size = 1
num_epochs = 3

old_metric = -1 # Depends. Either very low or very high initial value.
patience = 1
glob_count=0

training_files = open("trn_fstat.list").readlines()
num_training_samples = len(training_files)
print("Number of training samples: "+str(num_training_samples))

# Load Testing Data.
testing_files = open("tst_fstat.list").readlines()
num_testing_samples = len(testing_files)
print("Number of testing samples: "+str(num_testing_samples))
test_in = []
test_labels = []
for i in range(num_testing_samples):
    f = testing_files[i].split() # Asssuming the same format as in seniors' ann.py which uses the list file.
    f_in = open(f[0],"r")
    test_in.append(np.fromfile(f_in, dtype=np.double))
    test_labels.append(int(f[1]))
test_in = np.asarray(test_in)
test_labels = np.asarray(test_labels)

while True: 
    
    # List file creation
    """
    Ideally, we want a list array which is shuffled and can be read in by the training script. So, based on the
    batch_size, we create a new list file that has the shuffled data. That must be done here.

    Maybe we can have an outer loop to control which particular label dominates the training data - equivalent to
    fine tuning for each label. We can maybe do this if accuracy is low after basic training.

    This makes changes to training_files
    """
    
    for epoch_no in range(num_epochs):
        
        # For now, it is just shuffling the input data.
        for __shuff__ in range(3):
            shuffle(training_files)
        
        glob_count += 1 # Keeping track of total number of epochs.
        print("Epoch number: "+str(glob_count))
        
        # The following for loop running once is equivalent to one epoch on the whole training data.
        for i in range(int(num_training_samples/batch_size)):

            # Get a mini-batch of file names from the training files.
            try
                this_batch_files = training_files[batch_size*i:(batch_size*i)+batch_size]
            except:
                this_batch_files = training_files

            # Load data for the particular mini-batch.
            mini_batch_in = []
            mini_batch_labels = []
            for i in range(batch_size):
                f = this_batch_files[i].split() # Split into file name and label
                f_in=open(f[0],"r")
                mini_batch_in.append(np.fromfile(f_in, dtype=np.double))
                mini_batch_labels.append(int(f[1]))
            mini_batch_in = np.asarray(mini_batch_in)
            mini_batch_labels = np.asarray(mini_batch_labels)
            if (i%100==0):
                print("Current status (every 100 mini-batches)")
                model.fit(mini_batch_in, mini_batch_labels, epochs=1, batch_size=len(this_batch_files), verbose=1)
            else:
                model.fit(mini_batch_in, mini_batch_labels, epochs=1, batch_size=len(this_batch_files), verbose=0)
                
    # Validation after a few epochs
    test_pred = np.argmax(model.predict(test_in, batch_size=32, verbose=0),axis=1)
    confusion_matrix = skm.confusion_matrix(test_labels, test_pred, labels=np.arange(num_accents)) # The last value 5 is number of accents
    print("Validation Confusion Matrix")
    print(confusion_matrix)
    new_metric = skm.accuracy_score(test_labels, test_pred)
    print("Validation accuracy")
    print(new_metric)

    # Define early stopping criteria
    if old_metric < new_metric:
        old_metric = new_metric
        model.save_weights("accent.hdf5")
    else:
        # Load model
        model.load_weights("accent.hdf5")
        patience += 1
        print("Patience: "+str(patience))
    if patience == 3:
        print("Final confusion matrix")
        print(confusion_matrix)
        print("Final validation accuracy")
        print(new_metric)
        print("Done Training")
        break

