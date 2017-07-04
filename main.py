from __future__ import print_function
import numpy as np
import pickle
import os

os.environ['KERAS_BACKEND'] = 'theano'
from load import *
from metrics.accuracy import conlleval
from load import *
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D
from accuracy import conlleval
import progressbar



### Load Data
train_set, valid_set,test_set,dicts = load_dataset(2)
w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']

# get index to word dict
idx2w  = get_id_to_word_dict(dicts)
idx2la = get_id_to_label_dict(dicts)


### Model
n_classes = len(idx2la)
n_vocab = len(idx2w)

# Define model
model = Sequential()
model.add(Embedding(n_vocab,100))
model.add(Convolution1D(64,5,border_mode='same', activation='relu'))
model.add(Dropout(0.25))
model.add(LSTM(100,return_sequences=True))
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
model.compile('rmsprop', 'categorical_crossentropy')


train_x, train_ne, train_label = train_set
val_x, val_ne, val_label = valid_set
test_x, test_ne, test_label = test_set
validationX = [ list(map(lambda x: idx2w[x], w)) for w in val_x]
validationY = [ list(map(lambda x: idx2la[x], y)) for y in val_label]
trainX = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
trainY = [ list(map(lambda x: idx2la[x], y)) for y in train_label]
testX = [ list(map(lambda x: idx2w[x], w)) for w in test_x]
testY = [ list(map(lambda x: idx2la[x], y)) for y in test_label]

### Training
n_epochs = 100

train_f_scores = []
val_f_scores = []
test_f_scores = []
best_val_f1 = 0

for i in range(n_epochs):
    print("Epoch {}".format(i))
    
    print("Training =>")
    train_pred_label = []
    avgLoss = 0
        
    bar = progressbar.ProgressBar(max_value=len(train_x))
    for n_batch, sent in bar(enumerate(train_x)):
        label = train_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis,:]
        sent = sent[np.newaxis,:]
        #print(sent)
        if sent.shape[1] > 1: #some bug in keras
            loss = model.train_on_batch(sent, label)
            avgLoss += loss

        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred,-1)[0]
        #print(pred)
        train_pred_label.append(pred)

    avgLoss = avgLoss/n_batch
    predword_train = [ list(map(lambda x: idx2la[x], y)) for y in train_pred_label]
    con_dict = conlleval(predword_train, trainY, trainY , 'r.txt')
    train_f_scores.append(con_dict['f1'])
    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))

    print("Validating =>")
    
    val_pred_label = []
    avgLoss = 0
    
    bar = progressbar.ProgressBar(max_value=len(val_x))
    for n_batch, sent in bar(enumerate(val_x)):
        label = val_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis,:]
        sent = sent[np.newaxis,:]
        
        if sent.shape[1] > 1: #some bug in keras
            loss = model.test_on_batch(sent, label)
            avgLoss += loss

        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred,-1)[0]
        val_pred_label.append(pred)

    avgLoss = avgLoss/n_batch
    
    predword_val = [ list(map(lambda x: idx2la[x], y)) for y in val_pred_label]
    con_dict = conlleval(predword_val, validationY, validationX, 'r.txt')
    val_f_scores.append(con_dict['f1'])
    
    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))

    if con_dict['f1'] > best_val_f1:
        best_val_f1 = con_dict['f1']
        open('model_architecture.json','w').write(model.to_json())
        model.save_weights('best_model_weights.h5',overwrite=True)
        print("Best validation F1 score = {}".format(best_val_f1))
    print()

    print("Testing =>")
    
    test_pred_label = []
    avgLoss = 0
    
    bar = progressbar.ProgressBar(max_value=len(test_x))
    for n_batch, sent in bar(enumerate(test_x)):
        label = test_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis,:]
        sent = sent[np.newaxis,:]
        
        if sent.shape[1] > 1: #some bug in keras
            loss = model.test_on_batch(sent, label)
            avgLoss += loss

        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred,-1)[0]
        test_pred_label.append(pred)

    avgLoss = avgLoss/n_batch
    
    predword_test = [ list(map(lambda x: idx2la[x], y)) for y in test_pred_label]
    con_dict = conlleval(predword_val, testY, testX, 'r.txt')
    val_f_scores.append(con_dict['f1'])
    
    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))

    if con_dict['f1'] > best_val_f1:
        best_val_f1 = con_dict['f1']
        open('model_architecture.json','w').write(model.to_json())
        model.save_weights('best_model_weights.h5',overwrite=True)
        print("Best test F1 score = {}".format(best_val_f1))
    print()
