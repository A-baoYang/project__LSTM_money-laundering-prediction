import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, LSTM
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import SparseCategoricalCrossentropy

### Load data & preprocessing
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observ_daterange', type=str, required=True,
                        help='Observation date range of the transactions.')
    parser.add_argument('--label_daterange', type=str, required=True,
                        help='Labeled duration of actor_ids.')
    parser.add_argument('--try_date', type=str, required=True,
                        help='Whats the date today?')
    parser.add_argument('--version', type=str, required=False,
                        help='data format versions.')
    parser.add_argument('--desc', type=str, required=False,
                        help='description of the dataset')
    args = parser.parse_args()
    return args

args = parse_arguments()
observ_daterange = args.observ_daterange
label_daterange = args.label_daterange
try_date = args.try_date
version = args.version
desc = args.desc

train_X_res = np.load('OverSamp__train_x__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc))
train_Y_res = np.load('OverSamp__train_y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc))
val_X_res = np.load('OverSamp__val_x__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc))
val_Y_res = np.load('OverSamp__val_y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc))
test_X = np.load('test_X__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc))
test_Y = np.load('test_Y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc))


### Custom Metrics
class Metrics(Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or ()
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(' - val_f1: %f - val_precision: %f - val_recall: %f' % (_val_f1, _val_precision, _val_recall))
        return


### LSTM Model Structure
model = Sequential()
model.add(LSTM(units=100, input_shape=(train_X_res.shape[1], train_X_res.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=300, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=300, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=500, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=500, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(3))
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

ck_callback = ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_f1:.4f}.hdf5',
                              monitor='val_f1', mode='max', verbose=2, save_best_only=True, save_weights_only=True)
tb_callback = TensorBoard(log_dir='./logs', profile_batch=0)
er_callback = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto')


### Start Training !
sampling_method = 'over-sampling'
epochs = 20
batch_size = 512
history = model.fit(train_X_res, train_Y_res, epochs=epochs, batch_size=batch_size,
                    validation_data=(val_X_res, val_Y_res),
                    callbacks=[Metrics(valid_data=(val_X_res, val_Y_res)), ck_callback, tb_callback])

### Plot history
#### Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='best')
plt.show()
#### Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='best')
plt.show()
#### F1/Recall
plt.plot(history.history['val_f1'])
plt.plot(history.history['val_recall'])
plt.title('model f1/recall')
plt.ylabel('score/')
plt.xlabel('epoch')
plt.legend(['val_f1', 'val_recall'], loc='best')
plt.show()

### Confusion Matrix
classes = [0, 1, 2]
y_pred = np.argmax(model.predict(test_X), axis=-1)
con_mat = tf.math.confusion_matrix(labels=test_Y, predictions=y_pred)
with tf.Session():
    print('Confusion Matrix (Precision): \n', tf.Tensor.eval(con_mat, feed_dict=None, session=None))

    con_mat = tf.Tensor.eval(con_mat, feed_dict=None, session=None)
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=0)[np.newaxis, :], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    figure = plt.figure(figsize=(5, 5))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

print(classification_report(y_true=test_Y, y_pred=y_pred))
