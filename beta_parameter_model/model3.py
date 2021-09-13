import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import os

tf.random.set_seed(1234)


#Add all the training data .csv files into one dataset
path = r'Files_IAV/Training' # use your path

all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0)

    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)


#Ommission of variables var5 and var6 due to low correlation with target
column_names = [
    'var0',
    'var1',
    'var2',
    'var3',
    'var4',
    'var7',
    'var8',
    'var9'

]

#features and label into x and y
x=frame.loc[:,column_names].values
y=frame['target'].values

#Split the training dataset into training and validation data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3)





#Build model
model = keras.models.Sequential()
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(2,activation="linear"))


#loss function
def betaloss(Y_true,Y_pred):
    beta=.04
    pred_0 = tf.gather(Y_pred, [0], axis=1)
    pred_1 = tf.gather(Y_pred, [1], axis=1)

    if tf.split(tf.greater(pred_0,pred_1),1):
        return tf.reduce_mean(
            (tf.math.maximum(tf.square(Y_true - pred_0), 0)) + (tf.math.maximum(tf.square(pred_1 - Y_true), 0))+ ( beta * (pred_0 - pred_1)))
    else:
        return tf.reduce_mean(
            (tf.math.maximum(tf.square(Y_true - pred_1), 0)) + (tf.math.maximum(tf.square(pred_0 - Y_true), 0)) +( beta * (pred_1 -pred_0)))




model.compile(loss=betaloss, optimizer='sgd')


callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

#Creating a folder to save tensorboard logdir
path_log="./logs_model3/"
try:
    os.mkdir(path_log)
except OSError as error:
    print(error)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path_log+"model3", histogram_freq=1)


history = model.fit(x_train,y_train, batch_size=32, epochs=200,callbacks=[callback,tensorboard_callback],validation_data=(x_val, y_val))
#model.summary()
#callbacks=[callback,]

model.save("my_model3.h5")



#Adding test .csv files into one dataset
path = r'Files_IAV/Test' # use your path

all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0)

    li.append(df)

frame1 = pd.concat(li, axis=0, ignore_index=True)


x_test=frame1.loc[:,column_names].values
y_test=frame1['target'].values

#model = keras.models.load_model('my_model3.h5',custom_objects={'betaloss':betaloss})

#Predict using trained model on test data
testPredict = model.predict(x_test)

testpred_0 = tf.gather(testPredict, [0], axis=1)
testpred_1 = tf.gather(testPredict, [1], axis=1)


path_fig="./Figure_model3/"
try:
    os.mkdir(path_fig)
except OSError as error:
    print(error)

#plot actual data and confidence interval
plt.figure(figsize=(50,20),dpi=50)
l=500
a_list = list(range(0, 500))

plt.plot(a_list,y_test[:l],'or',label='True Values')

if tf.split(tf.greater(testpred_0, testpred_1), 1):
    plt.fill_between(a_list,testPredict[:l,1] ,testPredict[:l,0],color='gray', alpha=0.2)
else:
    plt.fill_between(a_list, testPredict[:l, 0], testPredict[:l, 1], color='gray', alpha=0.2)
plt.title('Model3 Prediction')
plt.legend()
plt.savefig(path_fig+"Predict.png")