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
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(2,activation="linear"))
#loss function
def neg_log_likelihood(y_true,y_pred ):

    pred_0=tf.gather(y_pred,[0],axis=1)
    pred_1=tf.gather(y_pred,[1],axis=1)
    sigma=tf.exp(pred_1)
    return tf.reduce_mean(tf.divide(tf.square(y_true - pred_0),2.0 * tf.square(sigma)) + (2.0 * tf.math.log(sigma)))




model.compile(loss=neg_log_likelihood, optimizer='sgd')

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
#Creating a folder to save tensorboard logdir
path_log="./logs_model2/"
try:
    os.mkdir(path_log)
except OSError as error:
    print(error)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path_log+"model2", histogram_freq=1)


history = model.fit(x_train,y_train, batch_size=32,callbacks=[callback,tensorboard_callback], epochs=100,validation_data=(x_val, y_val))

model.save("my_model_nll.h5")



#Adding test .csv files into one dataset
path = r'Files_IAV/Test' # use your path

all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0)

    li.append(df)

frame1 = pd.concat(li, axis=0, ignore_index=True)

time=frame1['time'].values
x_test=frame1.loc[:,column_names].values
y_test=frame1['target'].values

#model = keras.models.load_model('my_model_nll.h5',custom_objects={'neg_log_likelihood':neg_log_likelihood})

#Predict using trained model on test data
testPredict = model.predict(x_test)


#Creating a folder to save all figures
path_fig="./Figure_model2/"
try:
    os.mkdir(path_fig)
except OSError as error:
    print(error)

#plot actual data,Predicted data and confidence interval
plt.figure(figsize=(50,20),dpi=50)
l=500
a_list = list(range(0, 500))
plt.plot(a_list,testPredict[:l,0],'-',label='Prediction')
plt.plot(a_list,y_test[:l],'or',label='True Values')

plt.fill_between(a_list, testPredict[:l,0]-(2.0*tf.exp(testPredict[:l,1])) , testPredict[:l,0]+(2.0*tf.exp(testPredict[:l,1])),color='gray', alpha=0.2)
plt.title('NLL Prediction')
plt.legend()
plt.savefig(path_fig+"Predict.png")









