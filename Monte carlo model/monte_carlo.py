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





#Build model with dropout
model = keras.models.Sequential()
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(256, activation="relu",input_shape=(8,)))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(1, activation="linear"))


#loss function
def mse(Y_true,Y_pred):
    return tf.reduce_mean(tf.square(Y_true-Y_pred))
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
#model.compile(loss=mse, optimizer=optimizer)
model.compile(loss=mse, optimizer=opt,metrics=['mean_squared_error'])

#Creating a folder to save tensorboard logdir
path_log="./logs_model4/"
try:
    os.mkdir(path_log)
except OSError as error:
    print(error)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path_log+"model4", histogram_freq=1)


#Function to get predictive distribution over num_samples
def predict_dist(X, model, num_samples):
    preds = [model(X, training=True) for _ in range(num_samples)]
    return np.hstack(preds)
#function to get mean and standard deviation
def predict_point(X, model, num_samples):
    pred_dist = predict_dist(X, model, num_samples)
    return pred_dist.mean(axis=1),pred_dist.std(axis=1)

history = model.fit(x_train,y_train, batch_size=32, epochs=200,callbacks=[tensorboard_callback],validation_data=(x_val, y_val))
#model.summary()



model.save("my_model_monte.h5")



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

#model = keras.models.load_model('my_model_monte.h5')

#Predict using trained model on test data over 100 samples and then get the mean
y_pred_dist = predict_dist(x_test, model, 100)
y_pred_mean,y_pred_std = predict_point(x_test, model, 100)



path_fig="./Figure_model4/"
try:
    os.mkdir(path_fig)
except OSError as error:
    print(error)
#plot actual data,Predicted data and confidence interval
l=500
a_list = list(range(0, 500))
plt.figure(figsize=(50,20),dpi=50)
plt.plot(a_list,y_test[:l],'or',label='True Values')
plt.plot(a_list,y_pred_mean[:l],'-',label='Prediction')
plt.fill_between( a_list,y_pred_mean[:l]-y_pred_std[:l] , y_pred_mean[:l]+y_pred_std[:l],color='gray', alpha=0.2)
plt.title('Monte Carlo Prediction')
plt.legend()
plt.savefig(path_fig+"Predict.png")












