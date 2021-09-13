import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MeanSquaredError
import datetime
import seaborn as sns
import os


tf.random.set_seed(1234)



df = pd.read_csv('Files_IAV/Training/file_0.csv',parse_dates=True)
df.head()
df.shape


#Creating a folder to save all figures
path_fig="./Figure_model1/"
try:
    os.mkdir(path_fig)
except OSError as error:
    print(error)


#Visualize the data
plt.figure()
vis=df.plot(subplots=True)

vis_fig=vis[0].get_figure()
vis_fig.savefig(path_fig+"Visualization.png")



#Plot the correlation matrix to check the correlation between variables and target
plt.figure()
corr=df.corr()
sns_plot=sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
figure = sns_plot.get_figure()
figure.savefig(path_fig+"Heatmap_correlation.png")


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
model.add(keras.layers.Dense(128, activation="relu",input_shape=(8,)))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(1,activation="linear"))

#loss function
def mse(Y_true,Y_pred):
    return tf.reduce_mean(tf.square(Y_true-Y_pred))

model.compile(loss=mse, optimizer='sgd',metrics=['mean_squared_error'])

#model.compile(loss=MeanSquaredError(),optimizer='sgd',metrics=['mean_squared_error'])
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
#Creating a folder to save tensorboard logdir
path_log="./logs_model1/"
try:
    os.mkdir(path_log)
except OSError as error:
    print(error)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path_log+"model1", histogram_freq=1)


history = model.fit(x_train,y_train, batch_size=32, epochs=30,callbacks=[callback,tensorboard_callback],validation_data=(x_val, y_val))
#model.summary()


model.save("my_model.h5")



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

model = keras.models.load_model('my_model.h5')
#Predict using trained model on test data
testPredict = model.predict(x_test)

#plot actual data and Predicted data
plt.figure()
plt.figure(figsize=(50,20),dpi=50)
l=500
a_list = list(range(0, 500))
plt.plot(a_list,testPredict[:l],'-',label='Prediction')
plt.plot(a_list,y_test[:l],'or',label='True Values')
plt.title('MSE Model Prediction')
plt.legend()
plt.savefig(path_fig+"Predict.png")
