import glob
import math
from sklearn.utils import shuffle
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Dense, GlobalAveragePooling2D, TimeDistributed, Bidirectional, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.applications.mobilenet_v2 import MobileNetV2
from vidaug import augmentors as va

# Reading the paths of the videos
fight_train = glob.glob(r".\RWF-2000\train\Fight\*.mkv")
fight_labels_train = np.ones(800)
nonfight_train = glob.glob(r".\RWF-2000\train\NonFight\*.mkv")
nonfight_labels_train = np.zeros(800)

fight_val = glob.glob(r".\RWF-2000\val\Fight\*.mkv")
fight_labels_val = np.ones(200)
nonfight_val = glob.glob(r".\RWF-2000\val\NonFight\*.mkv")
nonfight_labels_val = np.zeros(200)

X_train = fight_train + nonfight_train
X_val = fight_val + nonfight_val

Y_train = np.append(fight_labels_train, nonfight_labels_train)
Y_val = np.append(fight_labels_val, nonfight_labels_val)

X_train, Y_train = shuffle(X_train,Y_train)
X_val, Y_val = shuffle(X_val, Y_val)

# Defining the custom data loader
class Data_Generator(tf.keras.utils.Sequence) :
	def __init__(self, image_filenames, labels, batch_size, skip_interval, mode) :
		self.image_filenames = image_filenames
		self.labels = labels
		self.batch_size = batch_size
		self.skip_interval = skip_interval
		self.mode = mode
		self.dim = va.Multiply(0.3)
		self.bright = va.Multiply(2)
		self.hflip = va.HorizontalFlip()
		self.vflip = va.VerticalFlip()
	
	def __len__(self) :
		return math.ceil(len(self.image_filenames) / self.batch_size)
	
	def __on_epoch_start__(self):
		self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
	
	def on_epoch_end(self):
		self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
	
	def __getitem__(self, idx) :
		batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
		batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
		
		X = []
		Y = []
		
		for video,label in zip(batch_x,batch_y):
			cap = cv2.VideoCapture(video)
			frames = []
			for i in range(0,150,5):
				cap.set(cv2.CAP_PROP_POS_FRAMES, i)
				img = cap.read()[1]
				img = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)
				frames.append(img)
			
			one_video = np.array(frames)
			X.append(one_video)
			Y.append(label)
			
			if self.mode == "Training":
				one_video_dim = np.array(self.dim(one_video))
				X.append(one_video_dim)
				Y.append(label)
				
				one_video_bright = np.array(self.bright(one_video))
				X.append(one_video_bright)
				Y.append(label)
				
				one_video_hflip = np.array(self.hflip(one_video))
				X.append(one_video_hflip)
				Y.append(label)

				one_video_vflip = np.array(self.vflip(one_video))
				X.append(one_video_vflip)
				Y.append(label)
		
		X = np.array(X)
		Y = np.array(Y)
		
		X = X / 255
		Y = np.expand_dims(Y, axis=1)

		return X, Y

# Using feature-extraction module of the MobileNetV2
cnn_model = MobileNetV2(include_top=False, weights="imagenet")

# Making a new Sequential model for video classification
model = Sequential()

# Defining the input of the model
model.add(Input(shape = (30, 224, 224, 3)))

# Using TimeDistributed layer to extract feature from all frames
model.add(TimeDistributed(cnn_model))
model.add(TimeDistributed(GlobalAveragePooling2D()))

# Defining Bidirectional LSTM
lstm_fw = LSTM(units=30)
lstm_bw = LSTM(units=30, go_backwards = True)  
model.add(Bidirectional(lstm_fw, backward_layer = lstm_bw))
model.add(Dropout(0.25))

# Using linear activation for classification
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(32,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(1, activation = 'sigmoid'))

# Defining hyperparameters and callbacks
filepath = 'my_best_model.epoch{epoch:02d}-loss{loss:.4f}-acc{accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(filepath)
optimizer = SGD(learning_rate=1e-2) 
batch_size = 4
skip_interval = 5
epochs = 25

# Defining the instance of the DataLoader
train_dataloader = Data_Generator(X_train, Y_train, batch_size, skip_interval, "Training")
val_dataloader = Data_Generator(X_val, Y_val, batch_size, skip_interval, "Validation")

# Compiling the model
model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ["accuracy"])

# Training the model
history = model.fit(train_dataloader, epochs=epochs, validation_data=val_dataloader, callbacks=[checkpoint])

# Saving the model
model.save("Model.h5")

# Visualizing the metrics graph
def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
	metric_value_1 = model_training_history.history[metric_name_1]
	metric_value_2 = model_training_history.history[metric_name_2]

	epochs = range(len(metric_value_1))

	plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
	plt.plot(epochs, metric_value_2, 'orange', label = metric_name_2)
	
	plt.title(str(plot_name))
	plt.legend()

plot_metric(history, 'accuracy', 'val_accuracy', 'Training Accuracy vs Validation Accuracy')
plot_metric(history, 'loss', 'val_loss', 'Training Loss vs Validation Loss')