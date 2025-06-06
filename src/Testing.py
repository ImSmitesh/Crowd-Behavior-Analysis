import cv2
import numpy as np
import math
import glob
import random
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from vidaug import augmentors as va
import tensorflow as tf

fight_train = glob.glob(r".\RWF-2000\train\Fight\*.mkv")
fight_labels_train = np.ones(800)
nonfight_train = glob.glob(r".\RWF-2000\train\NonFight\*.mkv")
nonfight_labels_train = np.zeros(800)

fight_val = glob.glob(r".\RWF-2000\val\Fight\*.mkv")
fight_labels_val = np.ones(200)
nonfight_val = glob.glob(r".\RWF-2000\val\NonFight\*.mkv")
nonfight_labels_val = np.zeros(200)

X = fight_train + nonfight_train + fight_val + nonfight_val
Y1 = np.append(fight_labels_train, nonfight_labels_train) 
Y2 = np.append(fight_labels_val, nonfight_labels_val)
Y = np.append(Y1, Y2)

# Selecting random sample from whole dataset for testing
test_sample = 200
all_sample = [ i for i in range(len(X)) ]
random_sample = random.sample(all_sample,k=test_sample)

X_test = [ X[i] for i in random_sample]
Y_test = [ Y[i] for i in random_sample]

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

# Loading the model
model = tf.keras.models.load_model("Model.h5", compile=False)

# Data Generator
batch_size = 4
skip_interval = 5
test_generator = Data_Generator(X_test, Y_test, batch_size, skip_interval, "Test")

# Predicting
pred_float = model.predict(test_generator)
pred = np.round(pred_float)

# Classification report
report = classification_report(Y_test, pred)
print(report)