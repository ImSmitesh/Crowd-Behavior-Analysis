import cv2
import numpy as np
import tensorflow as tf

# Loading model
model = tf.keras.models.load_model("Model.h5", compile=False)

# Live dectection
def live_detect():
	cap = cv2.VideoCapture(0)

	frame_count = 0
	single_chunk = []

	while True:
		img = cap.read()[1]
		cv2.imshow("Detection", img)

		if frame_count%5 == 0:
			img = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)
			single_chunk.append(img)
		
		frame_count += 1

		if len(single_chunk) == 30:
			video = np.array([single_chunk])
			video = video / 255
			pred = model.predict(video)
			pred = np.round(pred).reshape(-1)
			
			if pred[0] > 0.5:
				print("Fight detected")
			else:
				print("Non Fight")
			
			count = 0
			single_chunk = []
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			break

# Video prediction
def predict_video(path):
	cap = cv2.VideoCapture(path)
	frames = []
	for i in range(0,150,5):
		cap.set(cv2.CAP_PROP_POS_FRAMES, i)
		img = cap.read()[1]
		img = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)
		frames.append(img)
	
	video = np.array([frames])
	video = video / 255

	pred = model.predict(video)
	pred = np.round(pred).reshape(-1)
	
	if pred[0] > 0.5:
		print("Fight detected")
	else:
		print("Non Fight")


print("1. Live Video Detection")
print("2. Video Classification")
choice = input("Enter choice: ")

match choice:
	case "1":
		live_detect()
	
	case "2":
		path = input("Enter video path: ")
		predict_video(path)
	
	case _:
		pass