import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

camera = cv2.VideoCapture(0)

while True:

	status , frame = camera.read()

	if status:

			frame = cv2.flip(frame , 1)
		
	image = cv2.resize(frame,(224,224))
	test_image = np.array(img,dtype=np.float32)
	test_image = np.expand_dims(test_image, axis=0)
	normalized_image = test_image/255.0

	prediction = model.predict(normalized_image)
	print(prediction)

	cv2.imshow('feed' , frame)

	code = cv2.waitKey(1)
		
	if code == 32:
			break

camera.release()

cv2.destroyAllWindows()
