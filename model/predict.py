import tensorflow as tf
import os
import numpy as np
import cv2

model = tf.keras.models.load_model('./model/highaccuracymodel.h5')

classes = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ', 'अं', 'अः', 'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ']

def predict():
    predicted_string = ''
    images = [img for img in os.listdir('./model/temp')]
    for img in images:
        test_img = cv2.imread(os.path.join('./model/temp', img))
        test_img  = cv2.resize(test_img, (28,28))
        test_img  = cv2.resize(test_img, (32,32))
        test_input = test_img.reshape((1,32,32,3))
        prob = model.predict(test_input)
        pred = prob.argmax(axis=-1)
        print('Predicted as: ', classes[pred[0]])
        predicted_string = predicted_string + classes[pred[0]]
        os.remove(os.path.join('./model/temp',img))
    
    return predicted_string