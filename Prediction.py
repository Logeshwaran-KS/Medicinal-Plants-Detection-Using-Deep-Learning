import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow import expand_dims
from tensorflow.nn import sigmoid

# Class Names
class_names = ['Aloevera','Amla','Amruthaballi','Arali','Astma_weed','Badipala','Balloon_Vine','Bamboo','Beans','Betel','Bhrami','Bringaraja','Caricature','Castor','Catharanthus','Chakte','Chilly','Citron lime (herelikai)','Coffee','Common rue(naagdalli)','Coriender','Curry','Doddpathre','Drumstick','Ekka','Eucalyptus','Ganigale','Ganike','Gasagase','Ginger','Globe Amarnath','Guava','Henna','Hibiscus','Honge','Insulin','Jackfruit','Jasmine','Kambajala','Kasambruga','Kohlrabi','Lantana','Lemon','Lemongrass','Malabar_Nut','Malabar_Spinach','Mango','Marigold','Mint','Neem','Nelavembu','Nerale','Nooni','Onion','Padri','Palak(Spinach)','Papaya','Parijatha','Pea','Pepper','Pomoegranate','Pumpkin','Raddish','Rose','Sampige','Sapota','Seethaashoka','Seethapala','Spinach1','Tamarind','Taro','Tecoma','Thumbe','Tomato','Tulsi','Turmeric','ashoka','camphor','kamakasturi','kepala']

# Load the model
model = load_model('model_path')
# Load the image to predict
image = load_img('image_path', target_size=(299, 299))
image_array = img_to_array(image)
image_array = expand_dims(image_array, 0)
predictions = model.predict(image_array)
score = sigmoid(predictions[0])
# Predict along with it confidence
print("This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
