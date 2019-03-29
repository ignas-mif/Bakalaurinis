from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.viewer import ImageViewer
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
import os

def prepare_data():
    dataset = np.empty((1, 1000))
    descriptions = []
    folder = 'D:\\Uni\\adidas\\'
    data_paths = os.listdir(folder)

    # Universal Sentance Encoder for the description.
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)

    # ResNet model for the images.
    base_model = ResNet50(weights='imagenet')

    for file_path in data_paths:
        
        if "jpg" in file_path:
            # Preparing the images.
            img = imread(folder + file_path, as_grey=False)
            print(img.shape)
            img = resize(img, (224, 224), anti_aliasing=True) # Resnet feature extractor used 224x224 images, so doing this for simplicity.
            x = np.expand_dims(img, axis=0)
            x = preprocess_input(x)
            image_features = base_model.predict(x)
            print(image_features.shape)

            # Preparing the metadata.
            img_metadata_path = file_path.replace("jpg", "json")
            with open(folder + img_metadata_path) as f:
                img_metadata = json.load(f)
            prediction = img_metadata["days_until_sale"]
            price = img_metadata["price"]
            size = img_metadata["size"]
            description = img_metadata["description"]
            descriptions.append(description)

            joined_features = image_features[0]
            np.append(joined_features, price)
            np.append(joined_features, size)

            np.append(dataset, joined_features,axis = 0) 

    # Description.
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        description_features = session.run(embed(descriptions))

    print(dataset.shape)
    print(description_features.shape)

    dataset = np.concatenate((dataset, description_features), axis=0)

    return dataset

a = prepare_data()
