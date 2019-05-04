import os
import codecs
import json
import base64
import pickle
import pandas
from pprint import pprint

# KUO DAUGIAU TAŠKŲ TUO ILGIAU NEPARDUOS

def predictSale(car):
    soldInDays = 10
    try:
        soldInDays += pointsForDate(car['Pagaminimo data'])
        soldInDays += pointsForPrice(car['Kaina Lietuvoje'])
        soldInDays += pointsForMileage(car['Rida'])
        soldInDays += pointsForFuelType(car['Kuro tipas'])
        soldInDays += pointsForBodyType(car['Kėbulo tipas'])
        soldInDays += pointsForWheelPosition(car['Vairo padėtis'])
        soldInDays += pointsForDefects(car['Defektai'])
        soldInDays += pointsForLocation(car['Miestas'])
        
        car['Taškai už nuotrauką'] = pointsForImage(car['Defektai'], car['Pagaminimo data'], car['Spalva'], car['Kėbulo tipas'])
        soldInDays += pointsForImage(car['Defektai'], car['Pagaminimo data'], car['Spalva'],  car['Kėbulo tipas'])

        pass
    except Exception as e:
        print(e) 
        pass


    car['Parduota per'] = soldInDays

    return car

def pointsForDate(date):

    date = date.replace('-', '')
    if int(date) > 201500:
        return -50
    if int(date) > 201000:
        return 0
    if int(date) > 200500:
        return 50
    if int(date) > 200000:
        return 100
    return 150

def pointsForPrice(price):
    price = price.split('€', 1)[0].replace(' ', '')
    if int(price) > 15000:
        return 150
    if int(price) > 10000:
        return 100
    if int(price) > 8000:
        return 80
    if int(price) > 6000:
        return -30
    if int(price) > 2000:
        return -50
    if int(price) > 1000:
        return -80
    return -100

def pointsForMileage(mileage):
    if mileage == 'NA':
        return 50

    mileage = mileage.replace('km', '').replace(' ', '')
    if int(mileage) > 400000:
        return 100
    if int(mileage) > 300000:
        return 60
    if int(mileage) > 200000:
        return 40
    if int(mileage) > 100000:
        return -40
    return -60

def pointsForFuelType(fuel):
    if fuel == 'Dyzelinas':
        return -20
    if fuel == 'Benzinas':
        return 10
    return -30

def pointsForBodyType(bodyType):
    if bodyType == 'Hečbekas':
        return -20
    if bodyType == 'Sedanas':
        return -30
    if bodyType == 'Visureigis':
        return -10
    return 40

def pointsForWheelPosition(wheelPosition):
    if wheelPosition == 'Kairėje':
        return 0
    return 50
    
def pointsForDefects(defects):
    if defects == 'Be defektų':
        return 0
    return 40

def pointsForLocation(location):
    if location == 'Vilnius':
        return -30
    if location == 'Kaunas':
        return -20
    if location == 'Klaipėda':
        return -10
    if location == 'Marijampolė':
        return 0
    return 30

def pointsForImage(defected, date, color, bodyType):
    points = 0
    if defected == 'Be defektų':
        points += 0
    else:
        points += 50
    date = date.replace('-', '')

    if int(date) > 201500:
        points += 00
    elif int(date) > 201300:
        points += 30
    elif int(date) > 201100:
        points += 40
    elif int(date) > 200800:
        points += 50
    else:
        points += 60

    if color == 'Mėlyna / žydra':
        points += 50
    elif color == 'Raudona / vyšninė':
        points += 60

    if bodyType == 'Hečbekas':
        points += 40
    if bodyType == 'Sedanas':
        points += 20
    if bodyType == 'Visureigis':
        points += 10

    return points


def labelDictionaryMaker(labelee, dictionary):
    if labelee not in dictionary:
        dictionary[labelee] = len(dictionary)  
        
    return dictionary

def exportImage(fileName, base64String):
    with open(fileName, 'wb') as f:
        f.write(base64.b64decode(base64String))

import os
import codecs
import json
import base64
import pickle
import pandas
from pprint import pprint

# KUO DAUGIAU TAŠKŲ TUO ILGIAU NEPARDUOS

def predictSale(car):
    soldInDays = 10
    try:
        soldInDays += pointsForDate(car['Pagaminimo data'])
        soldInDays += pointsForPrice(car['Kaina Lietuvoje'])
        soldInDays += pointsForMileage(car['Rida'])
        soldInDays += pointsForFuelType(car['Kuro tipas'])
        soldInDays += pointsForBodyType(car['Kėbulo tipas'])
        soldInDays += pointsForWheelPosition(car['Vairo padėtis'])
        soldInDays += pointsForDefects(car['Defektai'])
        soldInDays += pointsForLocation(car['Miestas'])
        
        car['Taškai už nuotrauką'] = pointsForImage(car['Defektai'], car['Pagaminimo data'], car['Spalva'], car['Kėbulo tipas'])
        soldInDays += pointsForImage(car['Defektai'], car['Pagaminimo data'], car['Spalva'],  car['Kėbulo tipas'])

        pass
    except Exception as e:
        print(e) 
        pass


    car['Parduota per'] = soldInDays

    return car

def pointsForDate(date):

    date = date.replace('-', '')
    if int(date) > 201500:
        return -50
    if int(date) > 201000:
        return 0
    if int(date) > 200500:
        return 50
    if int(date) > 200000:
        return 100
    return 150

def pointsForPrice(price):
    price = price.split('€', 1)[0].replace(' ', '')
    if int(price) > 15000:
        return 150
    if int(price) > 10000:
        return 100
    if int(price) > 8000:
        return 80
    if int(price) > 6000:
        return -30
    if int(price) > 2000:
        return -50
    if int(price) > 1000:
        return -80
    return -100

def pointsForMileage(mileage):
    if mileage == 'NA':
        return 50

    mileage = mileage.replace('km', '').replace(' ', '')
    if int(mileage) > 400000:
        return 100
    if int(mileage) > 300000:
        return 60
    if int(mileage) > 200000:
        return 40
    if int(mileage) > 100000:
        return -40
    return -60

def pointsForFuelType(fuel):
    if fuel == 'Dyzelinas':
        return -20
    if fuel == 'Benzinas':
        return 10
    return -30

def pointsForBodyType(bodyType):
    if bodyType == 'Hečbekas':
        return -20
    if bodyType == 'Sedanas':
        return -30
    if bodyType == 'Visureigis':
        return -10
    return 40

def pointsForWheelPosition(wheelPosition):
    if wheelPosition == 'Kairėje':
        return 0
    return 50
    
def pointsForDefects(defects):
    if defects == 'Be defektų':
        return 0
    return 40

def pointsForLocation(location):
    if location == 'Vilnius':
        return -30
    if location == 'Kaunas':
        return -20
    if location == 'Klaipėda':
        return -10
    if location == 'Marijampolė':
        return 0
    return 30

def pointsForImage(defected, date, color, bodyType):
    points = 0
    if defected == 'Be defektų':
        points += 0
    else:
        points += 50
    date = date.replace('-', '')

    if int(date) > 201500:
        points += 00
    elif int(date) > 201300:
        points += 30
    elif int(date) > 201100:
        points += 40
    elif int(date) > 200800:
        points += 50
    else:
        points += 60

    if color == 'Mėlyna / žydra':
        points += 50
    elif color == 'Raudona / vyšninė':
        points += 60

    if bodyType == 'Hečbekas':
        points += 40
    if bodyType == 'Sedanas':
        points += 20
    if bodyType == 'Visureigis':
        points += 10

    return points


def labelDictionaryMaker(labelee, dictionary):
    if labelee not in dictionary:
        dictionary[labelee] = len(dictionary)  
        
    return dictionary

def exportImage(fileName, base64String):
    with open(fileName, 'wb') as f:
        f.write(base64.b64decode(base64String))


def getData():
    # Loading the data.
    with open('/home/ignas/Downloads/data.json') as f:
        data = json.load(f)

    # Predicting sale time.
    for car in data:
        car = predictSale(car)

    # Processing.
    dicFuel = {}
    dicBody = {}
    dicFuel = {}
    dicWheel = {}
    dicLocation = {}

    for car in data:
        dicFuel = labelDictionaryMaker(car['Kuro tipas'], dicFuel)
        dicBody = labelDictionaryMaker(car['Kėbulo tipas'], dicBody)
        dicLocation = labelDictionaryMaker(car['Miestas'], dicLocation)
        dicWheel = labelDictionaryMaker(car['Vairo padėtis'], dicWheel)

    for car in data:
        car['Kuro tipas'] = dicFuel[car['Kuro tipas']]
        car['Kėbulo tipas'] = dicBody[car['Kėbulo tipas']]
        car['Miestas'] = dicLocation[car['Miestas']]
        car['Vairo padėtis'] = dicWheel[car['Vairo padėtis']]
        car['Pagaminimo data'] = int(car['Pagaminimo data'].replace('-', ''))
        car['Kaina Lietuvoje'] = car['Kaina Lietuvoje'].split('€', 1)[0].replace(' ', '')
        car['Rida'] = car['Rida'].replace('km', '').replace(' ', '')
        if car['Rida'] == 'NA':
            car['Rida'] = 0

        # Special case.
        if car['Defektai'] == 'Be defektų':
            car['Defektai'] = 0
        else:
            car['Defektai'] = 1
 
    # Dumping data for later.
    with open('car.dictionary', 'wb') as config_dictionary_file:
        pickle.dump(data, config_dictionary_file)

    return data

def getSavedData():
    with open('car.dictionary', 'rb') as config_dictionary_file:
        data = pickle.load(config_dictionary_file)
    return data

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras import backend as K
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.viewer import ImageViewer
from scipy.stats import gamma
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import seaborn as sns
import pandas
import json
import base64
import os

def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = imread(imgdata, plugin="imageio")
    return img

def get_prepared_data():

    data = getData()
    sale = predictSale(data[0])


     # Inspecting the data.
    fig, axes = plt.subplots(1, 3)


    dfObj = pandas.DataFrame(data) 
    dfObj = dfObj.dropna()

    dfObj = dfObj.sample(frac=0.9)
    dfObj = dfObj.astype(str)
    sns.kdeplot(dfObj["Parduota per"], ax=axes[0])
    sns.kdeplot(dfObj["Rida"], ax=axes[1])
    sns.kdeplot(dfObj["Kaina Lietuvoje"], ax=axes[2])
    
    # sns.pairplot(dfObj[["Parduota per"]])
    plt.show()

    dataset = []
    descriptions = []
    labels = []
    # folder = "adidas\\"
    # data_paths = os.listdir(folder)

    # Universal Sentance Encoder for the description.
    # module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    # Import the Universal Sentence Encoder"s TF Hub module
    # embed = hub.Module(module_url)

    # ResNet model for the images.
    base_model = ResNet50(weights="imagenet")

    progress = 0
    for car in data:

        progress = progress + 1
        print(str(progress) + " / 2151")

        try:
            # Preparing the images.
            img = decode(car["image"])
            print(img.shape)
            img = resize(img, (224, 224, 3), anti_aliasing=True) # Resnet feature extractor used 224x224 images, so doing this for simplicity.
            x = np.expand_dims(img, axis=0)
            x = preprocess_input(x)
            image_features = base_model.predict(x)
            print(image_features.shape)

            # Preparing the metadata.
            fuel = car["Kuro tipas"]
            chasis = car["Kėbulo tipas"]
            city = car["Miestas"]
            wheel_position = car["Vairo padėtis"]
            date = car["Pagaminimo data"]
            cost = car["Kaina Lietuvoje"]
            mileage = car["Rida"]
            defects = car["Defektai"]
            # descriptions.append(description)
            prediction = car["Parduota per dienas (pagal taisykles)"]
            labels.append(prediction)

            joined_features = []
            joined_features.append(int(fuel))
            joined_features.append(int(chasis))
            joined_features.append(int(city))
            joined_features.append(int(wheel_position))
            joined_features.append(int(date))
            joined_features.append(int(cost))
            joined_features.append(int(mileage))
            joined_features.append(int(defects))
            joined_features.extend(image_features[0])

            dataset.append(joined_features)


        except Exception as e:
            print(e)
            pass

    # Description. Creating all description features in one bulk, because I don"t want to call session run for each product.
    # with tf.Session() as session:
    #     session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    #     description_features = session.run(embed(descriptions))

    data = np.array(dataset)
    # data = np.concatenate((data, description_features), axis=1)

    return data, labels

"""
    Discrete log-likelihood for Weibull hazard function on censored survival data
    y_true is a (samples, 2) tensor containing time-to-event (y), and an event indicator (u)
    ab_pred is a (samples, 2) tensor containing predicted Weibull alpha (a) and beta (b) parameters
    For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)
"""
def weibull_loglik_discrete(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = 1
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    hazard0 = K.pow((y_ + 1e-35) / a_, b_)
    hazard1 = K.pow((y_ + 1) / a_, b_)

    return -1 * K.mean(u_ * K.log(K.exp(hazard1 - hazard0) - 1.0) - hazard1)

"""
    Custom Keras activation function, outputs alpha neuron using exponentiation and beta using softplus
"""
def activate(ab):
  
    a = K.exp(ab[:, 0])
    b = K.exp(ab[:, 1])

    a = K.reshape(a, (K.shape(a)[0], 1))
    b = K.reshape(b, (K.shape(b)[0], 1))

    return K.concatenate((a, b), axis=1)


def plot_history(history):
  hist = pandas.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

data, labels = get_prepared_data()
training_data = data[0:1800]
training_labels = labels[0:1800]

validation_data = data[1800:]
validation_labels = labels[1800:]
def getData():
    # Loading the data.
    with open('/home/ignas/Downloads/data.json') as f:
        data = json.load(f)

    # Predicting sale time.
    for car in data:
        car = predictSale(car)

    # Processing.
    dicFuel = {}
    dicBody = {}
    dicFuel = {}
    dicWheel = {}
    dicLocation = {}

    for car in data:
        dicFuel = labelDictionaryMaker(car['Kuro tipas'], dicFuel)
        dicBody = labelDictionaryMaker(car['Kėbulo tipas'], dicBody)
        dicLocation = labelDictionaryMaker(car['Miestas'], dicLocation)
        dicWheel = labelDictionaryMaker(car['Vairo padėtis'], dicWheel)

    for car in data:
        car['Kuro tipas'] = dicFuel[car['Kuro tipas']]
        car['Kėbulo tipas'] = dicBody[car['Kėbulo tipas']]
        car['Miestas'] = dicLocation[car['Miestas']]
        car['Vairo padėtis'] = dicWheel[car['Vairo padėtis']]
        car['Pagaminimo data'] = int(car['Pagaminimo data'].replace('-', ''))
        car['Kaina Lietuvoje'] = car['Kaina Lietuvoje'].split('€', 1)[0].replace(' ', '')
        car['Rida'] = car['Rida'].replace('km', '').replace(' ', '')
        if car['Rida'] == 'NA':
            car['Rida'] = 0

        # Special case.
        if car['Defektai'] == 'Be defektų':
            car['Defektai'] = 0
        else:
            car['Defektai'] = 1
 
    # Dumping data for later.
    with open('car.dictionary', 'wb') as config_dictionary_file:
        pickle.dump(data, config_dictionary_file)

    return data

def getSavedData():
    with open('car.dictionary', 'rb') as config_dictionary_file:
        data = pickle.load(config_dictionary_file)
    return data

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras import backend as K
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.viewer import ImageViewer
from scipy.stats import gamma
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import seaborn as sns
import pandas
import json
import base64
import os

def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = imread(imgdata, plugin="imageio")
    return img

def get_prepared_data():

    data = getData()
    sale = predictSale(data[0])


     # Inspecting the data.
    fig, axes = plt.subplots(1, 3)


    dfObj = pandas.DataFrame(data) 
    dfObj = dfObj.dropna()

    dfObj = dfObj.sample(frac=0.9)
    dfObj = dfObj.astype(str)
    sns.kdeplot(dfObj["Parduota per"], ax=axes[0])
    sns.kdeplot(dfObj["Rida"], ax=axes[1])
    sns.kdeplot(dfObj["Kaina Lietuvoje"], ax=axes[2])
    
    # sns.pairplot(dfObj[["Parduota per"]])
    plt.show()

    dataset = []
    descriptions = []
    labels = []
    # folder = "adidas\\"
    # data_paths = os.listdir(folder)

    # Universal Sentance Encoder for the description.
    # module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    # Import the Universal Sentence Encoder"s TF Hub module
    # embed = hub.Module(module_url)

    # ResNet model for the images.
    base_model = ResNet50(weights="imagenet")

    progress = 0
    for car in data:

        progress = progress + 1
        print(str(progress) + " / 2151")

        try:
            # Preparing the images.
            img = decode(car["image"])
            print(img.shape)
            img = resize(img, (224, 224, 3), anti_aliasing=True) # Resnet feature extractor used 224x224 images, so doing this for simplicity.
            x = np.expand_dims(img, axis=0)
            x = preprocess_input(x)
            image_features = base_model.predict(x)
            print(image_features.shape)

            # Preparing the metadata.
            fuel = car["Kuro tipas"]
            chasis = car["Kėbulo tipas"]
            city = car["Miestas"]
            wheel_position = car["Vairo padėtis"]
            date = car["Pagaminimo data"]
            cost = car["Kaina Lietuvoje"]
            mileage = car["Rida"]
            defects = car["Defektai"]
            # descriptions.append(description)
            prediction = car["Parduota per dienas (pagal taisykles)"]
            labels.append(prediction)

            joined_features = []
            joined_features.append(int(fuel))
            joined_features.append(int(chasis))
            joined_features.append(int(city))
            joined_features.append(int(wheel_position))
            joined_features.append(int(date))
            joined_features.append(int(cost))
            joined_features.append(int(mileage))
            joined_features.append(int(defects))
            joined_features.extend(image_features[0])

            dataset.append(joined_features)


        except Exception as e:
            print(e)
            pass

    # Description. Creating all description features in one bulk, because I don"t want to call session run for each product.
    # with tf.Session() as session:
    #     session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    #     description_features = session.run(embed(descriptions))

    data = np.array(dataset)
    # data = np.concatenate((data, description_features), axis=1)

    return data, labels

"""
    Discrete log-likelihood for Weibull hazard function on censored survival data
    y_true is a (samples, 2) tensor containing time-to-event (y), and an event indicator (u)
    ab_pred is a (samples, 2) tensor containing predicted Weibull alpha (a) and beta (b) parameters
    For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)
"""
def weibull_loglik_discrete(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = 1
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    hazard0 = K.pow((y_ + 1e-35) / a_, b_)
    hazard1 = K.pow((y_ + 1) / a_, b_)

    return -1 * K.mean(u_ * K.log(K.exp(hazard1 - hazard0) - 1.0) - hazard1)

"""
    Custom Keras activation function, outputs alpha neuron using exponentiation and beta using softplus
"""
def activate(ab):
  
    a = K.exp(ab[:, 0])
    b = K.exp(ab[:, 1])

    a = K.reshape(a, (K.shape(a)[0], 1))
    b = K.reshape(b, (K.shape(b)[0], 1))

    return K.concatenate((a, b), axis=1)


def plot_history(history):
  hist = pandas.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

data, labels = get_prepared_data()
training_data = data[0:1800]
training_labels = labels[0:1800]

validation_data = data[1800:]
validation_labels = labels[1800:]