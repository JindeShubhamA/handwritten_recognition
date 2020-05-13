'''
from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
!pip install --upgrade tensorflow
!pip install --upgrade keras
'''

## Imports
# General
import os
from datetime import datetime
import tensorflow as tf
keras = tf.keras
from tensorflow.keras import layers, models, optimizers, regularizers


# from sklearn.metrics import accuracy_score
## Network input specs
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

## Hyperparameters
NUM_CLASSES = 27
EPOCHS = 300
BATCH_SIZE = 32
# ARCHITECTURE = 'ResNet50'
ARCHITECTURE = 'VGG16'
NODES_HIDDEN_0 = 512
NODES_HIDDEN_1 = 512
BASE_TRAINABLE = False
REGULARIZER = 'l2'  # 'None' | 'l1' | 'l2'
REGULARIZATZION_STRENGTH = '0.01'
AUGMENTATION = 1
OPTIMIZER = "adam"
## For documentation purposes - Add all parameters set above to this dict
params = dict(
    img_size=IMG_SIZE,
    img_shape=(IMG_SIZE, IMG_SIZE, 3),
    num_classes=NUM_CLASSES,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    architecture=ARCHITECTURE,
    nodes_hidden_0=NODES_HIDDEN_0,

    base_trainable=BASE_TRAINABLE,
    regularizer=REGULARIZER,
    augmentation=AUGMENTATION,
    optimier=OPTIMIZER,
    regularization_strength=REGULARIZATZION_STRENGTH,
)

## Build model components from string-specifications

architecture = eval('tf.keras.applications.' + ARCHITECTURE)
regularizer = None if REGULARIZER is 'None' else eval(
    'regularizers.' + REGULARIZER + '(' + REGULARIZATZION_STRENGTH + ')')

## Set path for saving training progress & data
now = datetime.now()
TIME_STAMP = now.strftime("_%Y_%d_%m__%H_%M_%S__%f")
MODEL_ID = 'Model_' + TIME_STAMP + '/'

DATA_STORAGE_PATH = '/data/s3993914/Dl_output/'
TRAINED_MODELS = 'Trained_Models/'
MODEL_ARCHITECTURE = ARCHITECTURE + '/'
path = DATA_STORAGE_PATH + TRAINED_MODELS + MODEL_ARCHITECTURE + MODEL_ID
TB_LOG_DIR = path + 'Tensorboard' + '/'
DATA = "monkbrill/"
TRAIN_RATIO = 0.8

TRAINING = "training/"
TESTTING = "testing/"



subfolders = [ f.path for f in os.scandir(DATA) if f.is_dir() ]

count = -1

class_count = 0
for subfolder in subfolders:
    folder_name = subfolder.split("/")[1]
    os.mkdir("/Users/jindeshubham/Desktop/handwritten_recognition/training/" + folder_name)
    os.mkdir("/Users/jindeshubham/Desktop/handwritten_recognition/testing/" + folder_name)

    subfolder = "/Users/jindeshubham/Desktop/handwritten_recognition/" + subfolder
    print("Subfolder is ",subfolder)
    count=count+1
    train_list=[]
    test_list =[]
    files = []
    for (dirpath, dirnames, filenames) in walk(subfolder):
        file_cnt = len(filenames)


        for file in filenames:
            print("file is ",file)
            file_jpg = file.split(".")[0]
            file_jpg = file_jpg + ".jpg"

            if np.random.rand(1) < 0.8:
               shutil.copy(subfolder + '/' + file,'training/' + folder_name + "/" + file_jpg)
               train_list.append(str(count))
            else:
               shutil.copy(subfolder + '/' + file, 'testing/' + folder_name + "/" + file_jpg)
               test_list.append(str(count))

        print("Training class\n ",train_list)
        with open("training.txt", "a+") as f:
             f.write("\n")
             f.write('\n'.join(train_list))

        with open("testing.txt", "a+") as f:
             f.write("\n")
             f.write('\n'.join(test_list))



## Obtain dataset

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data_test = image_generator.flow_from_directory('training/', class_mode="categorical", batch_size=32, target_size=(224, 224))
image_data_train = image_generator.flow_from_directory('testing/', class_mode="categorical", batch_size=32, target_size=(224, 224))


tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.


base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=IMG_SHAPE)

base_model.trainable = BASE_TRAINABLE



global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  # Suggested on TF tutorial page...
flatten_operation = layers.Flatten()
hidden_dense_layer_0 = layers.Dense(NODES_HIDDEN_0, activation='relu', kernel_regularizer=regularizer)
hidden_dense_layer_1 = layers.Dense(NODES_HIDDEN_1, activation='relu', kernel_regularizer=regularizer)
prediction_layer = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizer)

# Construct overall model
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    flatten_operation,
    hidden_dense_layer_0,
    prediction_layer
])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001,
                                           beta_1=0.9,
                                           beta_2=0.999,
                                           epsilon=1e-07,
                                           amsgrad=False,
                                           name='Adam'
                                           ),
              loss='sparse_categorical_crossentropy',  # Capable of working with regularization
              metrics=['accuracy', 'sparse_categorical_crossentropy'])


model.summary()

history = model.fit(
    x=image_data_train,
    epochs=EPOCHS,
    verbose=1,
    validation_data=image_data_test,
    initial_epoch=0)

