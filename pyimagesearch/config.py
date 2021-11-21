# import the necessary packages
import os

# specify the shape of the inputs for our network
# IMG_SHAPE = (28, 28, 1)
IMG_SHAPE = (64, 64, 3)

# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 50

# define the path to the base output directory
BASE_OUTPUT = "output"

# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TRAIN_DIR = f"../DataSet_Nao_RAW/DataSet_SEQUENCE_1"
VALIDATION_DIR = f"../DataSet_Nao_RAW/DataSet_SEQUENCE_2"
TEST_DIR = f"../DataSet_Nao_RAW/DataSet_SEQUENCE_3"
LABELS_DIR = f"../DataSet_Nao_PlaceRecognition/SEQUENCE_2"
SUBSAMPLE = True
