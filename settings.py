import os
COMPUTER_NAME = "kuberlab"

WORKER_POOL_SIZE = 8

TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320

BASE_DIR_SSD = os.environ['TRAINING_DIR']+"/"
BASE_DIR = os.environ['TRAINING_DIR']+"/"
#EXTRA_DATA_DIR = os.environ['BOWL_DIR'] + "/resources/"
EXTRA_DATA_DIR = "/bowl/resources/"
#NDSB3_RAW_SRC_DIR = os.environ['BOWL_DIR'] + "/stage1/"
NDSB3_RAW_SRC_DIR = '/notebooks/data/cniit/'
#LUNA16_RAW_SRC_DIR = os.environ['LUNA_DIR'] + "/"
LUNA16_RAW_SRC_DIR = "/luna/"

#NDSB3_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "ndsb3_extracted_images/"
NDSB3_EXTRACTED_IMAGE_DIR = '/notebooks/data/cniit-extracted/'
LUNA16_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "luna16_extracted_images/"
NDSB3_NODULE_DETECTION_DIR = BASE_DIR_SSD + "ndsb3_nodule_predictions/"

