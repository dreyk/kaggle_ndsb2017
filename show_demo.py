import sys
import os
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import pandas
import numpy
from keras import backend as K


sys.path.insert(0, os.environ['SRC_DIR'])
os.putenv('BOWL_DIR','')
os.putenv('LUNA_DIR','')
os.environ['BOWL_DIR'] = ''
os.environ['LUNA_DIR'] = ''


import settings
import helpers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
set_session(tf.Session(config=config))
import step2_train_nodule_detector
K.set_image_dim_ordering("tf")
CUBE_SIZE = step2_train_nodule_detector.CUBE_SIZE
MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE
NEGS_PER_POS = 20
P_TH = 0.6

PREDICT_STEP = 12
USE_DROPOUT = False

#LUNA16_EXTRACTED_IMAGE_DIR = os.environ['DATA_DIR']+'/data/'
LUNA16_EXTRACTED_IMAGE_DIR = os.environ['DATA_DIR']+'/cniit-extracted/'

def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img
def filter_patient_nodules_predictions(df_nodule_predictions, patient_id, view_size):
    src_dir = LUNA16_EXTRACTED_IMAGE_DIR
    patient_mask = helpers.load_patient_images(patient_id, src_dir, "*_m.png")
    delete_indices = []
    for index, row in df_nodule_predictions.iterrows():
        z_perc = row["coord_z"]
        y_perc = row["coord_y"]
        center_x = int(round(row["coord_x"] * patient_mask.shape[2]))
        center_y = int(round(y_perc * patient_mask.shape[1]))
        center_z = int(round(z_perc * patient_mask.shape[0]))

        mal_score = row["diameter_mm"]
        start_y = center_y - view_size / 2
        start_x = center_x - view_size / 2
        nodule_in_mask = False
        for z_index in [-1, 0, 1]:
            img = patient_mask[z_index + center_z]
            start_x = int(start_x)
            start_y = int(start_y)
            view_size = int(view_size)
            img_roi = img[start_y:start_y+view_size, start_x:start_x + view_size]
            if img_roi.sum() > 255:  # more than 1 pixel of mask.
                nodule_in_mask = True

        if not nodule_in_mask:
            if mal_score > 0:
                mal_score *= -1
            df_nodule_predictions.loc[index, "diameter_mm"] = mal_score
        else:
            if center_z < 30:
                if mal_score > 0:
                    mal_score *= -1
                df_nodule_predictions.loc[index, "diameter_mm"] = mal_score


    df_nodule_predictions.drop(df_nodule_predictions.index[delete_indices], inplace=True)
    return df_nodule_predictions

holdout_ext = ""
magnification=1
flip=False
holdout_no=-1
ext_name="luna_fs"
fold_count=2
flip_ext = ""
model_path = os.environ['DATA_DIR']+'/models/model_luna16_full__fs_best.hd5'
if flip:
    flip_ext = "_flip"

model = step2_train_nodule_detector.get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=model_path)

def predict_cubes(patient_ids,z0,model_path,magnification=1, flip=False, holdout_no=-1, ext_name="", fold_count=2):
    sw = helpers.Stopwatch.start_new()
    all_predictions_csv = []
    for patient_index, patient_id in enumerate(reversed(patient_ids)):
        if "metadata" in patient_id:
            continue
        if "labels" in patient_id:
            continue
        patient_img = helpers.load_patient_images(patient_id, LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png", [])
        if magnification != 1:
            patient_img = helpers.rescale_patient_images(patient_img, (1, 1, 1), magnification)

        patient_mask = helpers.load_patient_images(patient_id, LUNA16_EXTRACTED_IMAGE_DIR, "*_m.png", [])
        if magnification != 1:
            patient_mask = helpers.rescale_patient_images(patient_mask, (1, 1, 1), magnification, is_mask_image=True)

            # patient_img = patient_img[:, ::-1, :]
            # patient_mask = patient_mask[:, ::-1, :]

        step = PREDICT_STEP
        CROP_SIZE = CUBE_SIZE
        # CROP_SIZE = 48

        predict_volume_shape_list = [0, 0, 0]
        for dim in range(3):
            dim_indent = 0
            while dim_indent + CROP_SIZE < patient_img.shape[dim]:
                predict_volume_shape_list[dim] += 1
                dim_indent += step

        predict_volume_shape = (predict_volume_shape_list[0], predict_volume_shape_list[1], predict_volume_shape_list[2])
        predict_volume = numpy.zeros(shape=predict_volume_shape, dtype=float)
        done_count = 0
        skipped_count = 0
        batch_size = 128
        batch_list = []
        batch_list_coords = []
        patient_predictions_csv = []
        annotation_index = 0
        if z0 < 0:
            z0 = 0
            z1 = predict_volume_shape[0]
        else:
            z1 = z0+1
        for z in range(z0, z1):
            for y in range(0, predict_volume_shape[1]):
                for x in range(0, predict_volume_shape[2]):
                    #if cube_img is None:
                    cube_img = patient_img[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]
                    cube_mask = patient_mask[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]

                    if cube_mask.sum() < 2000:
                        skipped_count += 1
                    else:
                        if flip:
                            cube_img = cube_img[:, :, ::-1]

                        if CROP_SIZE != CUBE_SIZE:
                            cube_img = helpers.rescale_patient_images2(cube_img, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                            # helpers.save_cube_img("c:/tmp/cube.png", cube_img, 8, 4)
                            # cube_mask = helpers.rescale_patient_images2(cube_mask, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))

                        img_prep = prepare_image_for_net3D(cube_img)
                        batch_list.append(img_prep)
                        batch_list_coords.append((z, y, x))
                        if len(batch_list) % batch_size == 0:
                            batch_data = numpy.vstack(batch_list)
                            p = model.predict(batch_data, batch_size=batch_size)
                            for i in range(len(p[0])):
                                p_z = batch_list_coords[i][0]
                                p_y = batch_list_coords[i][1]
                                p_x = batch_list_coords[i][2]
                                nodule_chance = p[0][i][0]
                                predict_volume[p_z, p_y, p_x] = nodule_chance
                                if nodule_chance > P_TH:
                                    p_z = p_z * step + CROP_SIZE / 2
                                    p_y = p_y * step + CROP_SIZE / 2
                                    p_x = p_x * step + CROP_SIZE / 2
                                    p_z_perc = round(float(p_z) / patient_img.shape[0], 4)
                                    p_y_perc = round(float(p_y) / patient_img.shape[1], 4)
                                    p_x_perc = round(float(p_x) / patient_img.shape[2], 4)
                                    diameter_mm = round(p[1][i][0], 4)
                                    # diameter_perc = round(2 * step / patient_img.shape[2], 4)
                                    diameter_perc = round(2 * step / patient_img.shape[2], 4)
                                    diameter_perc = round(diameter_mm / patient_img.shape[2], 4)
                                    nodule_chance = round(nodule_chance, 4)
                                    patient_predictions_csv_line = [annotation_index,p_x,p_y,p_z,p_x_perc, p_y_perc, p_z_perc, diameter_perc, nodule_chance, diameter_mm]
                                    patient_predictions_csv.append(patient_predictions_csv_line)
                                    all_predictions_csv.append([patient_id] + patient_predictions_csv_line)
                                    annotation_index += 1

                            batch_list = []
                            batch_list_coords = []
                    done_count += 1


        df = pandas.DataFrame(patient_predictions_csv, columns=["anno_index","ax","ay","az", "coord_x", "coord_y", "coord_z", "diameter", "nodule_chance", "diameter_mm"])
        filter_patient_nodules_predictions(df, patient_id, CROP_SIZE * magnification)
        return df

patient_ids = []
test_images=os.environ['DATA_DIR']+'/cniit-extracted/'
src_dir = test_images
k = 0
for file_name in os.listdir(test_images):
    if "labels" in file_name:
        continue
    if not os.path.isdir(test_images + file_name):
        continue
    patient_ids.append(file_name)
    print('%s - %d' % (file_name,k))
    k = k + 1

import matplotlib.patches as patches
def slice_images(patient,p):
    df = predict_cubes([patient_ids[patient]],-1,os.environ['DATA_DIR']+'/models/model_luna16_full__fs_best.hd5', magnification=1, flip=False, holdout_no=None, ext_name="luna16_fs")
    groups = df.groupby(['az'])
    zdata = []
    for name,data in groups:
        zdata.append(data)
    def slice_image(z_index):
        j = 0
        fig = plt.figure()
        a=fig.add_subplot(111)
        print(patient_ids[patient])
        print(zdata[z_index][['coord_x','coord_y','coord_z','diameter','nodule_chance']])
        for index, row in zdata[z_index].iterrows():
            z = int(row["az"])
            x = int(row["ax"])
            y = int(row["ay"])
            if j == 0:
                plt.imshow(p[z], cmap=plt.cm.gray_r, interpolation='nearest')
            j = j + 1
            a.add_patch(patches.Circle((x,y),row["diameter_mm"],fill=False,edgecolor="red"))
        plt.axis('off')
        plt.show()
    usemax = (len(zdata)-1) if len(zdata)>1 else 0
    interact(slice_image, z_index=IntSlider(min=0,max=usemax,step=1,continuous_update=False))
def browse_images():
    def view_image(patient):
        p = helpers.load_patient_images(patient_ids[patient], src_dir, "*_i.png")
        #plt.imshow(p[0], cmap=plt.cm.gray_r, interpolation='nearest')
        #plt.show()
        slice_images(patient,p)
    usemax = (len(patient_ids)-1) if len(patient_ids)>1 else 0
    interact(view_image, patient=IntSlider(min=0,max=usemax,step=1,continuous_update=False))


