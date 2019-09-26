import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import scipy.misc, numpy, shutil, os, nibabel
import sys, getopt
import zipfile, os
import gzip
import math

# Load cognitive assesment spreadsheet, unzip nii images, convert nii images to png,
# organize into session folder structure
# Output new simplifed csv with session run and AD/healthy

# load import congitive assessment spreadsheet to pandas
path = input('Path to cognitive assesment csv: ')
header = ['ADRC_ADRCCLINICALDATA ID', 'dx1']
df = pd.read_csv(path, header = None, skiprows = 1, usecols = [0,8], names = header)


subject_regex = re.compile("OAS(?P<order>[0-9]+)")
subjects = [subject_regex.search(r).group(1) for r in df["ADRC_ADRCCLINICALDATA ID"]]
df['Subject'] = pd.Series(subjects, index=df.index)

date_regex = re.compile("d(?P<order>[0-9]+)")
dates = [date_regex.search(r).group(1) for r in df["ADRC_ADRCCLINICALDATA ID"]]
df['Date'] = pd.Series(list(map(int,dates)), index=df.index)


#unzip nii package
package_path = input('Path to nii zip: ')
with zipfile.ZipFile(package_path, "r") as zip_ref:
    zip_ref.extractall("./Data/Nii")


#unzip individual run nii pakcages
nii_package_path = "./Data/Nii"
output_path = "./Data/Nii2"

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("Created ouput directory: " + output_path)

for root, directories, filenames in os.walk(nii_package_path):
    for filename in filenames:
        run_package = os.path.join(root, filename)
        with gzip.open(run_package, 'rb') as s_file, open(os.path.join(output_path, filename[:-3]), 'wb') as d_file:
            shutil.copyfileobj(s_file, d_file, 65536)
            #shutil.move(d_file, output_path)
#shutil.rmtree("./Data/Nii")


#convert nii to png (borrowed from nii2png.py)
outputfile = "./Data/Images"
rotate_angle = int(input('Enter rotate angle (0 90 180 270): '))

if not (rotate_angle == 0 or rotate_angle == 90 or rotate_angle == 180 or rotate_angle == 270):
    print('You must enter a value that is either 90, 180, or 270. Quitting...')
    sys.exit()

data_path = "./Data/Nii2"
for root, directories, filenames in os.walk(data_path):
    for filename in filenames:
        if "T2" in filename or "echo" in filename:
            print(filename)
            continue

        nii_path = os.path.join(root, filename)
        image_array = nibabel.load(nii_path).get_data()
        print("Input file is ", nii_path)
        #print(len(image_array.shape))

        if len(image_array.shape) == 3:
            # set 4d array dimension values
            nx, ny, nz = image_array.shape

            if not os.path.exists(outputfile):
                os.makedirs(outputfile)
                print("Created ouput directory: " + outputfile)

            image_folder = os.path.join(outputfile, filename[:-8])

            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
                print("Created ouput directory: " + image_folder)

            print('Reading NIfTI file...')

            total_slices = image_array.shape[2]

            slice_counter = 0
            # iterate through slices
            for current_slice in range(0, total_slices):
                # alternate slices
                if (slice_counter % 1) == 0:
                    # rotate or no rotate
                    if rotate_angle == 90 or rotate_angle == 180 or rotate_angle == 270:
                        if rotate_angle == 90:
                            data = numpy.rot90(image_array[:, :, current_slice])
                        elif ask_rotate_num == 180:
                            data = numpy.rot90(numpy.rot90(image_array[:, :, current_slice]))
                        elif ask_rotate_num == 270:
                            data = numpy.rot90(numpy.rot90(numpy.rot90(image_array[:, :, current_slice])))
                    else:
                        data = image_array[:, :, current_slice]

                    #alternate slices and save as png
                    if (slice_counter % 1) == 0:
                        #print('Saving image...')
                        image_name = nii_path[:-4] + "_z" + "{:0>3}".format(str(current_slice+1))+ ".png"
                        scipy.misc.imsave(image_name, data)
                        #print('Saved.')

                        #move images to folder
                        #print('Moving image...')
                        src = image_name
                        shutil.move(src, image_folder)
                        slice_counter += 1
                        #print('Moved.')

            print('Finished converting {}'.format(filename))
        else:
            print('Not a 3D Image. Please try again.')


runs_list = next(os.walk('./Data/Images'))[1]
AD_list = []

for run in list(runs_list):
    # if dx1 starts with AD then patient has AD
    # either 0 or 1
    # print(run)
    subject_regex = re.compile("OAS(?P<order>[0-9]+)")
    subject = subject_regex.search(run).group(1)

    date_regex = re.compile("d(?P<order>[0-9]+)")
    date = date_regex.search(run).group(1)

    #print(subject, date)

    subject_df = df.loc[df["Subject"] == subject]
    #print(subject_df)

    #label = min(list(map(int, subject_df['Date'])), key=lambda x:(int(date)-x))
    df_dates = subject_df['Date']
    #print(type(df_dates))

    if int(date) > df_dates.iloc[-1]:
        label_date = df_dates.iloc[-1]
    else:
        label_date = min(i for i in df_dates if i >= int(date))

    label = subject_df.loc[df['Date'] == label_date, 'dx1'].item()
    if type(label) == float:
        print(subject, label)
        runs_list.remove(run)
        continue

#     if type(label) != str:
#         print(type(label))

    AD = 1 if label.startswith('AD') else 0
    AD_list.append(AD)


Labelsdf = pd.DataFrame()
#df = pd.DataFrame(columns=['Run', 'AD'])
Labelsdf['Run'] = pd.Series(runs_list)
Labelsdf['AD'] = pd.Series(AD_list)
Labelsdf.to_csv("./Data/Labels.csv")
