import cv2
import numpy as np
import glob
import argparse
import tqdm
import os

# Set up argument parser
argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-n', '--run_no',
    metavar='P',
    default=-1,
    type=int,
    help='Run no'
)
args = argparser.parse_args()

TOTAL_RUNS = 11
if args.run_no != -1:
    TOTAL_RUNS = args.run_no

# Create output directory if it doesn't exist
if not os.path.exists('Videos/'):
    os.makedirs('Videos/')

for i in tqdm.tqdm(range(1, TOTAL_RUNS + 1)): 
    RUN_NO = i
    
    # FIXED: Removed 'with_cbf_dynamic_updated/' to match your current root directory
    INPUT_FOLDER = 'run' + str(RUN_NO) + '_video'
    OUTPUT_FILE = 'Videos/run' + str(RUN_NO) + '_video.mp4'
    
    img_array = []
    
    # Find all PNG files in the folder to determine the count
    file_list = glob.glob(INPUT_FOLDER + '/*.png')
    n_files = len(file_list)
    
    # Skip if the folder is empty or doesn't exist
    if n_files == 0:
        print(f"\nWarning: No images found in {INPUT_FOLDER}. Skipping...")
        continue
        
    file_list = []
    # Assumes frames start at frame_16.png
    for j in range(16, n_files + 16):
        file_list.append(INPUT_FOLDER + '/frame_' + str(j) + '.png')
        
    size = None
    for filename in file_list:
        img = cv2.imread(filename)
        
        # SAFETY CHECK: Only append if the frame successfully loaded
        if img is not None:
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

    # Compile the video if frames were successfully loaded
    if size is not None and len(img_array) > 0:
        out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), 25, size)
        for img in img_array:
            out.write(img)
        out.release()
        print(f"\nSuccessfully created {OUTPUT_FILE}")
    else:
        print(f"\nFailed to create video for run {RUN_NO}")