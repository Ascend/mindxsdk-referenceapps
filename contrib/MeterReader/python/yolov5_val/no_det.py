'''
4.no_det.py:过滤没有检测到目标的图片文件
'''


import sys
import os
import glob

'''
validate_voc_PATH:验证集的voc数据
sdk_predict_voc_PATH：sdk预测的voc数据
检查是否有的数据是无目标的
'''
cur_path = os.path.abspath(os.path.dirname(__file__))
validate_voc_PATH = os.path.join(cur_path, 'det_val_data', 'det_val_voc').replace('\\', '/')
sdk_predict_voc_PATH = os.path.join(cur_path, 'det_val_data', 'det_sdk_voc').replace('\\', '/')

# validate_voc_PATH = r'./data/meter/val_voc'
# sdk_predict_voc_PATH = r'./data/meter/pre_voc'

backup_folder = 'backup_no_matches_found'  # must end without slash

os.chdir(validate_voc_PATH)
validate_voc_files = glob.glob('*.txt')
if len(validate_voc_files) == 0:
    print("Error: no .txt files found in", validate_voc_PATH)
    sys.exit()
os.chdir(sdk_predict_voc_PATH)
sdk_predict_voc_files = glob.glob('*.txt')
if len(sdk_predict_voc_files) == 0:
    print("Error: no .txt files found in", sdk_predict_voc_PATH)
    sys.exit()

validate_voc_files = set(validate_voc_files)
sdk_predict_voc_files = set(sdk_predict_voc_files)
print('total ground-truth files:', len(validate_voc_files))
print('total detection-results files:', len(sdk_predict_voc_files))
print()

validate_voc_backup = validate_voc_files - sdk_predict_voc_files
sdk_predict_voc_backup = sdk_predict_voc_files - validate_voc_files




# validate_voc
if not validate_voc_files:
    print('No backup required for', validate_voc_backup)
else:
    os.chdir(validate_voc_backup)
    ## create the backup dir if it doesn't exist already
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    for file in validate_voc_files:
        os.rename(file, backup_folder + '/' + file)

# sdk_predict_voc
if not sdk_predict_voc_files:
    print('No backup required for', sdk_predict_voc_backup)
else:
    os.chdir(sdk_predict_voc_backup)
    ## create the backup dir if it doesn't exist already
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    for file in sdk_predict_voc_files:
        os.rename(file, backup_folder + '/' + file)


if validate_voc_backup:
    print('total ground-truth backup files:', len(validate_voc_backup))
if sdk_predict_voc_backup:
    print('total detection-results backup files:', len(sdk_predict_voc_backup))

intersection_files = validate_voc_files & sdk_predict_voc_files
print('total intersected files:', len(intersection_files))
print("Intersection completed!")