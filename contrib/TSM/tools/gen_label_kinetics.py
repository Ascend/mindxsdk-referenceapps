# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os


DATASET_PATH = '../dataset'
LABEL_PATH = '../label'

if __name__ == '__main__':
    with open('kinetics_label_map.txt') as f:
        categories = f.readlines()
        categories = [c.strip().replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '') \
                      for c in categories]
    assert len(set(categories)) == 400
    dict_categories = {}
    for j, category in enumerate(categories):
        dict_categories[category] = j

    print(dict_categories)

    files_input = ['kinetics_val.csv']
    files_output = ['val_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        count_cat = {k: 0 for k in dict_categories.keys()}
        with open(os.path.join(LABEL_PATH, filename_input)) as f:
            lines = f.readlines()[1:]
        folders = []
        idx_categories = []
        categories_list = []
        for line in lines:
            line = line.rstrip()
            items = line.split(',')
            folders.append(items[1])
            this_catergory = items[0].replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace(
                             "'", '')
            categories_list.append(this_catergory)
            idx_categories.append(dict_categories.get(this_catergory))
            count_cat[this_catergory] += 1
        print(max(count_cat.values()))
        

        assert len(idx_categories) == len(folders)
        missing_folders = []
        output = []
        for i, folder in enumerate(folders):
            curFolder = folder
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            img_dir = os.path.join(DATASET_PATH, categories_list[i], curFolder)
            if not os.path.exists(img_dir):
                missing_folders.append(img_dir)
            else:
                dir_files = os.listdir(img_dir)
                output.append('%s %d %d' % (os.path.join('test', os.path.join(categories_list[i], curFolder)),\
                              len(dir_files), curIDX))
            print('%d/%d, missing %d' % (i, len(folders), len(missing_folders)))
        f = os.open(os.path.join(LABEL_PATH, filename_output), mode=0o777)
        f.write('\n'.join(output))
        g = os.open(os.path.join(LABEL_PATH, 'missing_' + filename_output), mode=0o777)
        g.write('\n'.join(missing_folders))