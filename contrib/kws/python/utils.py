# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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
import pypinyin
import yaml
import json
import os
def get_all_type_paths(file_dir, _type):
    """Gets the address of a file of the specified type in a folder
    Args:
        file_dir: Folder address
        _type: file type(str)
    Return:
        file address(list)
    """
    _file_paths = []
    for root_dir, sub_dir, files in os.walk(file_dir):
        for _file in files:
            if _file.endswith(_type):
                _file_paths.append(os.path.join(root_dir, _file))
    return _file_paths


def generate_label(keyword_list):
    """Generate labels for keywords
    Args:
        keyword_list: keywords list
    Returns:
        pinyin2ind's dictionary, ind2pinyin's dictionary, pinyin2char's dictionary, kws_pinyin_dict's dictionary
    """
    all_kws_pinyin = []
    pinyin2char = dict()
    kws_pinyin_dict = dict()
    kws_label_dict = dict()
    for index, kws in enumerate(keyword_list):
        kws_pinyin = pypinyin.pinyin(kws, heteronym=True)
        kws_pinyin = [x[0] for x in kws_pinyin]
        kws_label_dict[kws] = kws_pinyin
        all_kws_pinyin.extend(kws_pinyin)
        kws_pinyin_dict[kws] = " ".join(kws_pinyin)
        for idx, char in enumerate(kws):
            pinyin2char[kws_pinyin[idx]] = char

    unique_word_pinyin = list(sorted(set(all_kws_pinyin)))
    index = range(len(unique_word_pinyin))
    pinyin2ind = dict(zip(unique_word_pinyin, index))
    ind2pinyin = dict(zip(index, unique_word_pinyin))

    for index, kws in enumerate(keyword_list):
        kws_pinyin = kws_label_dict[kws]
        kws_label = [pinyin2ind[key] for key in kws_pinyin]
        kws_label_dict[kws] = kws_label
    return pinyin2ind, ind2pinyin, pinyin2char, kws_pinyin_dict, kws_label_dict


def read_conf(yaml_path, return_type="all"):
    """Read the configuration file and generate parameters
    Args:
        yaml_path: Save path of the configuration file
    Return:
         The dictionary of parameters
    """
    with open(yaml_path, "r", encoding="utf-8") as fid:
        params = yaml.load(fid, Loader=yaml.SafeLoader)
    keyword_list = params["data"]["keyword_list"].split(" ")
    params["data"]["keyword_list"] = keyword_list
    values = generate_label(keyword_list)
    params["data"]["pinyin2ind"] = values[0]
    params["data"]["ind2pinyin"] = values[1]
    params["data"]["pinyin2char"] = values[2]
    params["data"]["keyword_pinyin_dict"] = values[3]
    params["data"]["keyword_label"] = values[4]
    params["data"]["num_classes"] = len(values[0])+2
    return params if return_type == "all" else params[return_type]


def read_info(info_path, max_duration=20., min_duration=1.):
    """
    Args:
        info_path: Save path of the data information file
        max_duration: (float)
        min_duration: (float)
    Return:
        json
    """
    info = []
    with open(info_path, "r", encoding="utf-8") as f:
        json_lines = f.readlines()
        for line in json_lines:
            try:
                json_data = json.loads(line)
            except Exception as e:
                raise IOError("Error reading manifest: %s" % str(e))
            if min_duration <= json_data["duration"] <= max_duration:
                info.append(json_data)
    return info
