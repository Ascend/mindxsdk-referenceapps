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
import os
import torch
import torch.nn.functional as F
import numpy as np

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


def get_trials(enroll_dir, eval_dir, trials_path):
    """Recognition, getting trials used to calculate eer"""
    if os.path.exists(trials_path):
        os.remove(trials_path)
    enroll_utters = get_all_type_paths(enroll_dir, ".npy")
    eval_utters = get_all_type_paths(eval_dir, ".npz")
    with open(trials_path, "w", encoding="utf-8") as fw:
        for eval_utter in eval_utters:
            data = np.load(eval_utter)
            eval_embedding = data["embedding"]
            eval_speaker = data["speaker"]
            for enroll_utter in enroll_utters:
                enroll_speaker = os.path.basename(enroll_utter).split(".")[0]
                enroll_embedding = np.load(enroll_utter)

                eval_embedding = torch.tensor(eval_embedding.reshape(1, -1))
                enroll_embedding = torch.tensor(enroll_embedding.reshape(1, -1))
                # Cosine similarity
                score = F.cosine_similarity(eval_embedding, enroll_embedding)
                score = score.numpy()
                # save to trials file
                if str(eval_speaker) == enroll_speaker:
                    text = enroll_speaker + " " + os.path.basename(eval_utter).split(".")[0] \
                           + " %.4f" % score + " target" + "\n"
                else:
                    text = enroll_speaker + " " + os.path.basename(eval_utter).split(".")[0] \
                           + " %.4f" % score + " nontarget" + "\n"
                fw.write(text)


def cal_eer(trails_path):
    """calculate eer"""
    target_scores = []
    nontarget_scores = []
    # read trials file
    lines = open(trails_path, "r", encoding="utf-8").readlines()
    for line in lines:
        line = line.strip().split(" ")
        if line[3] == "target":
            target_scores.append(float(line[2]))
        else:
            nontarget_scores.append(float(line[2]))
    # sort
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)
    target_size = len(target_scores)
    target_position = 0
    for target_position in range(target_size):
        nontarget_size = len(nontarget_scores)
        nontarget_n = nontarget_size * target_position / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break
    threshold = target_scores[target_position]
    print(trails_path)
    print("threshold is --> ", threshold)
    eer = target_position / target_size
    print("eer is --> ", eer)
    return eer
