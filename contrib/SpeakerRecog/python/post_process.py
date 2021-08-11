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
import numpy as np
import os
import torch
import torch.nn.functional as F


def speaker_recognition(embedding, speaker_name, enroll_embedding_dir, thres=0.7):
    """speaker recognition
    Args:
        embedding: The embedding of the speaker to be identified.
        speaker_name: The name of the speaker to be identified is the file name of the speaker audio.
                     Use speaker_name as the registered name when no current speaker exists in the voiceprint library.
        enroll_embedding_dir: Voice print library address
        thres: The threshold is obtained by calculating the error rate of the development set.
    Return:
        result:
    """
    enroll_speakers = os.listdir(enroll_embedding_dir)
    if len(enroll_speakers) == 0:
        # If the current voice print library is empty, register the current speaker
        print("There is no speaker in the voice print library!")
        print("Register the current speaker...")
        np.save(os.path.join(enroll_embedding_dir, speaker_name + ".npy"), embedding)
        print("{} registration complete!".format(speaker_name))
    else:
        embedding = torch.tensor(embedding.reshape(1, -1))
        score_list = []
        for enroll_speaker in enroll_speakers:
            enroll_embedding = np.load(os.path.join(enroll_embedding_dir, enroll_speaker))
            enroll_embedding = torch.tensor(enroll_embedding.reshape(1, -1))
            score = F.cosine_similarity(embedding, enroll_embedding)
            score = score.numpy()
            score_list.append(score)
        max_score = max(score_list)
        if max_score < thres:
            print("The speaker is not included in the voice print library")
            print("Register the current speaker...")
            np.save(os.path.join(enroll_embedding_dir, speaker_name + ".npy"), embedding)
            print("{} registration complete!".format(speaker_name))
        else:
            max_index = score_list.index(max_score)
            result = enroll_speakers[max_index].split(".")[0]
            print("The current audio {}.wav  is from speaker {}".format(speaker_name, result))
