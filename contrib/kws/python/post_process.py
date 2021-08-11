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
# limitations under the License
import numpy as np
import torch
import torch.nn.functional as F


def ctc_decode(matrix, spike_thres=0.3, score=0.5, continue_frames=10):
    """CTC decoding
    Args:
        matrix: output of the model, decoding matrix [time_step, num_classes]
        spike_thres: The threshold at which a frame is considered a spike
        score: The output threshold for decoding
        continue_frames: Maximum number of consecutive frames of the same label allowed
    Return:
        The decoded sequence
    """
    np.set_printoptions(precision=4, threshold=np.inf,
                        suppress=True)
    # Remove the CTC blank
    matrix = matrix[:, :-1]
    # Gets the initial spike when the sum of all probabilities for a frame is greater than a certain threshold
    init_spike_index = []
    for index, line in enumerate(matrix):
        if sum(line) > spike_thres:
            init_spike_index.append(index)
    # It is assumed that the candidate with the highest probability is the category of the frame
    # Successive identical frames are removed
    # Only the frame with the highest probability is retained
    if len(init_spike_index) == 0:
        return []
    group_list = []
    final_spike_index = []
    group_list.append(init_spike_index[0])
    previous_label = np.argmax(matrix[init_spike_index[0], :])
    for i in range(1, len(init_spike_index)):
        if init_spike_index[i] == init_spike_index[i - 1] + 1:
            current_label = np.argmax(matrix[init_spike_index[i], :])
            if current_label != previous_label or len(group_list) >= continue_frames:
                # Successive frames with the same label are a group, and the frame with the largest probability is taken
                # It also limits a maximum of n consecutive frames that must generate a spike
                max_value_index = np.argmax(np.max(matrix[group_list, :], axis=1))
                final_spike_index.append(group_list[max_value_index])
                # Puts the current frame into a new group
                group_list = list()
                group_list.append(init_spike_index[i])
                previous_label = current_label
            else:
                # Successive frames with the same label are put into the same group
                group_list.append(init_spike_index[i])

        else:
            max_value_index = np.argmax(np.max(matrix[group_list, :], axis=1))
            final_spike_index.append(group_list[max_value_index])
            # Puts the current frame into a new group
            group_list = list()
            group_list.append(init_spike_index[i])
            previous_label = np.argmax(matrix[init_spike_index[i], :])
        if i == len(init_spike_index) - 1:
            max_value_index = np.argmax(np.max(matrix[group_list, :], axis=1))
            final_spike_index.append(group_list[max_value_index])
    matrix = matrix[final_spike_index, :]
    result_list = get_output(matrix, score)
    return result_list


def get_output(matrix, score):
    result_list = []
    for index, line in enumerate(matrix):
        if np.max(line) > score:
            result_list.append(np.argmax(line))
        else:
            result_list.append(len(line) - 1)
    return result_list


def convert_index_to_text(ind2pinyin, keyword_pinyin_dict, pinyin2char, predict_index, return_type="x"):
    """The predictive index is converted to text
    Because homophones may exist, ind is converted to pinyin, which matches the overall pinyin of the keyword,
    and then matches the single word
    Args:
        ind2pinyin: The dictionary of index to pinyin
        keyword_pinyin_dict: The dictionary of Keywords' pinyin
        pinyin2char: The dictionary of pinyin to the word
        predict_index: The predictive index
        return_type: "k" or "x"
                     "k" means return value: "keywordA, keywordB";
                     "x" means return value "XXXXkeywordAkeywordBXXXXX",
                     That is, all non-keyword symbols are denoted by X
    Return:
        The decoded sequence
    """
    result = [ind2pinyin[x] if x in ind2pinyin.keys() else "X" for x in predict_index]
    str_result = " ".join(result)
    for key, value in keyword_pinyin_dict.items():
        str_result = str_result.replace(value, key)
    if return_type == "x":
        for key, value in pinyin2char.items():
            str_result = str_result.replace(key, value)
        str_result = str_result.replace(" ", "")
    else:
        temp = str_result.split(" ")
        str_result = [item for item in temp if item in keyword_pinyin_dict.keys()]
        str_result = ",".join(str_result)
    return str_result


def infer(model_output, seq_len, ind2pinyin, keyword_pinyin_dict, pinyin2char):
    """ inference
    Args:
        model_output:  The output of OM model
        seq_len:  The actual output length of the audio file after the model
        ind2pinyin:  The dictionary of index to pinyin
        keyword_pinyin_dict:  The dictionary of Keywords' pinyin
        pinyin2char:  The dictionary of pinyin to the word
    Return:
        predict_text
    """
    # reshape
    pred_matrix = model_output.reshape(-1, 14)
    pred_matrix = torch.from_numpy(np.array(pred_matrix))
    decode_matrix = F.softmax(pred_matrix, dim=-1)
    decode_matrix = decode_matrix.numpy()
    # ctc decoding
    predict_id = ctc_decode(decode_matrix[:seq_len, :], score=0.1)
    predict_text = convert_index_to_text(ind2pinyin,
                                         keyword_pinyin_dict,
                                         pinyin2char,
                                         predict_id,
                                         return_type="x")
    return predict_text
