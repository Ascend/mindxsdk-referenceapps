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


import cv2
import numpy as np
import faiss
import MxpiDataType_pb2 as MxpiDataType
import scipy.ndimage as ndimage


class FaissNN(object):
    def __init__(self, num_workers: int = 4) -> None:
        """FAISS Nearest neighbourhood search.

        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.search_index = None

    def load(self, path):
        self.search_index = faiss.read_index(path)

    def run(
            self,
            n_nearest_neighbours,
            query_features: np.ndarray,
    ):
        """
        Returns distances and indices of nearest neighbour search.

        Args:
            query_features: Features to retrieve.
            index_features: [optional] Index features to search in.
        """
        return self.search_index.search(query_features, n_nearest_neighbours)

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None




def unpatch_scores(x, batchsize):
    return x.reshape(batchsize, -1, *x.shape[1:])

def score_max(x):
    while x.ndim > 1:
        x = np.max(x, axis=-1)
    return x


def preprocess(image):
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    print("prepare_bin success")

    # gen tensor data
    mxpi_tensor_pack_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package_vec = mxpi_tensor_pack_list.tensorPackageVec.add()

    print("preprocess tensor success")

    # add noise data
    tensorvec_noise = tensor_package_vec.tensorVec.add()
    tensorvec_noise.memType = 1
    tensorvec_noise.deviceId = 0
    # bs为1，4指向float32数据
    tensorvec_noise.tensorDataSize = int(1 * 3 * 224 * 224 * 4)
    tensorvec_noise.tensorDataType = 0
    for i in image.shape:
        tensorvec_noise.tensorShape.append(i)
    tensorvec_noise.dataStr = image.tobytes()

    return mxpi_tensor_pack_list


def norm(scores):
    scores = np.array(scores)
    min_scores = scores.min(axis=-1).reshape(-1, 1)
    max_scores = scores.max(axis=-1).reshape(-1, 1)
    scores = (scores - min_scores) / (max_scores - min_scores)
    scores = np.mean(scores, axis=0)
    return scores


class RescaleSegmentor:
    def __init__(self, target_size=224):
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):
        _scores = patch_scores
        _scores = np.transpose(_scores, (1, 2, 0))
        _scores = cv2.resize(_scores, self.target_size)
        _scores = np.expand_dims(_scores, axis=0)
        patch_scores = _scores

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]
