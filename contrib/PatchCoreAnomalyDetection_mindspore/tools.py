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


import csv
import datetime
import os
import pickle
import stat
import time
import logging
import faiss
import mindspore
from mindspore import nn
import numpy as np
import tqdm
from mindspore import Tensor
import mindspore as ms
from mindspore import ops
import scipy.ndimage as ndimage
from sklearn import metrics

LOGGER = logging.getLogger(__name__)


def reduce_features(features):
    if features.shape[1] == 128:
        return features
    features = mindspore.Tensor.from_numpy(features)
    mapper = nn.Dense(
        features.shape[1], 128, has_bias=False
    )
    _ = mapper
    return mapper(features)


def compute_greedy_coreset_indices(features, number_of_starting_points, percentage=0.01):
    """Runs approximate iterative greedy coreset selection.

    This greedy coreset implementation does not require computation of the
    full N x N distance matrix and thus requires a lot less memory, however
    at the cost of increased sampling times.

    Args:
        features: [NxD] input feature bank to sample.
    """
    number_of_starting_points = np.clip(
        number_of_starting_points, None, len(features)
    )
    start_points = np.random.choice(
        len(features), number_of_starting_points, replace=False
    ).tolist()

    approximate_distance_matrix = compute_batchwise_differences(
        features, features[start_points]
    )

    approximate_coreset_anchor_distances = approximate_distance_matrix.mean(axis=1).reshape(-1, 1)
    coreset_indices = []
    num_coreset_samples = int(len(features) * percentage)
    for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
        select_idx = approximate_coreset_anchor_distances.argmax().asnumpy().item()

        coreset_indices.append(select_idx)

        approximate_coreset_anchor_distances = approximate_coreset_anchor_distances.reshape(-1, 1)

        coreset_select_distance = compute_batchwise_differences(
            features, ops.gather(features, Tensor([select_idx]), 0)  # noqa: E203
        )

        approximate_coreset_anchor_distances = ops.Concat(axis=-1)(
            [approximate_coreset_anchor_distances, coreset_select_distance],
        )
        approximate_coreset_anchor_distances = ops.ArgMinWithValue(axis=1)(
            approximate_coreset_anchor_distances
        )[1]

    return np.array(coreset_indices)


def compute_greedy_coreset_indices_base(features, percentage=0.01) -> np.ndarray:
    """Runs iterative greedy coreset selection.

    Args:
        features: [NxD] input feature bank to sample.
    """
    distance_matrix = compute_batchwise_differences(features, features)

    coreset_anchor_distances = ops.LpNorm(axis=1)(distance_matrix, dim=1)

    coreset_indices = []
    num_coreset_samples = int(len(features) * percentage)

    for _ in range(num_coreset_samples):
        select_idx = coreset_anchor_distances.argmax().item()
        coreset_indices.append(select_idx)

        coreset_select_distance = distance_matrix[
                                  :, select_idx: select_idx + 1  # noqa E203
                                  ]
        coreset_anchor_distances = ops.Concat(axis=1)(
            [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance]
        )
        coreset_anchor_distances = ops.ArgMinWithValue(axis=1)(coreset_anchor_distances).values
        coreset_anchor_distances = coreset_anchor_distances[1], coreset_anchor_distances[0]

    return np.array(coreset_indices)


bmm = ops.BatchMatMul()
mm = ops.MatMul()
expendDims = ops.ExpandDims()


def compute_batchwise_differences(
        matrix_a, matrix_b
):
    matrix_a1 = expendDims(matrix_a, 1)
    matrix_a2 = expendDims(matrix_a, 2)

    matrix_b1 = expendDims(matrix_b, 1)
    matrix_b2 = expendDims(matrix_b, 2)

    a_times_a = bmm(matrix_a1, matrix_a2).reshape(-1, 1)

    b_times_b = bmm(matrix_b1, matrix_b2).reshape(1, -1)

    a_times_b = mm(matrix_a, matrix_b.T)

    tmp = ops.Sqrt()(ops.clip_by_value(-2 * (a_times_b) + a_times_a + b_times_b, Tensor(0, dtype=ms.float16),
                                       Tensor(65504, dtype=ms.float32)))

    return tmp


class FaissNN(object):
    def __init__(self, num_workers: int = 4):
        """FAISS Nearest neighbourhood search.

        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.search_index = None

    def fit(self, features: np.ndarray) -> None:
        """
        Adds features to the FAISS search index.

        Args:
            features: Array of size NxD.
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self.search_index.add(features)

    def _create_index(self, dimension):
        return faiss.IndexFlatL2(dimension)

    def run(
            self,
            n_nearest_neighbours,
            query_features: np.ndarray,
            index_features: np.ndarray = None,
    ):
        """
        Returns distances and indices of nearest neighbour search.

        Args:
            query_features: Features to retrieve.
            index_features: [optional] Index features to search in.
        """
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1])
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self.search_index, filename)

    def load(self, filename: str) -> None:
        self.search_index = faiss.read_index(filename)

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxC
        return np.mean(
            features.reshape([features.shape[0], features.shape[1], -1]), axis=-1
        )


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        return features.reshape(len(features), -1)


class NearestNeighbourScorer(object):
    def __init__(self, n_nearest_neighbours: int, nn_method=FaissNN(4)) -> None:
        """
        Neearest-Neighbourhood Anomaly Scorer class.

        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous pixels.
            nn_method: Nearest neighbour search method.
        """
        self.feature_merger = ConcatMerger()
        self.detection_features = None

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method

        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features):
        """Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.nn_method.fit(self.detection_features)

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    def predict(
            self, query_features
    ):
        """Predicts anomaly score.

        Searches for nearest neighbours of test images in all
        support training images.

        Args:
             detection_query_features: [dict of np.arrays] List of np.arrays
                 corresponding to the test features generated by
                 some backbone network.
        """
        query_features = self.feature_merger.merge(
            query_features,
        )
        query_features = query_features.astype("float32")
        query_distances, query_nns = self.imagelevel_nn(query_features)
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    def save(
            self,
            save_folder: str,
            prepend: str = "",
    ):
        self.nn_method.save(self._index_file(save_folder, prepend))

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))


squeeze = ops.Squeeze(1)


class RescaleSegmentor:
    def __init__(self, target_size=224):
        self.smoothing = 4
        self.target_size = target_size

    def convert_to_segmentation(self, patch_scores):
        if isinstance(patch_scores, np.ndarray):
            patch_scores = mindspore.Tensor.from_numpy(np.ascontiguousarray(patch_scores))
        _scores = patch_scores
        _scores = expendDims(_scores, 1)
        resize_bilinear = nn.ResizeBilinear()
        _scores = resize_bilinear(
            _scores,
            size=self.target_size,
            align_corners=False,
        )
        _scores = squeeze(_scores)
        patch_scores = _scores.asnumpy()

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


def unpatch_scores(x, batchsize):
    return x.reshape(batchsize, -1, *x.shape[1:])


def score_max(x):
    while x.ndim > 1:
        x = x.max(axis=-1)
    return x


def score_max_2(x):
    while x.ndim > 1:
        x = x.reshape(1, 784)
    return x


def select_topk(topK=5, scores2=None):
    scorestopkn = []
    for i, item in enumerate(scores2):
        scores_topk = np.array(scores2[i]).reshape(-1)
        index = np.argsort(scores_topk)[::-1][0:topK]
        TOPK_SUM = 0
        for idx in index:
            TOPK_SUM += scores_topk[idx.item()]
        avg = TOPK_SUM / topK
        avg = avg.item()
        scorestopkn.append(avg)
    scorestopkn = norm(scorestopkn)
    return scorestopkn


def norm(scores):
    scores = np.array(scores)
    min_scores = scores.min(axis=-1).reshape(-1, 1)
    max_scores = scores.max(axis=-1).reshape(-1, 1)
    scores = (scores - min_scores) / (max_scores - min_scores)
    scores = np.mean(scores, axis=0)
    return scores


def compute_imagewise_retrieval_metrics(
        anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return auroc


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    return auroc


def compute_and_store_final_results(
        results_path,
        results,
        row_names=None,
        column_names=None,
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."
    try:
        mean_metrics = {}
        for i, result_key in enumerate(column_names):
            mean_metrics[result_key] = np.mean([x[i] for x in results])
            LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

        savename = os.path.join(results_path, "results.csv")
        MODES = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(savename, os.O_RDWR | os.O_CREAT, MODES), 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            header = column_names
            if row_names is not None:
                header = ["Row Names"] + header

            csv_writer.writerow(header)
            for i, result_list in enumerate(results):
                csv_row = result_list
                if row_names is not None:
                    csv_row = [row_names[i]] + result_list
                csv_writer.writerow(csv_row)
            mean_scores = list(mean_metrics.values())
            if row_names is not None:
                mean_scores = ["Mean"] + mean_scores
            csv_writer.writerow(mean_scores)

        mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    except KeyError:
        print("the data_dict does not have the key!")
        exit()
    return mean_metrics


def create_storage_folder(
        main_folder_path, project_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = project_path
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path
