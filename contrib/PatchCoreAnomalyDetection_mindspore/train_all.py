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


import argparse
import datetime
import logging
import os
import random
import time
import yaml

import mindspore
import numpy as np

from mvtec import createDataset
from mindspore import context, load_checkpoint, load_param_into_net, nn
from models import wide_resnet101_2
from network import PatchCore
from mindspore import export
from mindspore import Tensor
from mindspore import ops

from tools import reduce_features, compute_greedy_coreset_indices, NearestNeighbourScorer, FaissNN, RescaleSegmentor, \
    PatchMaker, select_topK, norm, compute_imagewise_retrieval_metrics, \
    compute_pixelwise_retrieval_metrics, compute_and_store_final_results, create_storage_folder

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='train')

parser.add_argument('--results', type=str, default="results")
parser.add_argument('--data', '-d', type=str, default="bottle")
parser.add_argument('--layer', '-le', type=str, default="layer2")
parser.add_argument('--backbone', '-b', type=str, default="wideresnet101")
parser.add_argument("--resize", type=int, default=256)
parser.add_argument("--imagesize", type=int, default=224)
parser.add_argument("--patchsize", type=int, default=3)

parser.add_argument('--num_epochs', type=int, default=1, help='Epoch size')
parser.add_argument('--gpu', type=int, default=0, help='Device id')
parser.add_argument('--percentage', '-p', type=float, default=0.01, help='coreset percentage')
parser.add_argument('--dataset_path', type=str, default="/data/jtc/", help='Dataset path')

args = parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend',
                        save_graphs=False)
    context.set_context(device_id=args.gpu)

    cfg = open("config.yaml", 'r', encoding='utf-8')
    data_dict = yaml.safe_load(cfg)
    cfg.close()
    LOGGER.info(
        "Evaluating dataset ..."
    )
    result_path = create_storage_folder(
        args.results, "exp", mode="iterate"
    )

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)

    network = wide_resnet101_2()

    param_dict = load_checkpoint("PatchCore_wideresnet101.ckpt")
    load_param_into_net(network, param_dict, strict_load=False)

    for p in network.trainable_params():
        p.requires_grad = False

    # export model
    model_layer2 = PatchCore(network, "layer2")
    model_layer3 = PatchCore(network, "layer3")

    input_tensor = Tensor(np.ones([1, 3, args.imagesize, args.imagesize]).astype(np.float32))
    export(model_layer2, input_tensor,
           file_name=f'{args.backbone}_layer2', file_format='AIR')
    export(model_layer3, input_tensor,
           file_name=f'{args.backbone}_layer3', file_format='AIR')

    result_collect = []

    print("***************start train***************")
    for idx, key in enumerate(data_dict):
        train_dataset, test_dataset, _, _ = createDataset(args.dataset_path, key, data_dict[key]["resize"],
                                                          data_dict[key]["imagesize"])
        LOGGER.info(f"{idx + 1}/{len(data_dict)}\t{key}")
        if data_dict[key]["layer"] == "layer2":
            model = model_layer2
        else:
            model = model_layer3
        model = mindspore.Model(model)
        data_iter = train_dataset.create_dict_iterator()
        step_size = train_dataset.get_dataset_size()
        features = []
        step = 0
        pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        pool = nn.AvgPool2d(data_dict[key]["patchsize"], 1, pad_mode="valid")
        for data in train_dataset.create_dict_iterator():
            # time
            step += 1
            start = datetime.datetime.fromtimestamp(time.time())
            feature = model.predict(data['image'])
            feature = pad(feature)
            feature = pool(feature)

            patch_num = [[feature.shape[2], feature.shape[3]]]
            feature = ops.transpose(feature, (0, 2, 3, 1))
            feature = feature.reshape(-1, feature.shape[3])
            feature = feature.asnumpy()
            features.append(feature)

        LOGGER.info("##########subsample...###########################")

        features = np.concatenate(features, axis=0)
        LOGGER.info("##########subsample...###########################")
        features = features.astype("float32")
        reduced_features = reduce_features(features)
        sample_indices = compute_greedy_coreset_indices(reduced_features, 10, percentage=data_dict[key]["percentage"])
        sample_indices = sample_indices.tolist()
        features = features[sample_indices]
        nn_method = FaissNN(False, 4)
        anomaly_scorer = NearestNeighbourScorer(n_nearest_neighbours=1, nn_method=nn_method)
        LOGGER.info("##########subsample complete#####################")
        LOGGER.info("construct memory bank")
        anomaly_scorer.fit(detection_features=[features])

        print("***************end train***************")

        print("***************start eval***************")
        test_data_iter = test_dataset.create_dict_iterator()
        scores = []
        scores2 = []
        segmentations = []
        labels_gt = []
        masks_gt = []
        anomaly_segmentor = RescaleSegmentor(
            target_size=(data_dict[key]["imagesize"], data_dict[key]["imagesize"])
        )
        step = 0
        patch_maker = PatchMaker(patchsize=data_dict[key]["patchsize"], stride=1)
        for step, image in enumerate(test_data_iter):
            step += 1
            start = datetime.datetime.fromtimestamp(time.time())
            labels_gt.extend(image["is_anomaly"].asnumpy().tolist())
            masks_gt.extend(image["mask"].asnumpy().tolist())
            image = image["image"]
            batchsize = image.shape[0]
            features = model.predict(image)

            features = pad(features)
            features = pool(features)
            patch_shapes = [[features.shape[2], features.shape[3]]]
            features = ops.transpose(features, (0, 2, 3, 1))
            features = features.reshape(-1, features.shape[3])

            features = features.asnumpy()

            patch_scores = image_scores = anomaly_scorer.predict([features])[0]
            image_scores = patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = patch_maker.score(image_scores)

            patch_scores = patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = anomaly_segmentor.convert_to_segmentation(patch_scores)

            _scores, _masks, _scores2 = [score for score in image_scores], [mask for mask in masks], [score2 for score2
                                                                                                      in patch_scores]
            for score, mask, score2 in zip(_scores, _masks, _scores2):
                scores.append(score)
                scores2.append(score2)
                segmentations.append(mask)

        scores = np.array(scores)
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = scores.mean(axis=0)
        scoresmax = []
        scorestopK = []
        scorespx = []
        scoresdx = []
        scoresrangex = []
        for i in range(len(scores2)):
            scoresmax.append(np.max(np.array(scores2[i])).item())
            scorespx.append(np.mean(np.array(scores2[i])).item())
            scoresdx.append(np.var(np.array(scores2[i])).item())
            scoresrangex.append(np.max(np.array(scores[i])).item() - np.min(scores[i]).item())

        scorestopK5 = select_topK(5, scores2)
        scorestopK8 = select_topK(8, scores2)
        scorestopK10 = select_topK(10, scores2)
        scorestopK15 = select_topK(15, scores2)
        scorestopK20 = select_topK(20, scores2)
        scorestopK40 = select_topK(40, scores2)
        scorestopK60 = select_topK(60, scores2)
        scorestopK100 = select_topK(100, scores2)
        scoresmax = norm(scoresmax)
        scorespx = norm(scorespx)
        scoresdx = norm(scoresdx)
        scoresrangex = norm(scoresrangex)

        anomaly_labels = [
            x != 0 for x in labels_gt
        ]

        LOGGER.info("Computing evaluation metrics.")
        segmentations = np.array(segmentations)
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1).max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)
        segmentations = np.ascontiguousarray(segmentations)
        # Compute PRO score & PW Auroc for all images
        pixel_scores = compute_pixelwise_retrieval_metrics(
            segmentations, masks_gt
        )
        full_pixel_auroc = pixel_scores["auroc"]

        auroc = compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )["auroc"]

        auroc_max = compute_imagewise_retrieval_metrics(
            scoresmax, anomaly_labels
        )["auroc"]

        auroc_dx = compute_imagewise_retrieval_metrics(
            scoresdx, anomaly_labels
        )["auroc"]

        auroc_px = compute_imagewise_retrieval_metrics(
            scorespx, anomaly_labels
        )["auroc"]

        auroc_topK5 = compute_imagewise_retrieval_metrics(
            scorestopK5, anomaly_labels
        )["auroc"]

        auroc_topK8 = compute_imagewise_retrieval_metrics(
            scorestopK8, anomaly_labels
        )["auroc"]

        auroc_topK10 = compute_imagewise_retrieval_metrics(
            scorestopK10, anomaly_labels
        )["auroc"]

        auroc_topK15 = compute_imagewise_retrieval_metrics(
            scorestopK15, anomaly_labels
        )["auroc"]

        auroc_topK20 = compute_imagewise_retrieval_metrics(
            scorestopK20, anomaly_labels
        )["auroc"]

        auroc_topK40 = compute_imagewise_retrieval_metrics(
            scorestopK40, anomaly_labels
        )["auroc"]

        auroc_topK60 = compute_imagewise_retrieval_metrics(
            scorestopK60, anomaly_labels
        )["auroc"]

        auroc_topK100 = compute_imagewise_retrieval_metrics(
            scorestopK100, anomaly_labels
        )["auroc"]

        result_collect.append(
            {
                "dataset_name": key,
                "instance_auroc": auroc,
                "instance_auroc_max": auroc_max,
                "instance_auroc_dx": auroc_dx,
                "instance_auroc_px": auroc_px,
                "instance_auroc_topK5": auroc_topK5,
                "instance_auroc_topK8": auroc_topK8,
                "instance_auroc_topK10": auroc_topK10,
                "instance_auroc_topK15": auroc_topK15,
                "instance_auroc_topK20": auroc_topK20,
                "instance_auroc_topK40": auroc_topK40,
                "instance_auroc_topK60": auroc_topK60,
                "instance_auroc_topK100": auroc_topK100,
                "full_pixel_auroc": full_pixel_auroc,
            }
        )

        for key_res, item in result_collect[-1].items():
            if key_res != "dataset_name":
                LOGGER.info("{0}: {1:3.3f}".format(key_res, item))

        patchcore_save_path = os.path.join(
            result_path, "models", key
        )
        os.makedirs(patchcore_save_path, exist_ok=True)
        print("Saving PatchCore data.")
        anomaly_scorer.save(
            patchcore_save_path, save_features_separately=False, prepend=""
        )

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    compute_and_store_final_results(
        result_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )
