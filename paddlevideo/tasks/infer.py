# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
from paddlevideo.utils import get_logger
from ..loader.builder import build_dataloader, build_dataset
from ..modeling.builder import build_model
from paddlevideo.utils import load

logger = get_logger("paddlevideo")


def feature_trans(feats):
    batch_size = feats.shape[0]
    feat_pad_list = []
    feat_len_list = []
    feat_mask_list = []

    for i in range(batch_size):
        feat = feats[i].numpy()
        feat_len = feat.shape[0]
        #feat pad step 1. padding
        feat_add = np.zeros((512 - feat.shape[0], feat.shape[1]),
                            dtype=np.float32)
        feat_pad = np.concatenate((feat, feat_add), axis=0).astype("float32")
        #feat pad step 2. mask
        feat_mask_origin = np.ones(feat.shape, dtype=np.float32)
        feat_mask_add = feat_add
        feat_mask = np.concatenate((feat_mask_origin, feat_mask_add),
                                   axis=0).astype("float32")

        feat_pad_list.append(feat_pad)
        feat_len_list.append(feat_len)
        feat_mask_list.append(feat_mask)

    predictor_input = [
        paddle.to_tensor(feat_pad_list),
        paddle.to_tensor(feat_len_list),
        paddle.to_tensor(feat_mask_list)
    ]
    return predictor_input


def infer_model(extractor_cfg, predictor_cfg, weights, parallel=True):
    """Test model entry

    Args:
        cfg (dict): configuration.
        weights (str): weights path to load.
        parallel (bool): Whether to do multi-cards testing. Default: True.

    """
    # 1. Construct model.
    extractor_model = build_model(extractor_cfg.MODEL)
    if parallel:
        extractor_model = paddle.DataParallel(extractor_model)

    predictor_model = build_model(predictor_cfg.MODEL)
    if parallel:
        predictor_model = paddle.DataParallel(predictor_model)

    # 2. Construct dataset and dataloader.
    dataset = build_dataset(
        (extractor_cfg.DATASET.test, extractor_cfg.PIPELINE.test))
    batch_size = extractor_cfg.DATASET.get("test_batch_size", 1)
    places = paddle.set_device('gpu')
    # default num worker: 0, which means no subprocess will be created
    num_workers = extractor_cfg.DATASET.get('num_workers', 0)
    dataloader_setting = dict(batch_size=batch_size,
                              num_workers=num_workers,
                              places=places,
                              drop_last=False,
                              shuffle=False)

    data_loader = build_dataloader(dataset, **dataloader_setting)

    extractor_model.eval()

    state_dicts = load(weights)
    extractor_model.set_state_dict(state_dicts)

    for batch_id, data in enumerate(data_loader):
        if parallel:
            feats = extractor_model._layers.test_step(data)
        else:
            feats = extractor_model.test_step(data)

        #tranform feature
        predictor_input = feature_trans(feats)

        if parallel:
            outputs = predictor_model._layers.test_step(predictor_input)
        else:
            outputs = predictor_model.test_step(predictor_input)
        print("outputs: ", outputs.shape)
