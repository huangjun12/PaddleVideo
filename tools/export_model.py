# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import sys
import os.path as osp

import paddle
import paddle.nn.functional as F
from paddle.jit import to_static
from paddle.static import InputSpec

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from paddlevideo.modeling.builder import build_model
from paddlevideo.utils import get_config


def parse_args():

    parser = argparse.ArgumentParser("PaddleVideo export model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')

    parser.add_argument("-p",
                        "--pretrained_params",
                        default='./best.pdparams',
                        type=str,
                        help='params path')
    parser.add_argument("-o",
                        "--output_path",
                        type=str,
                        default="./inference",
                        help='output path')
#    parser.add_argument("--img_size", type=int, default=224, help='image size')
#    parser.add_argument("--num_seg",
#                        type=int,
#                        default=8,
#                        help='the number of segments')

    return parser.parse_args()


def trim_config(cfg, args):
    """
    Reuse the trainging config will bring useless attributes, such as: backbone.pretrained model.
    and some build phase attributes should be overrided, such as: backbone.num_seg.
    Trim it here.
    """
    model_name = cfg.model_name
#    cfg = cfg.MODEL
    if cfg.MODEL.backbone.get('pretrained'):
        cfg.MODEL.backbone.pretrained = ""

#    if model_name == "TSM":
#        cfg.backbone.num_seg = args.num_seg

    return cfg, model_name


def get_input_spec(cfg, model_name):
    if model_name in ['ppTSM']:
        input_spec=[[InputSpec(shape=[None, cfg.num_seg, 3, cfg.img_size, cfg.img_size],
                               dtype='float32'),]
                   ]
    elif model_name in ['BMN']:
        input_spec=[[InputSpec(shape=[None, cfg.feat_dim, cfg.tscale], 
                               dtype='float32', name='feat_input'),
                   ]]
    return input_spec


def main():
    args = parse_args()
    cfg, model_name = trim_config(get_config(args.config, show=False), args)
    print(f"Building model({model_name})...")
    model = build_model(cfg.MODEL)
    assert osp.isfile(
        args.pretrained_params
    ), f"pretrained params ({args.pretrained_params} is not a file path.)"

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    print(f"Loading params from ({args.pretrained_params})...")
    params = paddle.load(args.pretrained_params)
    model.set_dict(params)
    model.eval()

    input_spec = get_input_spec(cfg.INFER, model_name)
    model = to_static(model, input_spec=input_spec)
    paddle.jit.save(model, osp.join(args.output_path, model_name))
    print(
        f"model ({model_name}) has been already saved in ({args.output_path}).")


if __name__ == "__main__":
    main()
