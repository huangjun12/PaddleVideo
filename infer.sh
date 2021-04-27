python3.7 tools/predict.py --video_file data/example.avi \
                           --model_file inference/ppTSM/ppTSM.pdmodel \
                           --params_file inference/ppTSM/ppTSM.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
