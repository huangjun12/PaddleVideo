#python3.7 tools/export_model.py -c configs/recognition/tsm/pptsm_k400.yaml \
#                                -p data/ppTSM.pdparams \
#                                -o inference/ppTSM

python3.7 tools/export_model.py -c configs/localization/bmn.yaml \
                                -p data/BMN.pdparams \
                                -o inference/BMN
