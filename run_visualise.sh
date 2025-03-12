# visualise boxes
CUDA_VISIBLE_DEVICES=0 python visualise_boxes.py --config-file configs/diverse_weather.yaml --eval-only MODEL.WEIGHTS all_outs/diverse_weather-dynamic/model_best.pth
