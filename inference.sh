#!/usr/bin/env bash

# python process.py "$@"

model_path="smile/checkpoint/cls10/net_epoch=273.tar"
output_dir="/output/"

# Step 1: perform nuclei inference with SMILE.
# ####################################################
# 10cls
gpu='0'
nr_types=11
type_info_path="smile/type_info/puma_10_type_info.json"
batch_size=2
model_mode="fast"
nr_inference_workers=1
nr_post_proc_workers=2
input_dir="/input/images/melanoma-wsi/"
mem_usage=0.1

# # ####################################################
# # 3cls
# gpu='0'
# nr_types=4
# type_info_path="smile/type_info/puma_3_type_info.json"
# batch_size=2
# model_mode="fast"
# nr_inference_workers=1
# nr_post_proc_workers=2
# input_dir="/input/images/melanoma-wsi/"
# mem_usage=0.1

echo "Running SMILE inference............"

python smile/run_infer.py \
    --gpu=$gpu \
    --nr_types=$nr_types \
    --type_info_path=$type_info_path \
    --batch_size=$batch_size \
    --model_mode=$model_mode \
    --model_path=$model_path \
    --nr_inference_workers=$nr_inference_workers \
    --nr_post_proc_workers=$nr_post_proc_workers \
    tile \
    --input_dir=$input_dir \
    --output_dir=$output_dir \
    --mem_usage=$mem_usage


# # Step 2: perform tissue inference with deeplabv3_plus
# python deeplabv3_plus/get_miou.py


# # Step 2: perform tissue inference with H-SAM
# CUDA_VISIBLE_DEVICES=0 python H-SAM-test/test.py

# Step 2: perform tissue inference with nnUNet-hw
python nnUNet/in.py
# inference
CUDA_VISIBLE_DEVICES=0 python nnUNet/nnunetv2/inference/predict_from_raw_data.py
# convert gt json file into png with correspond name
python nnUNet/output_rename.py


# # Step 2: perform tissue inference with nnunet
# # create the original name and nnunet name list
# python create_convert_dict.py
# # reload tiff image into png and change name
# python image_transfer.py

# # inference
# cd ./nnunetv2/inference/
# python predict_from_raw_data.py

# # convert gt json file into png with correspond name
# cd ../../
# python output_rename.py

