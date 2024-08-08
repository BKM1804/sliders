CUDA_VISIBLE_DEVICES="0" \
python trainscripts/textsliders/train_lora.py \
    --name 'age' \
    --rank 4 --alpha 1 \
    --config_file 'trainscripts/textsliders/data/custom/config.yaml' \
    --prompts_file 'trainscripts/textsliders/data/custom/age_posneg.yaml'