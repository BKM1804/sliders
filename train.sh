CUDA_VISIBLE_DEVICES="1" \
python trainscripts/textsliders/train_lora.py \
    --name 'veronika_age_guidance_scale_1' \
    --rank 4 --alpha 1 \
    --config_file 'trainscripts/textsliders/data/custom/config.yaml' \
    --prompts_file 'trainscripts/textsliders/data/custom/veronika_age.yaml' \
    # --attributes "full body, upper body, focus on face; front view, side view, back view" \