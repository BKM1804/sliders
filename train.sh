CUDA_VISIBLE_DEVICES="0" \
python trainscripts/textsliders/train_lora.py \
    --name 'veronika_dreambooth_prompt_weight_new_env' \
    --rank 8 --alpha 1 \
    --config_file 'trainscripts/textsliders/data/custom/config.yaml' \
    --prompts_file '/workspace/sliders/trainscripts/textsliders/data/custom/veronika_chubby.yaml'
    # --attributes "full body, upper body, focus on face; front view, side view, back view" \

# CUDA_VISIBLE_DEVICES="0" \