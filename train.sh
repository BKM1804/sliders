CUDA_VISIBLE_DEVICES="0" \
python trainscripts/textsliders/train_lora.py \
    --name 'veronika_dreambooth_prompt_butt' \
    --rank 4 --alpha 1 \
    --config_file 'trainscripts/textsliders/data/custom/config.yaml' \
    --prompts_file 'trainscripts/textsliders/data/custom/veronika_butt.yaml'
    # --attributes "full body, upper body, focus on face; front view, side view, back view" \

# CUDA_VISIBLE_DEVICES="0" \
# python trainscripts/textsliders/train_lora.py \
#     --name 'veronika-butt_gs_1' \
#     --rank 4 --alpha 1 \
#     --config_file 'trainscripts/textsliders/data/custom/config.yaml' \
#     --prompts_file 'trainscripts/textsliders/data/custom/butt.yaml' \
#     # --attributes "full body, lower body; front view, side view, back view" \