# CUDA_VISIBLE_DEVICES="0" \
# python trainscripts/textsliders/train_lora.py \
#     --name 'butt_test_148' \
#     --rank 4 --alpha 1 \
#     --config_file 'trainscripts/textsliders/data/custom/config.yaml' \
#     --prompts_file '/workspace/sliders/trainscripts/textsliders/data/custom/veronika_butt.yaml'
#     # --attributes "full body, upper body, focus on face; front view, side view, back view" \

# CUDA_VISIBLE_DEVICES="0" \
# python trainscripts/textsliders/train_lora.py \
#     --name 'weight_test_148' \
#     --rank 4 --alpha 1 \
#     --config_file 'trainscripts/textsliders/data/custom/config.yaml' \
#     --prompts_file '/workspace/sliders/trainscripts/textsliders/data/custom/veronika_chubby.yaml'

CUDA_VISIBLE_DEVICES="0" \
python trainscripts/textsliders/train_lora.py \
    --name 'breast_test_148_v2' \
    --rank 4 --alpha 1 \
    --config_file 'trainscripts/textsliders/data/custom/config.yaml' \
    --prompts_file '/workspace/sliders/trainscripts/textsliders/data/custom/veronika_breast.yaml'