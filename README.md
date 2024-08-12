# Concept Sliders
###  [Project Website](https://sliders.baulab.info) | [Arxiv Preprint](https://arxiv.org/pdf/2311.12092.pdf) | [Trained Sliders](https://sliders.baulab.info/weights/xl_sliders/) | [Colab Demo](https://colab.research.google.com/github/rohitgandikota/sliders/blob/main/demo_concept_sliders.ipynb) | [Huggingface Demo](https://huggingface.co/spaces/baulab/ConceptSliders) <br>
Official code implementation of "Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models"

<div align='center'>
<img src = 'images/main_figure.png'>
</div>

## Colab Demo
Try out our colab demo here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohitgandikota/sliders/blob/main/demo_concept_sliders.ipynb)

## UPDATE
You can now use GPT-4 (or any other openAI model) to create prompts for your text sliders. All you need to do is describe what slider you want to create (e.g: "i want to make people look happy"). <br>
Please refer to the [GPT-notebook](https://github.com/rohitgandikota/sliders/blob/main/GPT_prompt_helper.ipynb)

## Setup
To set up your python environment:
```
conda create -n sliders python=3.9
conda activate sliders

git  clone https://github.com/rohitgandikota/sliders.git
cd sliders
pip install -r requirements.txt
```
If you are running on Windows - please refer to these Windows setup guidelines [here](https://github.com/rohitgandikota/sliders/issues/27#issuecomment-1833572579)
## Textual Concept Sliders
### Training SD-1.x and SD-2.x LoRa
To train an age slider - go to `train-scripts/textsliders/data/prompts.yaml` and edit the `target=person` and `positive=old person` and `unconditional=young person` (opposite of positive) and `neutral=person` and `action=enhance` with `guidance=4`. <br>
If you do not want your edit to be targetted to person replace it with any target you want (eg. dog) or if you need it global replace `person` with `""`  <br>
Finally, run the command:
```
python trainscripts/textsliders/train_lora.py 
  --rank 4 --alpha 1 
  --attributes 'male, female' 
  --name 'ageslider' 
  --config_file 'trainscripts/textsliders/data/config.yaml'
```
`--attributes` argument is used to disentangle concepts from the slider. For instance age slider makes all old people male (so instead add the `"female, male"` attributes to allow disentanglement)
For more information, refer to `train.sh`.
### Infer
Run this script to infer:
```
python script.py
  --lora_weight: Path to the LoRA weight file (required).
  --prompt_file: Path to the prompt file (default: prompts.txt).
  --out_dir: Output directory for the generated images and GIFs (default: output).
  --model_path: Path to the pretrained model (default: base_models/veronika).
  --revision: Model revision (optional).
  --device: Device to run the model on (default: cuda:0).
  --weight_dtype: Weight data type (default: float16).
```
Output structure
```
output_directory/
    weight_name/
        prompt_1_0.png
        prompt_1_1.png
        ...
        prompt_1.gif
        prompt_2_0.png
        ...
        prompt_2.gif
        ...
```
For more information, use the notebook `SD1-sliders-inference.ipynb`
## Citing our work
The preprint can be cited as follows
```
@article{gandikota2023sliders,
  title={Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Tingrui Zhou and Antonio Torralba and David Bau},
  journal={arXiv preprint arXiv:2311.12092},
  year={2023}
}
```
