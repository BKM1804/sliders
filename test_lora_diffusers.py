from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
def concatenate_images(images_list):
    widths, heights = zip(*(img.size for img in images_list))
    total_width = sum(widths)
    max_height = max(heights)

    concatenated_image = Image.new('RGB', (total_width, max_height), "white")
    
    x_offset = 0
    for img in images_list:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return concatenated_image
pipe = StableDiffusionPipeline.from_single_file('/workspace/veronika/veronika.safetensors' , dtype = torch.float16).to('cuda')
scales = [-0.5 , -0.4 , -0.3 , -0.1 , 0, 0.1 , 0.2 , 0.3 , 0.4 ,0.5]
seed = 46
generator = torch.Generator(device="cuda").manual_seed(seed)
img_list = []
for idx , scale in enumerate(scales):
    pipe.load_lora_weights('/workspace/sliders/models/veronika_dreambooth_prompt_weight_alpha1.0_rank8_full/veronika_dreambooth_prompt_weight_alpha1.0_rank8_full_200steps.safetensors' , adapter_name = 'a')
    
    pipe.fuse_lora(lora_scale = scale)
    prompt = "a professional photo of ohwx woman, wearing a croptop and short skirt, full body"
    negative_prompt = 'nude, old, wrinkles, mole, blemish, scar, cg, 3d'
    img = pipe(prompt = prompt ,
              negavtive_prompt = negative_prompt , 
              guidance_scale = 7.5,
               generator = generator
              ).images[0]
    img_list.append(img)
    pipe.delete_adapters(adapter_names = 'a')
rs = concatenate_images(img_list)
rs.save('out.png')
    