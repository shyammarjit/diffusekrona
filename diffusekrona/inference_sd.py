
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline
import torch, os, clip
from tqdm import trange
import torchvision.transforms as T
import torch.nn.functional as F
from train_dreambooth_lora_sdxl import parse_args
import pandas as pd


def generator(args, prompts, from_checkpoint):
    if(args.diffusion_model == "sdxl"):
        # load the SDXL model
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, 
            adapter_type=args.adapter_type, # Added
            attn_update_unet=args.attn_update_unet, # Added
            attn_update_text=args.attn_update_text, # Added
            # text_tune_mlp=args.text_tune_mlp, # No need for this one # Added
            train_text_encoder=args.train_text_encoder, # Added
        )
        pipe = pipe.to("cuda")
        pipe.load_lora_weights(os.path.join(args.output_dir, from_checkpoint), 
            adapter_type=args.adapter_type, 
            attn_update_unet=args.attn_update_unet,
            attn_update_text=args.attn_update_text,
        )
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
            # adapter_type = args.adapter_type,
            # adapter_low_rank = args.adapter_low_rank,
            # tune_mlp=args.tune_mlp
        )
        refiner.to("cuda"); generator = torch.Generator("cuda").manual_seed(args.seed)
    
    elif(args.diffusion_model == "base"):
        model_id = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        pipe.load_lora_weights(args.output_dir, 
            adapter_type=args.adapter_type, 
            attn_update_unet=args.attn_update_unet,
            attn_update_text=args.attn_update_text,
            train_text_encoder=args.train_text_encoder,
            weight_name="pytorch_lora_weights.safetensors",
        )
    else:
        raise AttributeError("only supported base and sdxl model")

    # create the image folder
    image_dir = os.path.join(args.output_dir, f'images-{from_checkpoint}') 
    if(os.path.exists(image_dir)): pass
    else: os.mkdir(image_dir)
    
    for i in trange(1, desc = "generating images"): # run for one prompt only to test all OK
        if(args.diffusion_model == "sdxl"):
            image = pipe(prompt=prompts[i], output_type="latent", generator=generator).images[0]
            image = refiner(prompt=prompts[i], image=image[None, :], generator=generator).images[0]
        else: # without sdxl run
            image = pipe(prompts[i], num_inference_steps=50, guidance_scale=7).images[0]
        
        image_save_path = os.path.join(image_dir, f"image_{i+1}_{args.seed}.jpg")
        image.save(image_save_path)
    print(f"Image generation completed.")
    if args.diffusion_model == "sdxl":
        del pipe, refiner
    else:
        del pipe



if __name__ == "__main__":
    # load the clip model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    # get the arguments
    args = parse_args()
    # generate the prompts
    prompts = get_promts(os.path.basename(args.instance_data_dir))
    
    # find the availabel checkpoint list name
    from_checkpoint = "/home/pnoel/tune_diffusion/results/cat/krona_k64:8q64:8v64:8o64:8_sdxl_0.001/checkpoint-1000"
    # generate images based on given prompts
    generator(args, prompts, from_checkpoint)
    # compute the quantiative results (CLIP-I, CLIP-T)
    # clipi, clipt = evaluator(args, prompts, from_checkpoint)
    # save_metrics(args, clipi, clipt, from_checkpoint)

