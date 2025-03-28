from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch, os


def generator(checkpoint_path, output_dir, prompt, seed=0):
    # create the image folder
    image_dir = os.path.join(output_dir, 'images') 
    if(os.path.exists(image_dir)): pass
    else: os.mkdir(image_dir)

    # load the SDXL model
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, 
        adapter_type="krona", # Added
        attn_update_unet="kqvo", # Added
    )
    pipe = pipe.to("cuda")
    pipe.load_lora_weights(checkpoint_path, 
        adapter_type="krona", 
        attn_update_unet="kqvo",
    )
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    )
    refiner.to("cuda"); generator = torch.Generator("cuda").manual_seed(seed)
    
    # generate images
    image = pipe(prompt=prompt, output_type="latent", generator=generator).images[0]
    image = refiner(prompt=prompt, image=image[None, :], generator=generator).images[0]
        
    image_save_path = os.path.join(image_dir, f"image_{seed}.jpg")
    image.save(image_save_path)
    print(f"Image generation completed.")


prompt = "A sksdog6 op top of sofa" # prompt for the image generation
checkpoint_path = "../outputs/dog6/krona_k64:8q64:8v64:8o64:8_sdxl_0.001/"
output_path = checkpoint_path # where you want to save the generated images, it will generate a folder named images
generator(checkpoint_path, output_path, prompt, seed=5) # generate the image