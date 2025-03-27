from diffusers import StableDiffusionPipeline
import torch, os


def generator(checkpoint_path, output_dir, prompt, seed=0):
    # create the image folder
    image_dir = os.path.join(output_dir, 'images') 
    if(os.path.exists(image_dir)): pass
    else: os.mkdir(image_dir)

    # load the SD model
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.load_lora_weights(checkpoint_path, 
        adapter_type="krona", 
        attn_update_unet='kqvo',
        weight_name="pytorch_lora_weights.safetensors",
    )

    # generate images
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
    image_save_path = os.path.join(image_dir, f"image_{seed}.jpg")
    image.save(image_save_path)
    print(f"Image generation completed.")


prompt = "A sksteapot op top of sofa" # prompt for the image generation
checkpoint_path = "../outputs/teapot/krona_k64:8q64:8v64:8o64:8_base_0.001"
output_path = checkpoint_path # where you want to save the generated images, it will generate a folder named images
generator(checkpoint_path, output_path, prompt, seed=5) # generate the image