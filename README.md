# DiffuseKronA

### [webpage](https://diffusekrona.github.io/) | [paper](https://openaccess.thecvf.com/content/WACV2025/papers/Marjit_DiffuseKronA_A_Parameter_Efficient_Fine-Tuning_Method_for_Personalized_Diffusion_Models_WACV_2025_paper.pdf) | [video](https://www.youtube.com/watch?v=BLpPFKcPKNY) | [dataset](https://github.com/diffusekrona/Data)


## 💡 Highlight
✔️ Parameter Efficient: A minimum 35% reduction in parameters. By changing Kronecker factors, we can even achieve up to a 75% reduction with results comparable to LoRA-DreamBooth.

✔️ Enhanced Stability: Our method is more stable compared to LoRA-DreamBooth. Stability refers to variations in images generated across different learning rates and Kronecker factor/ranks, which makes LoRA-DreamBooth harder to fine-tune.

✔️ Text Alignment and Fidelity: On average, DiffusekronA captures better subject semantics and large contextual prompts.

✔️ Interpretability: Leverages the advantages of the Kronecker product to capture structured relationships in attention-weight matrices. More controllable decomposition makes DiffusekronA more interpretable.
<hr />

<video width="1190" height="438" controls><source src="/assets/diffusekrona.mp4" type="video/mp4">

## Installation Steps

Create conda environment
```
conda create -y -n diffusekrona python=3.11
conda activate diffusekrona
```

Package Installations
```
pip install -e ".[torch]" # To install HuggingFace Diffusers library
pip install -r requirements.txt # To install extra requirements
pip install accelerator

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
# RUN `conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0`, only when pip fails
```

Train dreambooth_sdxl using script file
```
cd diffusekrona/                                        # Run inside diffusekrona folder
bash run_lora_sdxl.sh                                   # when only one GPU
CUDA_VISIBLE_DEVICES=$GPU_ID bash run_lora_sdxl.sh      # when having only one GPU
```

Generate images from the finetuned weights 
```
python generator.py
```

## What to run?
To run without text encoder config please hit this inside ```dreambooth``` folder
```
bash dreambooth/test.sh
```


To run with text encoder config please hit this inside ```dreambooth``` folder
```
bash test_text.sh
```


## ✏️ Citation
If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```bash
@InProceedings{Marjit_2025_WACV,
    author    = {Marjit, Shyam and Singh, Harshit and Mathur, Nityanand and Paul, Sayak and Yu, Chia-Mu and Chen, Pin-Yu},
    title     = {DiffuseKronA: A Parameter Efficient Fine-Tuning Method for Personalized Diffusion Models},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {3529-3538}
}
```
## ☎️ Contact

Shyam Marjit: marjitshyam@gmail.com or shyam.marjit@iiitg.ac.in