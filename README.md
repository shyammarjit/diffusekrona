<div align="center">

## 🚀 DiffuseKronA <br> [webpage](https://diffusekrona.github.io/) | [paper](https://openaccess.thecvf.com/content/WACV2025/papers/Marjit_DiffuseKronA_A_Parameter_Efficient_Fine-Tuning_Method_for_Personalized_Diffusion_Models_WACV_2025_paper.pdf) | [video](https://www.youtube.com/watch?v=BLpPFKcPKNY) | [dataset](https://github.com/diffusekrona/data)<br> <p align="left">💡 Highlight</p>
</div>
✔️ Parameter Efficient: A minimum 35% reduction in parameters. By changing Kronecker factors, we can even achieve up to a 75% reduction with results comparable to LoRA-DreamBooth.<br/>
✔️ Enhanced Stability: Our method is more stable compared to LoRA-DreamBooth. Stability refers to variations in images generated across different learning rates and Kronecker factor/ranks, which makes LoRA-DreamBooth harder to fine-tune.<br/>
✔️ Text Alignment and Fidelity: On average, DiffusekronA captures better subject semantics and large contextual prompts.<br/>
✔️ Interpretability: Leverages the advantages of the Kronecker product to capture structured relationships in attention-weight matrices. More controllable decomposition makes DiffusekronA more interpretable.<br/>

## ⭐ Method Details
Overview of DiffuseKronA:</br>
✨ Fine-tuning process involves optimizing the multi-head attention parameters (Q, K, V , and O) using Kronecker Adapter, elaborated in the subsequent blocks. </br>
✨ During inference, newly trained parameters, denoted as θ, are integrated with the original weights Dϕ and images are synthesized using the updated personalized model D<sub>ϕ+θ</sub>.</br>
✨ We also present a schematic illustration of LoRA vs DiffuseKronA; LoRA is limited to one controllable parameter, the rank r; while the Kronecker product showcases enhanced interpretability by introducing two controllable parameters a<sub>1</sub> and a<sub>2</sub> (or equivalently b<sub>1</sub> and b<sub>2</sub>). Furthermore, we also showcase
the advantages of the proposed method.
<br>
<div class="gif">
<p align="center">
<img src='assets/diffusekrona.gif' align="center" width=800>
</p>
</div>


## 🛠️ Installation Steps

1. Create conda environment
```python
conda create -y -n diffusekrona python=3.11
conda activate diffusekrona
```

2. Package Installations
```python
pip install -e ".[torch]"           # To install HuggingFace Diffusers library
pip install -r requirements.txt     # To install extra requirements
pip install accelerator
```

3. Install CLIP
```python
pip install git+https://github.com/openai/CLIP.git
```

## 🔥 Quickstart
1. Clone the Datasets and remove the *subject/generated subfolders
```python
git clone https://github.com/diffusekrona/data && rm -rf data/.git
mkdir results
cd diffusekrona/
python format_datasets.py
```

2. Finetune diffusekrona (leberaing sdxl model) using script file
```python
cd diffusekrona/                # RUN inside diffusekrona folder
bash diffusekrona_sdxl.sh       # RUN when you have single GPU
```
If you have multiple GPUs then please run `CUDA_VISIBLE_DEVICES=$GPU_ID bash diffusekrona_sdxl.sh`. 

Generate images from the finetuned weights 
```python
python generator.py
```

<!-- ## What to run?
To run without text encoder config please hit this inside ```dreambooth``` folder
```
bash dreambooth/test.sh
```


To run with text encoder config please hit this inside ```dreambooth``` folder
```
bash test_text.sh
``` -->

## 🙏🏼 Acknowledgement
Our codebase is built on top of the Hugging Face [Diffusers](https://github.com/huggingface/diffusers) library, and we’re incredibly grateful for their amazing work!

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
## ✉️ Contact

Shyam Marjit: marjitshyam@gmail.com or shyam.marjit@iiitg.ac.in