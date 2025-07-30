<!-- ## **PromptTea4Wan2.1** -->
# PromptTea4Wan2.1

https://github.com/user-attachments/assets/717ad415-ba3e-4962-82b0-c4fe4e6a2ba4

## 📈 Inference Latency Comparisons on a Single H100

| Method | FLPOs(P) ↓ | Speedup ↑ | Latency (ms) ↓ | VBench2 ↑ | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Wan2.1 (129frames, 1280×720)** | 181.62 | 1x | 1880.34 | 0.5235 | -- | -- | -- |
| TeaCache (slow) [Tea](https://github.com/ali-vilab/TeaCache) | 91.14 | 1.93× | 972.92 | 0.4565 | 0.2993 | 0.6290 | 17.50 |
| TeaCache (fast) [Tea](https://github.com/ali-vilab/TeaCache) | 66.81 | 2.64× | 713.35 | 0.4480 | 0.3371 | 0.6057 | 16.81 |
| Ours (PromptTea) | 62.50 | 2.79× | 674.46 | 0.4710 | 0.1380 | 0.7884 | 23.00 |


## Usage

Follow [Wan2.1](https://github.com/Wan-Video/Wan2.1) to clone the repo and finish the installation. Note: After setting up the environment, make sure to install the additional dependencies `joblib` and `pandas` using `pip install joblib pandas`. Then copy 'prompttea_sample.py', 'codebook_wan2.1.csv', 'fit_model_wan2.1.pkl', 'reference_embeddings_wan2.1.pt' in this repo to the Wan2.1 repo.

For T2V with 14B model, you can use the following command:

```bash
python prompttea_sample.py  --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B  --prompt "A person is listening to music, then they suddenly start washing the dishes." --base_seed 42 --offload_model True --t5_cpu  --use_pca
```

## Acknowledgements

We would like to thank the contributors to the [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [TeaCache](https://github.com/ali-vilab/TeaCache/tree/main).