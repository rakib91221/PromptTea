<!-- ## **PromptTea4CogVideoX1.5** -->
# PromptTea4CogVideoX1.5


https://github.com/user-attachments/assets/be86db7b-5993-46a2-a196-2748a5be87dc

## 📈 Inference Latency Comparisons on a Single H100 GPU

| Method | FLPOs(P) ↓ | Speedup ↑ | Latency (ms) ↓ | VBench2 ↑ | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
| **CogVideoX1.5 (81feames, 1360×768, DPMScheduler)** | 150.12| 1x | 485.84 | 0.4633 | -- | -- | -- |
| TeaCache (slow) [Tea](https://github.com/ali-vilab/TeaCache) | 111.67 | 1.32x | 367.56 | 0.4669 | 0.0214 | 0.9540 | 36.26 |
| TeaCache (fast) [Tea](https://github.com/ali-vilab/TeaCache) | 99.83 | 1.47× | 329.58 | 0.4604 | 0.4712 | 0.5169 | 15.04 |
| Ours (PromptTea) | 84.63 | 1.75× | 278.14 | 0.4963 | 0.0836 | 0.8878 | 27.78 |

## Installation

```shell
pip install --upgrade diffusers[torch] transformers protobuf tokenizers sentencepiece imageio imageio-ffmpeg joblib pandas
```

## Usage

You can change the `ckpts_path`, `prompt`to customize your identity-preserving video.

For T2V inference, you can use the following command:

```bash
cd PromptTea4CogVideoX1.5

python3 prompttea_smaple.py \
    --rel_l1_thresh 0.2 \
    --ckpts_path THUDM/CogVideoX1.5-5B \
    --prompt "A flower changes from purple to orange." \
    --seed 42 \
    --num_inference_steps 50 \
    --output_path ./results \
    --use_pca \
```


## Acknowledgements

We would like to thank the contributors to the [CogVideoX](https://github.com/THUDM/CogVideo) and [Diffusers](https://github.com/huggingface/diffusers).