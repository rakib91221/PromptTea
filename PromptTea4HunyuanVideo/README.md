<!-- ## **PromptTea4HunyuanVideo** -->
# PromptTea4HunyuanVideo

https://github.com/user-attachments/assets/444d7207-d40b-4064-a5ae-69b1cf3cf167

## 📈 Inference Latency Comparisons on a Single H100

| Method | FLPOs(P) ↓ | Speedup ↑ | Latency (ms) ↓ | VBench2 ↑ | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **HunyuanVideo (129frames, 1280×720)** | 85.16 | 1x | 1825.67 | 0.4875 | -- | -- | -- |
| TeaCache (slow) [Tea](https://github.com/ali-vilab/TeaCache) | 52.80 | 1.61× | 1130.90 | 0.4125 | 0.1477 | 0.8083 | 24.02 |
| TeaCache (fast) [Tea](https://github.com/ali-vilab/TeaCache) | 39.85 | 2.17× | 842.22 | 0.4318 | 0.1554 | 0.8011 | 23.68 |
| Ours (PromptTea) | 33.12 | 2.56× | 689.37 | 0.4444 | 0.1468 | 0.8138 | 24.42 |

## Usage

Follow [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) to clone the repo and finish the installation. Note: After setting up the environment, make sure to install the additional dependencies `joblib` and `pandas` using `pip install joblib pandas`. Then copy 'prompttea_sample.py', ''fit_model_hunyuanvideo.pkl', 'reference_embeddings_hunyuanvideo.pt' in this repo to the HunyuanVideo repo.

For single-gpu inference, you can use the following command:

```bash
cd HunyuanVideo

python3 prompttea_sample.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A lion with the ears of a bat, the body of a whale, the claws of an eagle, and the wings of a dragon, an unstoppable predator both in the sea and in the sky." \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./teacache_results \
    --use_pca \
```

## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and [TeaCache](https://github.com/ali-vilab/TeaCache/tree/main).