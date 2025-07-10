# This script is modified based on https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4HunyuanVideo/teacache_sample_video.py
import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import add_network_args, add_extra_models_args, add_denoise_schedule_args, add_inference_args, add_parallel_args, sanity_check_args
from hyvideo.inference import HunyuanVideoSampler

from hyvideo.modules.modulate_layers import modulate
from hyvideo.modules.attenion import attention, parallel_attention, get_cu_seqlens
from typing import Any, List, Tuple, Optional, Union, Dict
import torch
import json
import numpy as np
import joblib
import pandas as pd
from hyvideo.diffusion.pipelines.pipeline_hunyuan_video import EXAMPLE_DOC_STRING, retrieve_timesteps, rescale_noise_cfg, HunyuanVideoPipelineOutput
from hyvideo.constants import PRECISION_TO_TYPE
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
import torch.distributed as dist
import argparse

from diffusers.utils import (
    deprecate,
    replace_example_docstring,

)

def add_dynateacache_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="DyanTeaCache args")

    # Model path
    group.add_argument(
        "--ref_emb_path",
        type=str,
        default="reference_embeddings_hunyuanvideo.pt",
    )
    group.add_argument(
        "--fit_model_path",
        type=str,
        default="fit_model_hunyuanvideo.pkl",
    )

    # threshold
    group.add_argument(
        "--use_pca",
        action="store_true",
        default=False,
        help="Using Prompt Complexity Aware will automatically select the threshold for TeaCache based on prompt words."
    )


    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.15,
        help="Higher speedup will cause to worse quality -- 0.1 for 2.0x speedup -- 0.15 for 2.8x speedup"
    )

    return parser

def parse_args(namespace=None):
    parser = argparse.ArgumentParser(description="HunyuanVideo inference script")

    parser = add_network_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_inference_args(parser)
    parser = add_parallel_args(parser)
    parser = add_dynateacache_args(parser)

    args = parser.parse_args(namespace=namespace)
    args = sanity_check_args(args)

    return args

# multivariate polynomial feature expansion
def calculate_PCA_threshold(reference_embeddings, context, t1=0.09, t2=0.17):
    """
    Calculate the caching threshold after PCA.

    Args:
        reference_embeddings (dict): Reference prompt embeddings.
        context (list): List of text embeddings.
        t1 (float): Lower bound of the threshold.
        t2 (float): Upper bound of the threshold.

    Returns:
        threshold (float): PCA threshold.
    """
    context_mean = torch.mean(context,dim=1).unsqueeze(0)
    context_mean = [t.to(reference_embeddings["complex"].device) for t in context_mean]
    simple_sim = torch.mean(torch.cosine_similarity(
        context_mean[0], 
        reference_embeddings["simple"], 
        dim=1
    ))
    complex_sim = torch.mean(torch.cosine_similarity(
        context_mean[0],
        reference_embeddings["complex"],
        dim=1
    ))
    k = 200  
    ratio = complex_sim / (simple_sim + complex_sim + 1e-6)
    w = (1 / (1 + torch.exp(-k * (ratio - 0.5)))).item()
    threshold = w*t1 + (1-w)*t2
    return threshold

@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def pca_call(
    self,
    prompt: Union[str, List[str]],
    height: int,
    width: int,
    video_length: int,
    data_type: str = "video",
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_videos_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_attention_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[
        Union[
            Callable[[int, int, Dict], None],
            PipelineCallback,
            MultiPipelineCallbacks,
        ]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    vae_ver: str = "88-4c-sd",
    enable_tiling: bool = False,
    n_tokens: Optional[int] = None,
    embedded_guidance_scale: Optional[float] = None,
    **kwargs,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
        height (`int`):
            The height in pixels of the generated image.
        width (`int`):
            The width in pixels of the generated image.
        video_length (`int`):
            The number of frames in the generated video.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            A higher guidance scale value encourages the model to generate images closely linked to the text
            `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide what to not include in image generation. If not defined, you need to
            pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
            to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
            not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a
            plain tuple.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
            [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
            using zero terminal SNR.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
        callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
            A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
            each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
            DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
            list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.

    Examples:

    Returns:
        [`~HunyuanVideoPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned,
            otherwise a `tuple` is returned where the first element is a list with the generated images and the
            second element is a list of `bool`s indicating whether the corresponding generated image contains
            "not-safe-for-work" (nsfw) content.
    """
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 0. Default height and width to unet
    # height = height or self.transformer.config.sample_size * self.vae_scale_factor
    # width = width or self.transformer.config.sample_size * self.vae_scale_factor
    # to deal with lora scaling and other possible forward hooks

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        video_length,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        callback_on_step_end_tensor_inputs,
        vae_ver=vae_ver,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = torch.device(f"cuda:{dist.get_rank()}") if dist.is_initialized() else self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None)
        if self.cross_attention_kwargs is not None
        else None
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        prompt_mask,
        negative_prompt_mask,
    ) = self.encode_prompt(
        prompt,
        device,
        num_videos_per_prompt,
        self.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        attention_mask=attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_attention_mask=negative_attention_mask,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
        data_type=data_type,
    )

    # pca->teacache_threshold
    threshold = None
    if self.reference_embeddings != None:
        threshold = calculate_PCA_threshold(self.reference_embeddings, prompt_embeds)
        print(f'caching threshold = {threshold}')
    

    if self.text_encoder_2 is not None:
        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_mask_2,
            negative_prompt_mask_2,
        ) = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            attention_mask=None,
            negative_prompt_embeds=None,
            negative_attention_mask=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
            text_encoder=self.text_encoder_2,
            data_type=data_type,
        )
    else:
        prompt_embeds_2 = None
        negative_prompt_embeds_2 = None
        prompt_mask_2 = None
        negative_prompt_mask_2 = None

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        if prompt_mask is not None:
            prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
        if prompt_embeds_2 is not None:
            prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
        if prompt_mask_2 is not None:
            prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])


    # 4. Prepare timesteps
    extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
        self.scheduler.set_timesteps, {"n_tokens": n_tokens}
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        **extra_set_timesteps_kwargs,
    )

    if "884" in vae_ver:
        video_length = (video_length - 1) // 4 + 1
    elif "888" in vae_ver:
        video_length = (video_length - 1) // 8 + 1
    else:
        video_length = video_length

    # 5. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        video_length,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_func_kwargs(
        self.scheduler.step,
        {"generator": generator, "eta": eta},
    )

    target_dtype = PRECISION_TO_TYPE[self.args.precision]
    autocast_enabled = (
        target_dtype != torch.float32
    ) and not self.args.disable_autocast
    vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
    vae_autocast_enabled = (
        vae_dtype != torch.float32
    ) and not self.args.disable_autocast

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)

    # if is_progress_bar:
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )

            t_expand = t.repeat(latent_model_input.shape[0])
            guidance_expand = (
                torch.tensor(
                    [embedded_guidance_scale] * latent_model_input.shape[0],
                    dtype=torch.float32,
                    device=device,
                ).to(target_dtype)
                * 1000.0
                if embedded_guidance_scale is not None
                else None
            )

            # predict the noise residual
            with torch.autocast(
                device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
            ):
                t1 = time.time()
                noise_pred = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                    latent_model_input,  # [2, 16, 33, 24, 42]
                    t_expand,  # [2]
                    text_states=prompt_embeds,  # [2, 256, 4096]
                    text_mask=prompt_mask,  # [2, 256]
                    text_states_2=prompt_embeds_2,  # [2, 768]
                    freqs_cos=freqs_cis[0],  # [seqlen, head_dim]
                    freqs_sin=freqs_cis[1],  # [seqlen, head_dim]
                    guidance=guidance_expand,
                    return_dict=True,
                    threshold=threshold
                )[
                    "x"
                ]
                # print(f"花费时间{time.time()-t1}---------------------------------------")

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred,
                    noise_pred_text,
                    guidance_rescale=self.guidance_rescale,
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop(
                    "negative_prompt_embeds", negative_prompt_embeds
                )

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                if progress_bar is not None:
                    progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    if not output_type == "latent":
        expand_temporal_dim = False
        if len(latents.shape) == 4:
            if isinstance(self.vae, AutoencoderKLCausal3D):
                latents = latents.unsqueeze(2)
                expand_temporal_dim = True
        elif len(latents.shape) == 5:
            pass
        else:
            raise ValueError(
                f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
            )

        if (
            hasattr(self.vae.config, "shift_factor")
            and self.vae.config.shift_factor
        ):
            latents = (
                latents / self.vae.config.scaling_factor
                + self.vae.config.shift_factor
            )
        else:
            latents = latents / self.vae.config.scaling_factor

        with torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
        ):
            if enable_tiling:
                self.vae.enable_tiling()
                image = self.vae.decode(
                    latents, return_dict=False, generator=generator
                )[0]
            else:
                image = self.vae.decode(
                    latents, return_dict=False, generator=generator
                )[0]

        if expand_temporal_dim or image.shape[2] == 1:
            image = image.squeeze(2)

    else:
        image = latents

    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().float()

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return image

    return HunyuanVideoPipelineOutput(videos=image)

def create_poly_features(x, t):
    X = np.column_stack((
        np.ones(len(x)),     
        x,                  
        t,                   
        x**2,               
        t**2,                
        x*t,                
        x**3,                
        t**3,               
        x**2 * t,            
        x * t**2,            
        x**4,               
        t**4                
    ))
    return X

def prepare_codebook(csv_path):
    """
    Reads a CSV file from the specified path and sets 'num_step' as the index to return as a codebook.

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Codebook DataFrame with MultiIndex set
    """
    df = pd.read_csv(csv_path)
    df.set_index(["step"], inplace=True)
    return df


def teacache_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
        threshold:  Optional[float] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)
        teacache_threshold = threshold if threshold!=None else self.rel_l1_thresh
        if self.enable_teacache:
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
                should_calc_cfg = True
                self.accumulated_cfg_l1_distance = 0
            else: 
                delta_vec = ((vec-self.previous_vec).abs().mean() / self.previous_vec.abs().mean()).cpu().item()
                x_and_t = create_poly_features(np.array([delta_vec]), np.array([self.cnt]))
                l1 = self.fit_model.predict(x_and_t)
                self.accumulated_rel_l1_distance += l1
                if self.accumulated_rel_l1_distance < teacache_threshold:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
                # cfgcache
                if self.use_cfgcache:
                    cfg_l1 = self.cfg_codebook.get(self.cnt, 0)
                    self.accumulated_cfg_l1_distance += cfg_l1
                    if self.accumulated_cfg_l1_distance < self.cfgcache_thresh:
                        should_calc_cfg = False
                        
                    else:
                        should_calc_cfg = True
                        self.accumulated_cfg_l1_distance = 0
                else:
                    should_calc_cfg = True
            self.previous_vec = vec.clone()
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0 

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        # cfgcache Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q_cfgcache = get_cu_seqlens(text_mask[0:1,], img_seq_len)
        cu_seqlens_kv_cfgcache = cu_seqlens_q_cfgcache
        max_seqlen_q_cfgcache = img_seq_len + txt_seq_len
        max_seqlen_kv_cfgcache = max_seqlen_q_cfgcache

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        
        if self.enable_teacache:
            if not should_calc:
                img += self.previous_residual
            else:
                ori_img = img.clone()
                if should_calc_cfg:
                    # --------------------- Pass through DiT blocks ------------------------
                    for _, block in enumerate(self.double_blocks):
                        double_block_args = [
                            img,
                            txt,
                            vec,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            max_seqlen_q,
                            max_seqlen_kv,
                            freqs_cis,
                        ]

                        img, txt = block(*double_block_args)

                    # Merge txt and img to pass through single stream blocks.
                    x = torch.cat((img, txt), 1)
                    if len(self.single_blocks) > 0:
                        for _, block in enumerate(self.single_blocks):
                            single_block_args = [
                                x,
                                vec,
                                txt_seq_len,
                                cu_seqlens_q,
                                cu_seqlens_kv,
                                max_seqlen_q,
                                max_seqlen_kv,
                                (freqs_cos, freqs_sin),
                            ]

                            x = block(*single_block_args)
                else: # 跳过uncond
                    img = img[1:,]
                    txt = txt[1:,]
                    vec = vec[1:,]
                    # --------------------- Pass through DiT blocks ------------------------
                    for _, block in enumerate(self.double_blocks):
                        double_block_args = [
                            img,
                            txt,
                            vec,
                            cu_seqlens_q_cfgcache,
                            cu_seqlens_kv_cfgcache,
                            max_seqlen_q_cfgcache,
                            max_seqlen_kv_cfgcache,
                            freqs_cis,
                        ]

                        img, txt = block(*double_block_args)

                    # Merge txt and img to pass through single stream blocks.
                    x = torch.cat((img, txt), 1)
                    if len(self.single_blocks) > 0:
                        for _, block in enumerate(self.single_blocks):
                            single_block_args = [
                                x,
                                vec,
                                txt_seq_len,
                                cu_seqlens_q_cfgcache,
                                cu_seqlens_kv_cfgcache,
                                max_seqlen_q_cfgcache,
                                max_seqlen_kv_cfgcache,
                                (freqs_cos, freqs_sin),
                            ]

                            x = block(*single_block_args)
                    x = torch.cat([x]*2, dim=0)
                img = x[:, :img_seq_len, ...]
                self.previous_residual = img - ori_img
        else:        
            # --------------------- Pass through DiT blocks ------------------------
            for _, block in enumerate(self.double_blocks):
                double_block_args = [
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                ]

                img, txt = block(*double_block_args)

            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                    ]

                    x = block(*single_block_args)

            img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    # pac
    use_pca = args.use_pca
    hunyuan_video_sampler.pipeline.__class__.reference_embeddings = None
    hunyuan_video_sampler.pipeline.__class__.__call__ = pca_call
    
    if use_pca:
        reference_embeddings = torch.load(args.ref_emb_path)
        for emb in reference_embeddings:
            reference_embeddings[emb] = reference_embeddings[emb].to(int(os.getenv("LOCAL_RANK", 0)))
        hunyuan_video_sampler.pipeline.__class__.reference_embeddings = reference_embeddings

    # cfgcache
    use_cfgcache = False
    hunyuan_video_sampler.pipeline.transformer.__class__.use_cfgcache = use_cfgcache

    
    # TeaCache
    hunyuan_video_sampler.pipeline.transformer.__class__.enable_teacache = True
    hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
    hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = args.infer_steps
    hunyuan_video_sampler.pipeline.transformer.__class__.rel_l1_thresh = args.teacache_thresh 
    hunyuan_video_sampler.pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
    hunyuan_video_sampler.pipeline.transformer.__class__.previous_modulated_input = None
    hunyuan_video_sampler.pipeline.transformer.__class__.previous_residual = None
    hunyuan_video_sampler.pipeline.transformer.__class__.forward = teacache_forward
    hunyuan_video_sampler.pipeline.transformer.__class__.fit_model = joblib.load(args.fit_model_path)
    
    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample save to: {save_path}')

if __name__ == "__main__":
    main()