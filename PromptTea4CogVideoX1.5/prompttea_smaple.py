# This script is modified based on https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4CogVideoX1.5/teacache_sample_video.py
import argparse
import torch
import numpy as np
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers, export_to_video, load_image
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
import joblib
import pandas as pd
from diffusers.utils import replace_example_docstring
from diffusers.pipelines.cogvideo.pipeline_cogvideox import EXAMPLE_DOC_STRING, retrieve_timesteps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
import time
import math
from diffusers.schedulers import CogVideoXDPMScheduler, CogVideoXDDIMScheduler

def calculate_PCA_threshold(reference_embeddings, context, t1=0.2, t2=0.3):
    """
    Calculate the caching threshold after PCA.

    Args:
        reference_embeddings (dict): Reference prompt embeddings.
        context (list): List of text embeddings.
        t1 (float): Lower bound of the threshold.
        t2 (float): Upper bound of the threshold.

    Returns:
        threshold of the threshold.

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
    k = 50  
    ratio = complex_sim / (simple_sim + complex_sim + 1e-6)
    w = (1 / (1 + torch.exp(-k * (ratio - 0.5)))).item()
    threshold = w*t1 + (1-w)*t2
    return threshold

@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def pca_call(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 6,
    use_dynamic_cfg: bool = False,
    num_videos_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 226,
) -> Union[CogVideoXPipelineOutput, Tuple]:
    """
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
            The height in pixels of the generated image. This is set to 480 by default for the best results.
        width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
            The width in pixels of the generated image. This is set to 720 by default for the best results.
        num_frames (`int`, defaults to `48`):
            Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
            contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
            num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
            needs to be satisfied is that of divisibility mentioned above.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        guidance_scale (`float`, *optional*, defaults to 7.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of videos to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
            of a plain tuple.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        max_sequence_length (`int`, defaults to `226`):
            Maximum sequence length in encoded prompt. Must be consistent with
            `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

    Examples:

    Returns:
        [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
        [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
        `tuple`. When returning a tuple, the first element is a list with the generated images.
    """

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
    width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
    num_frames = num_frames or self.transformer.config.sample_frames

    num_videos_per_prompt = 1

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds,
        negative_prompt_embeds,
    )
    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._interrupt = False

    # 2. Default call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        negative_prompt,
        do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    # pca->teacache_threshold
    threshold = None
    if self.reference_embeddings != None:
        threshold = calculate_PCA_threshold(self.reference_embeddings, prompt_embeds)
        print(f'caching threshold = {threshold}')

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
    self._num_timesteps = len(timesteps)

    # 5. Prepare latents
    latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = self.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * self.vae_scale_factor_temporal

    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        latent_channels,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if self.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        # for DPM-solver++
        old_pred_original_sample = None
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            t1 = time.time()
            # predict noise model_output
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
                threshold=threshold,
            )[0]
            noise_pred = noise_pred.float()
            # perform guidance
            if use_dynamic_cfg:
                self._guidance_scale = 1 + guidance_scale * (
                    (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            else:
                latents, old_pred_original_sample = self.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(prompt_embeds.dtype)

            # call the callback, if provided
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    if not output_type == "latent":
        # Discard any padding frames that were added for CogVideoX 1.5
        latents = latents[:, additional_frames:]
        video = self.decode_latents(latents)
        video = self.video_processor.postprocess_video(video=video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return CogVideoXPipelineOutput(frames=video)

# multivariate polynomial feature expansion
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



def prompttea_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        threshold: float = None,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        teacache_threshold = threshold if threshold!=None else self.rel_l1_thresh
        if self.enable_teacache:
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
                should_calc_cfg = True
                self.accumulated_cfg_l1_distance = 0
            else: 
                delta_vec = ((emb-self.previous_emb).abs().mean() / self.previous_emb.abs().mean()).cpu().item()
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
            self.previous_emb = emb
                      
        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
                encoder_hidden_states += self.previous_residual_encoder
            else:
                ori_hidden_states = hidden_states.clone()
                ori_encoder_hidden_states = encoder_hidden_states.clone()
                if should_calc_cfg == False:
                    hidden_states = hidden_states[1:,]
                    encoder_hidden_states = encoder_hidden_states[1:,]
                    ori_emb = emb.clone()
                    emb = emb[1:,]
                    # 4. Transformer blocks
                for i, block in enumerate(self.transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            emb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )
                    else:
                        hidden_states, encoder_hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=emb,
                            image_rotary_emb=image_rotary_emb,
                        )
                if should_calc_cfg == False:   
                    hidden_states = torch.cat([hidden_states]*2, dim=0)
                    encoder_hidden_states = torch.cat([encoder_hidden_states]*2, dim=0)
                    emb = ori_emb

                self.previous_residual = hidden_states - ori_hidden_states
                self.previous_residual_encoder = encoder_hidden_states - ori_encoder_hidden_states
        else:
            # 4. Transformer blocks
            for i, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B and CogvideoX1.5-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 5. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0  

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def main(args):
    seed = args.seed
    ckpts_path = args.ckpts_path
    output_path = args.output_path
    num_inference_steps = args.num_inference_steps
    rel_l1_thresh = args.rel_l1_thresh
    generate_type = args.generate_type
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    height = args.height
    width = args.width
    num_frames = args.num_frames
    guidance_scale = args.guidance_scale
    fps = args.fps
    image_path = args.image_path
    # mode = ckpts_path.split("/")[-1]

    if generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(ckpts_path, torch_dtype=torch.bfloat16)
    else:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(ckpts_path, torch_dtype=torch.bfloat16)
        image = load_image(image=image_path)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    
    # pac
    use_pca = args.use_pca
    pipe.transformer.__class__.reference_embeddings = None
    pipe.__class__.__call__ = pca_call
    pipe.__class__.reference_embeddings = None
    if use_pca:
        reference_embeddings = torch.load(args.ref_emb_path)
        for emb in reference_embeddings:
            reference_embeddings[emb] = reference_embeddings[emb].to("cuda")
        pipe.__class__.reference_embeddings = reference_embeddings
    
    # cfgcache
    use_cfgcache = True if guidance_scale>1 else False
    # use_cfgcache = True if (args.cfg_scale>1) else False
    pipe.transformer.__class__.use_cfgcache = use_cfgcache
    codebook = prepare_codebook(args.codebook_path)
    codebook_dict = {(i): value for i, value in zip(codebook.index.get_level_values(0),
                                                    codebook['delta_cfg'])}
    pipe.transformer.__class__.cfgcache_thresh = args.cfgcache_thresh
    pipe.transformer.__class__.accumulated_cfg_l1_distance = 0
    pipe.transformer.__class__.cfg_codebook = codebook_dict

    # TeaCache
    pipe.transformer.__class__.enable_teacache = True
    pipe.transformer.__class__.rel_l1_thresh = rel_l1_thresh
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_emb = None
    pipe.transformer.__class__.previous_residual = None
    pipe.transformer.__class__.previous_residual_encoder = None
    pipe.transformer.__class__.num_steps = num_inference_steps
    pipe.transformer.__class__.cnt = 0
    # pipe.transformer.__class__.coefficients = coefficients_dict[mode]
    pipe.transformer.__class__.fit_model = joblib.load(args.fit_model_path)
    pipe.transformer.__class__.forward = prompttea_forward

    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if generate_type == "t2v":
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cuda").manual_seed(seed)
        ).frames[0]
    else:
        video = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    words = prompt.split()[:5]
    video_path = f"{output_path}/prompttea_cogvideox1.5-5B_{words}_{rel_l1_thresh}.mp4"
    export_to_video(video, video_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CogVideoX1.5-5B with given parameters")
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument("--output_path", type=str, default="./results", help="The path where the generated video will be saved")
    parser.add_argument('--ckpts_path', type=str, default="./THUDM/CogVideoX1.5-5B", help='Path to checkpoint, t2v->THUDM/CogVideoX1.5-5B, i2v->THUDM/CogVideoX1.5-5B-I2V')
    
    parser.add_argument('--prompt', type=str, default="A clear, turquoise river flows through a rocky canyon, cascading over a small waterfall and forming a pool of water at the bottom.The river is the main focus of the scene, with its clear water reflecting the surrounding trees and rocks. The canyon walls are steep and rocky, with some vegetation growing on them. The trees are mostly pine trees, with their green needles contrasting with the brown and gray rocks. The overall tone of the scene is one of peace and tranquility.", help='Description of the video for the model to generate')
    parser.add_argument('--negative_prompt', type=str, default=None, help='Description of unwanted situations in model generated videos')
    parser.add_argument("--image_path",type=str,default=None,help="The path of the image to be used as the background of the video")
    parser.add_argument("--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v')")
    parser.add_argument("--width", type=int, default=1360, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=768, help="Number of steps for the inference process")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of steps for the inference process")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--fps", type=int, default=16, help="Number of steps for the inference process")
    parser.add_argument("--fit_model_path", type=str, default="fit_model_cogvideox1.5.pkl", help="Fit model path")
    parser.add_argument("--codebook_path", type=str, default="codebook_cogvideox1.5.csv", help="Codebook path")
    parser.add_argument("--ref_emb_path", type=str, default="reference_embeddings_cogvideox1.5.pt", help="Path of reference embedding")
    parser.add_argument('--rel_l1_thresh', type=float, default=0.2, help='Higher speedup will cause to worse quality')
    parser.add_argument('--cfgcache_thresh', type=float, default=0.02, help='Higher speedup will cause to worse quality')
    parser.add_argument('--use_pca', action="store_true", default=False, help='Using Prompt Complexity Aware will automatically select the threshold for TeaCache based on prompt words.')


    args = parser.parse_args()

    main(args)