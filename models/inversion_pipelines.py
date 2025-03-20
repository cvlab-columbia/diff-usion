import torch
import PIL
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable, Union
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    BlipForConditionalGeneration,
    BlipProcessor,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMInverseScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.deprecation_utils import deprecate
from diffusers.utils.outputs import BaseOutput
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
import torch.nn.functional as F
import numpy as np


@dataclass
class InversionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        latents (`torch.Tensor`)
            inverted latents tensor
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    latents: torch.Tensor
    images: Union[List[PIL.Image.Image], np.ndarray]


class DDIMInversionSDPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        inverse_scheduler: Optional[DDIMInverseScheduler] = None,
        caption_generator: Optional[BlipForConditionalGeneration] = None,
        caption_processor: Optional[BlipProcessor] = None,
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

        self.register_modules(
            feature_extractor=feature_extractor,
            caption_processor=caption_processor,
            caption_generator=caption_generator,
            inverse_scheduler=inverse_scheduler,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    @torch.no_grad()
    def generate_caption(self, images, text: str = "a photography of"):
        """
        Generates caption for a given image.
        if text is None, generates uncond captions
        """

        prev_device = self.caption_generator.device

        device = self._execution_device
        inputs = self.caption_processor(images, text, return_tensors="pt").to(
            device=device, dtype=self.caption_generator.dtype
        )
        self.caption_generator.to(device)
        outputs = self.caption_generator.generate(**inputs, max_new_tokens=128)

        # offload caption generator
        self.caption_generator.to(prev_device)

        caption = self.caption_processor.batch_decode(
            outputs, skip_special_tokens=True
        )[0]
        return caption

    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.vae.encode(image).latent_dist.sample(generator)

            latents = self.vae.config.scaling_factor * latents

        if batch_size != latents.shape[0]:
            if batch_size % latents.shape[0] == 0:
                # expand image_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                deprecate(
                    "len(prompt) != len(image)",
                    "1.0.0",
                    deprecation_message,
                    standard_warn=False,
                )
                additional_latents_per_image = batch_size // latents.shape[0]
                latents = torch.cat([latents] * additional_latents_per_image, dim=0)
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts."
                )
        else:
            latents = torch.cat([latents], dim=0)

        return latents

    def get_epsilon(
        self, model_output: torch.Tensor, sample: torch.Tensor, timestep: int
    ):
        pred_type = self.inverse_scheduler.config.prediction_type
        alpha_prod_t = self.inverse_scheduler.alphas_cumprod[timestep]

        beta_prod_t = 1 - alpha_prod_t

        if pred_type == "epsilon":
            return model_output
        elif pred_type == "sample":
            return (sample - alpha_prod_t ** (0.5) * model_output) / beta_prod_t ** (
                0.5
            )
        elif pred_type == "v_prediction":
            return (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {pred_type} must be one of `epsilon`, `sample`, or `v_prediction`"
            )

    def auto_corr_loss(self, hidden_states, generator=None):
        reg_loss = 0.0
        for i in range(hidden_states.shape[0]):
            for j in range(hidden_states.shape[1]):
                noise = hidden_states[i : i + 1, j : j + 1, :, :]
                while True:
                    roll_amount = torch.randint(
                        noise.shape[2] // 2, (1,), generator=generator
                    ).item()
                    reg_loss += (
                        noise * torch.roll(noise, shifts=roll_amount, dims=2)
                    ).mean() ** 2
                    reg_loss += (
                        noise * torch.roll(noise, shifts=roll_amount, dims=3)
                    ).mean() ** 2

                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss

    def kl_divergence(self, hidden_states):
        mean = hidden_states.mean()
        var = hidden_states.var()
        return var + mean**2 - 1 - torch.log(var + 1e-7)

    @torch.no_grad()
    def invert(
        self,
        prompt: Optional[str] = None,
        image: PipelineImageInput = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        lambda_auto_corr: float = 20.0,
        lambda_kl: float = 20.0,
        num_reg_steps: int = 5,
        num_auto_corr_rolls: int = 5,
    ):
        r"""
        Function used to generate inverted latents given a prompt and image.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.Tensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, or tensor representing an image batch which will be used for conditioning. Can also accept
                image latents as `image`, if passing latents directly, it will not be encoded again.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 1):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            cross_attention_guidance_amount (`float`, defaults to 0.1):
                Amount of guidance needed from the reference cross-attention maps.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            lambda_auto_corr (`float`, *optional*, defaults to 20.0):
                Lambda parameter to control auto correction
            lambda_kl (`float`, *optional*, defaults to 20.0):
                Lambda parameter to control Kullback–Leibler divergence output
            num_reg_steps (`int`, *optional*, defaults to 5):
                Number of regularization loss steps
            num_auto_corr_rolls (`int`, *optional*, defaults to 5):
                Number of auto correction roll steps

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero.Pix2PixInversionPipelineOutput`] or
            `tuple`:
            [`~pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero.Pix2PixInversionPipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is the inverted
            latents tensor and then second is the corresponding decoded image.
        """
        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Preprocess image
        image = self.image_processor.preprocess(image)

        # 4. Prepare latent variables
        latents = self.prepare_image_latents(
            image, batch_size, self.vae.dtype, device, generator
        )

        # 5. Encode input prompt
        num_images_per_prompt = 1
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 5. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        )
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.inverse_scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # regularization of the noise prediction
                with torch.enable_grad():
                    for _ in range(num_reg_steps):
                        if lambda_auto_corr > 0:
                            for _ in range(num_auto_corr_rolls):
                                var = torch.autograd.Variable(
                                    noise_pred.detach().clone(), requires_grad=True
                                )

                                # Derive epsilon from model output before regularizing to IID standard normal
                                var_epsilon = self.get_epsilon(
                                    var, latent_model_input.detach(), t
                                )

                                l_ac = self.auto_corr_loss(
                                    var_epsilon, generator=generator
                                )
                                l_ac.backward()

                                grad = var.grad.detach() / num_auto_corr_rolls
                                noise_pred = noise_pred - lambda_auto_corr * grad

                        if lambda_kl > 0:
                            var = torch.autograd.Variable(
                                noise_pred.detach().clone(), requires_grad=True
                            )

                            # Derive epsilon from model output before regularizing to IID standard normal
                            var_epsilon = self.get_epsilon(
                                var, latent_model_input.detach(), t
                            )

                            l_kld = self.kl_divergence(var_epsilon)
                            l_kld.backward()

                            grad = var.grad.detach()
                            noise_pred = noise_pred - lambda_kl * grad

                        noise_pred = noise_pred.detach()

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(
                    noise_pred, t, latents
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        inverted_latents = latents.detach().clone()

        # 8. Post-processing
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (inverted_latents, image)

        return InversionPipelineOutput(latents=inverted_latents, images=image)


class CounterfactualSDPipeline(StableDiffusionPipeline):
    def encode_text(self, prompts: Union[str, list[str]]):
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def sample_xts_from_x0(
        self,
        x0: torch.Tensor,
        num_inference_steps: int,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Samples from P(x_1:T|x_0)
        """
        bsz, in_channels, height, width = x0.shape
        alpha_bar = self.scheduler.alphas_cumprod
        sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        alphas = self.scheduler.alphas

        timesteps = self.scheduler.timesteps.to(self.device)
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(
            (
                bsz,
                num_inference_steps + 1,
                in_channels,
                height,
                width,
            )
        ).to(x0.device)
        xts[:, 0, ...] = x0
        for t in reversed(timesteps):
            idx = num_inference_steps - t_to_idx[int(t)]
            noise = randn_tensor(
                shape=x0.shape, generator=generator, device=self.device, dtype=x0.dtype
            )
            xts[:, idx, ...] = x0 * (alpha_bar[t] ** 0.5) + noise * sqrt_one_minus_alpha_bar[t]

        return xts

    def get_variance(self, timestep):
        prev_timestep = (
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )
        return variance

    def forward_step(self, model_output, timestep, sample):
        next_timestep = min(
            self.scheduler.config.num_train_timesteps - 2,
            timestep
            + self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps,
        )

        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)

        # 5. TODO: simple noising implementatiom
        next_sample = self.scheduler.add_noise(
            pred_original_sample, model_output, torch.LongTensor([next_timestep])
        )
        return next_sample

    def inversion_forward_process(
        self,
        x0: torch.Tensor,
        guidance_scale_src: float,
        prompt: Union[str, list[str]],
        prog_bar: bool = True,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        etas: Optional[Union[List, int, float]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        do_cfg = guidance_scale_src > 1.0
        bsz, in_channels, height, width = x0.shape

        # Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt,
            do_cfg,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        timesteps = self.scheduler.timesteps.to(self.device)
        variance_noise_shape = (bsz, num_inference_steps, in_channels, height, width)
        if etas is None or (type(etas) in [int, float] and etas == 0):
            eta_is_zero = True
            zs = None
        else:
            eta_is_zero = False
            if type(etas) in [int, float]:
                etas = [etas] * self.scheduler.num_inference_steps
            xts = self.sample_xts_from_x0(
                x0, num_inference_steps=num_inference_steps, generator=generator
            )
            alpha_bar = self.scheduler.alphas_cumprod
            zs = torch.zeros(size=variance_noise_shape, device=self.device)
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xt = x0
        op = tqdm(timesteps) if prog_bar else timesteps

        for t in op:
            idx = num_inference_steps - t_to_idx[int(t)] - 1
            # 1. predict noise residual
            if not eta_is_zero:
                xt = xts[:, idx + 1, ...]

            with torch.no_grad():
                latent_model_input = torch.cat([xt] * 2) if do_cfg else xt
                noise_pred = self.unet(
                    latent_model_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

            # perform guidance
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale_src * (
                    noise_pred_text - noise_pred_uncond
                )

            if eta_is_zero:
                # 2. compute more noisy image and set x_t -> x_t+1
                xt = self.forward_step(noise_pred, t, xt)

            else:
                xtm1 = xts[:, idx, ...]
                # pred of x0
                pred_original_sample = (
                    xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred
                ) / alpha_bar[t] ** 0.5

                # direction to xt
                prev_timestep = (
                    t
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
                )
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[prev_timestep]
                    if prev_timestep >= 0
                    else self.scheduler.final_alpha_cumprod
                )

                variance = self.get_variance(t)
                pred_sample_direction = (
                    1 - alpha_prod_t_prev - etas[idx] * variance
                ) ** (0.5) * noise_pred

                mu_xt = (
                    alpha_prod_t_prev ** (0.5) * pred_original_sample
                    + pred_sample_direction
                )

                z = (xtm1 - mu_xt) / (etas[idx] * variance**0.5)
                zs[:, idx, ...] = z

                # correction to avoid error accumulation
                xtm1 = mu_xt + (etas[idx] * variance**0.5) * z
                xts[:, idx, ...] = xtm1

        if not zs is None:
            zs[:, 0, ...] = torch.zeros_like(zs[:, 0, ...])

        return xt, zs, xts

    def reverse_step(self, model_output, timestep, sample, eta=0, variance_noise=None):
        # 1. get previous step value (=t-1)
        prev_timestep = (
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps
        )
        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # variance = self.scheduler._get_variance(timestep, prev_timestep)
        variance = self.get_variance(timestep)  # , prev_timestep)
        std_dev_t = eta * variance ** (0.5)
        # Take care of asymetric reverse process (asyrp)
        model_output_direction = model_output
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (
            0.5
        ) * model_output_direction
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )
        # 8. Add noice if eta > 0
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn(model_output.shape, device=self.device)
            sigma_z = eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z

        return prev_sample

    def inversion_reverse_process(
        self,
        xT: torch.Tensor,
        guidance_scale: float,
        prompt: Union[str, list[str]],
        etas: Optional[Union[List, int, float]] = 0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        prog_bar: bool = True,
        zs: Optional[torch.Tensor] = None,
        controller: Optional[Any] = None,
    ):
        do_cfg = guidance_scale > 1

        # Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt,
            do_cfg,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if etas is None:
            etas = 0
        if type(etas) in [int, float]:
            etas = [etas] * self.scheduler.num_inference_steps
        assert len(etas) == self.scheduler.num_inference_steps
        timesteps = self.scheduler.timesteps.to(self.device)

        xt = xT.clone()
        op = tqdm(timesteps[-zs.shape[1] :]) if prog_bar else timesteps[-zs.shape[1] :]

        t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[1] :])}

        for t in op:
            idx = (
                self.scheduler.num_inference_steps
                - t_to_idx[int(t)]
                - (self.scheduler.num_inference_steps - zs.shape[1] + 1)
            )
            with torch.no_grad():
                latent_model_input = torch.cat([xt] * 2) if do_cfg else xt
                noise_pred = self.unet(
                    latent_model_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

            z = zs[:, idx, ...] if not zs is None else None

            # perform guidance
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # 2. compute less noisy image and set x_t -> x_t-1
            xt = self.reverse_step(noise_pred, t, xt, eta=etas[idx], variance_noise=z)
            if controller is not None:
                xt = controller.step_callback(xt)
        return xt, zs

    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.vae.encode(image).latent_dist.sample(generator)

            latents = self.vae.config.scaling_factor * latents

        if batch_size != latents.shape[0]:
            if batch_size % latents.shape[0] == 0:
                # expand image_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                deprecate(
                    "len(prompt) != len(image)",
                    "1.0.0",
                    deprecation_message,
                    standard_warn=False,
                )
                additional_latents_per_image = batch_size // latents.shape[0]
                latents = torch.cat([latents] * additional_latents_per_image, dim=0)
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts."
                )
        else:
            latents = torch.cat([latents], dim=0)

        return latents

    def ef_ddpm_inversion(
        self,
        prompt: Union[str, list[str]],
        image: PipelineImageInput = None,
        height: int = 512,
        width: int = 512,
        prompt_embeds: Optional[torch.Tensor] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        eta: Union[float, List[float]] = 1.0,
        generator: Optional[torch.Generator] = None,
    ):
        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Preprocess image
        image = self.image_processor.preprocess(image, height=height, width=width)

        # Prepare latent variables
        w0 = self.prepare_image_latents(
            image, batch_size, self.vae.dtype, self.device, generator
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # find Zs and wts - forward process
        wt, zs, wts = self.inversion_forward_process(
            x0=w0,
            prompt=prompt,
            etas=eta,
            num_inference_steps=num_inference_steps,
            guidance_scale_src=guidance_scale,
            generator=generator,
        )

        return wts, zs

    def sample_ddpm(
        self,
        latents: torch.Tensor,
        zs: torch.Tensor,
        prompt: Union[str, list[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        eta: Union[float, List[float]] = 1.0,
        T_skip: Union[int, float] = 0.36,
        output_type: str = "pt",
        return_dict: bool = True,
        controller: Optional[Any] = None,
        image: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        if isinstance(T_skip, float):
            # support T_skip which is a fraction of num_inference_steps
            T_skip = int(T_skip * num_inference_steps)

        # reverse process (via Zs and wT)
        latents, _ = self.inversion_reverse_process(
            xT=latents[:, num_inference_steps - T_skip],
            etas=eta,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            prog_bar=True,
            num_images_per_prompt=num_images_per_prompt,
            zs=zs[:, : (num_inference_steps - T_skip)],
            controller=controller,
        )

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
