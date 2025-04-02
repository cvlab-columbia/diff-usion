import torch
import lpips
import PIL.Image
from enum import Enum, auto
from typing import Optional, Union, Callable, Dict, List, Tuple, Any
from diffusers.image_processor import PipelineImageInput
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from diffusers.models import UNet2DConditionModel, VQModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.configuration_utils import register_to_config
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers import DDIMInverseScheduler, DDIMScheduler
from diffusers.models.attention_processor import Attention
from diffusers.utils.deprecation_utils import deprecate
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.v2 as transforms


class ManipulateMode(Enum):
    cond_avg = auto()
    cond_cls = auto()
    cond_activations = auto()
    target_cfg_avg = auto()
    target_cfg_cls = auto()
    sliders = auto()


class VQVAEImageProcessor(VaeImageProcessor):

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        resample: str = "lanczos",
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = False,
        do_convert_grayscale: bool = False,
    ):
        super().__init__(
            do_resize=do_resize,
            resample=resample,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_rgb=do_convert_rgb,
            do_convert_grayscale=do_convert_grayscale,
        )
        if do_convert_rgb and do_convert_grayscale:
            raise ValueError(
                "`do_convert_rgb` and `do_convert_grayscale` can not both be set to `True`,"
                " if you intended to convert the image into RGB format, please set `do_convert_grayscale = False`.",
                " if you intended to convert the image into grayscale format, please set `do_convert_rgb = False`",
            )


class KandinskyV22PipelineWithInversion(KandinskyV22Pipeline):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        movq: VQModel,
        inverse_scheduler: Optional[DDIMInverseScheduler] = None,
    ):
        super().__init__(unet=unet, scheduler=scheduler, movq=movq)

        self.register_modules(inverse_scheduler=inverse_scheduler)
        self.unet: UNet2DConditionModel
        self.scheduler: DDIMScheduler
        self.movq: VQModel

        # prior model for null image embeds
        self.prior: KandinskyV22PriorPipeline = (
            KandinskyV22PriorPipeline.from_pretrained(
                "kandinsky-community/kandinsky-2-2-prior"
            )
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", subfolder="image_processor"
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", subfolder="image_encoder"
        )

        self.movq_processor = VQVAEImageProcessor()

    # Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_img2img.prepare_image
    @staticmethod
    def prepare_image(pil_image, w=512, h=512):
        pil_image = pil_image.resize((w, h), resample=PIL.Image.BICUBIC, reducing_gap=1)
        arr = np.array(pil_image.convert("RGB"))
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2, 0, 1])
        image = torch.from_numpy(arr).unsqueeze(0)
        return image

    @staticmethod
    def downscale_height_and_width(height, width, scale_factor=8):
        new_height = height // scale_factor**2
        if height % scale_factor**2 != 0:
            new_height += 1
        new_width = width // scale_factor**2
        if width % scale_factor**2 != 0:
            new_width += 1
        return new_height * scale_factor, new_width * scale_factor

    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.movq.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.movq.encode(image).latent_dist.sample(generator)

            init_latents = self.movq.config.scaling_factor * init_latents

        init_latents = torch.cat([init_latents], dim=0)
        latents = init_latents

        return latents

    @torch.no_grad()
    def invert(
        self,
        image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        negative_image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        image: PipelineImageInput = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        device = self._execution_device
        self._guidance_scale = guidance_scale

        # 1. Define call parameters
        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)

        batch_size = image_embeds.shape[0] * num_images_per_prompt
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = torch.cat(negative_image_embeds, dim=0)

        if self.do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_image_embeds = negative_image_embeds.repeat_interleave(
                num_images_per_prompt, dim=0
            )

            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(
                dtype=self.unet.dtype, device=device
            )

        # 2. Preprocess image
        if not isinstance(image, list):
            image = [image]
        if not all(isinstance(i, (PIL.Image.Image, torch.Tensor)) for i in image):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support  PIL image and pytorch tensor"
            )

        image = torch.cat(
            [self.prepare_image(i, width, height) for i in image], dim=0
        ).to(device)

        # 3. Prepare image latent variables (x0)
        latents = self.movq.encode(image).latents
        # latents = self.movq.config.scaling_factor * latents
        # latents = self.prepare_image_latents(
        #     image, batch_size, image_embeds.dtype, device, generator
        # )
        latents = latents.repeat_interleave(num_images_per_prompt, dim=0)

        # 4. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 5. Inversion loop
        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )

            added_cond_kwargs = {"image_embeds": image_embeds}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample

        inverted_latents = latents.detach().clone()

        # Offload all models
        self.maybe_free_model_hooks()

        return inverted_latents

    def next_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
    ):
        timestep, next_timestep = (
            min(
                timestep
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps,
                999,
            ),
            timestep,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = (
            alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        )
        return next_sample

    def get_noise_pred(
        self, latents: torch.Tensor, t: int, context: Union[Dict, torch.Tensor]
    ):
        # expand the latents if we are doing classifier free guidance
        latents_input = (
            torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        )

        noise_pred = self.unet(
            sample=latents_input,
            timestep=t,
            encoder_hidden_states=None,
            added_cond_kwargs=context,
            return_dict=False,
        )[0]

        if self.do_classifier_free_guidance:
            noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            _, variance_pred_text = variance_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

        if not (
            hasattr(self.scheduler.config, "variance_type")
            and self.scheduler.config.variance_type in ["learned", "learned_range"]
        ):
            noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

        return noise_pred

    @torch.no_grad()
    def ddim_inversion(
        self,
        image: PipelineImageInput,
        image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        negative_image_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        latents: Optional[torch.Tensor] = None,
    ):
        device = self._execution_device
        self._guidance_scale = guidance_scale

        # Preprocess image (x0)
        image = self.movq_processor.preprocess(image, height=height, width=width).to(
            self.device
        )
        bsz = image.shape[0]

        # Set cond embeds and null embeds
        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)

        if isinstance(negative_image_embeds, list):
            negative_image_embeds = torch.cat(negative_image_embeds, dim=0)

        if negative_image_embeds is None:
            negative_image_embeds = self.prior.get_zero_embed(batch_size=bsz).to(device)

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(
                dtype=self.unet.dtype, device=device
            )

        # 3. Prepare image latent variables (x0)
        latents = self.movq.encode(image).latents

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Inversion loop
        self._num_timesteps = len(timesteps)
        for i in tqdm(range(self.num_timesteps)):
            t = self.scheduler.timesteps[len(timesteps) - i - 1]
            context = {"image_embeds": image_embeds}
            noise_pred = self.get_noise_pred(latents, t, context)
            latents = self.next_step(noise_pred, t, latents)

        inverted_latents = latents.detach().clone()

        # Offload all models
        self.maybe_free_model_hooks()

        return inverted_latents

    def sample_xts_from_x0(
        self,
        x0: torch.Tensor,
        timesteps: List,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Samples from P(x_1:T|x_0)
        """
        bsz, in_channels, height, width = x0.shape
        alpha_bar = self.scheduler.alphas_cumprod
        sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5

        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(
            (bsz, num_inference_steps + 1, in_channels, height, width)
        ).to(x0.device)
        xts[:, 0, ...] = x0
        for t in reversed(timesteps):
            idx = num_inference_steps - t_to_idx[int(t)]
            noise = randn_tensor(
                shape=x0.shape, generator=generator, device=self.device, dtype=x0.dtype
            )
            xts[:, idx, ...] = (
                x0 * (alpha_bar[t] ** 0.5) + noise * sqrt_one_minus_alpha_bar[t]
            )

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
        num_inference_steps: int = 50,
        image_embeds: Optional[torch.Tensor] = None,
        etas: Optional[Union[List, int, float]] = None,
        generator: Optional[torch.Generator] = None,
    ):
        device = self._execution_device
        bsz, in_channels, height, width = x0.shape
        self.prior.to(device)
        # uncond_embeds = self.prior(prompt=[""] * bsz).image_embeds
        uncond_embeds = self.prior.get_zero_embed(batch_size=bsz)
        do_cfg = guidance_scale_src > 1.0

        timesteps = self.scheduler.timesteps.to(device)
        variance_noise_shape = (
            bsz,
            num_inference_steps,
            in_channels,
            height,
            width
        )
        if etas is None or (type(etas) in [int, float] and etas == 0):
            eta_is_zero = True
            zs = None
        else:
            eta_is_zero = False
            if type(etas) in [int, float]:
                etas = [etas] * num_inference_steps
            xts = self.sample_xts_from_x0(
                x0, timesteps, num_inference_steps, generator=generator
            )
            alpha_bar = self.scheduler.alphas_cumprod
            zs = torch.zeros(size=variance_noise_shape, device=device)
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xt = x0

        for t in tqdm(timesteps):
            idx = num_inference_steps - t_to_idx[int(t)] - 1
            # 1. predict noise residual
            if not eta_is_zero:
                xt = xts[:, idx + 1, ...]

            with torch.no_grad():
                context = {"image_embeds": uncond_embeds}
                out = self.unet(
                    xt,
                    timestep=t,
                    encoder_hidden_states=None,
                    added_cond_kwargs=context,
                )
                if image_embeds is not None:
                    context = {"image_embeds": image_embeds}
                    cond_out = self.unet(
                        xt,
                        timestep=t,
                        encoder_hidden_states=None,
                        added_cond_kwargs=context,
                    )

            if image_embeds is not None:
                ## classifier free guidance
                noise_pred = out.sample + guidance_scale_src * (
                    cond_out.sample - out.sample
                )
            else:
                noise_pred = out.sample

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(x0.shape[1], dim=1)

            if eta_is_zero:
                # ddim
                # 2. compute more noisy image and set x_t -> x_t+1
                xt = self.forward_step(noise_pred, t, xt)

            else:
                # ddpm
                xtm1 = xts[:, idx, ...]
                # pred of x0 -> P(f_t(x_t))
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
                # D(f_t(x_t))
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
        device = self._execution_device
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
                variance_noise = torch.randn(model_output.shape, device=device)
            sigma_z = eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z

        return prev_sample

    def inversion_reverse_process(
        self,
        xT: torch.Tensor,
        guidance_scale: float,
        image_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_image_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        etas: Optional[Union[List, int, float]] = 0,
        prog_bar: bool = False,
        zs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        controller: Optional[Any] = None,
        mode: Optional[ManipulateMode] = None,
        manipulation_scale: Optional[float] = None,
        image: Optional[torch.Tensor] = None,
    ):
        device = self._execution_device
        bsz = xT.shape[0]
        do_cfg = guidance_scale > 1

        if mask is not None:
            latent_dim = xT.shape[-1]
            height, width = mask.shape[-2:]
            if not len(mask.shape) == 4:
                mask = mask.expand(1, 1, height, width)

            if not height == width == latent_dim:
                mask = F.interpolate(
                    mask,
                    (latent_dim, latent_dim),
                    mode="bilinear",
                    align_corners=False,
                ).to(torch.float32)

        if negative_image_embeds is None:
            self.prior.to(device)
            uncond_embeds = self.prior.get_zero_embed(batch_size=bsz)
        elif isinstance(negative_image_embeds, list):
            uncond_embeds = torch.cat(negative_image_embeds, dim=0)

        if uncond_embeds.shape[0] != bsz:
            raise ValueError(
                f"mismatch in batch_size between condition and negative embeds"
            )

        len_image_embeds = 1
        if isinstance(image_embeds, list):
            len_image_embeds = len(image_embeds)
            image_embeds = torch.cat(image_embeds)

        if do_cfg:
            image_embeds = torch.cat([uncond_embeds, image_embeds])

        if etas is None:
            etas = 0
        if type(etas) in [int, float]:
            etas = [etas] * self.scheduler.num_inference_steps
        assert len(etas) == self.scheduler.num_inference_steps
        timesteps = self.scheduler.timesteps.to(device)

        xt = xT.clone()
        op = tqdm(timesteps[-zs.shape[1] :]) if prog_bar else timesteps[-zs.shape[1] :]

        t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[1] :])}
        for t in op:
            idx = (
                self.scheduler.num_inference_steps
                - t_to_idx[int(t)]
                - (self.scheduler.num_inference_steps - zs.shape[1] + 1)
            )

            # for distance guidance
            with torch.no_grad():
                # with torch.enable_grad():
                # x_in = xt.detach().requires_grad_(True)

                latents = torch.cat([xt] * len_image_embeds)
                latent_model_input = torch.cat([xt, latents]) if do_cfg else latents
                context = {"image_embeds": image_embeds}
                noise_pred = self.unet(
                    latent_model_input,
                    timestep=t,
                    encoder_hidden_states=None,
                    added_cond_kwargs=context,
                    return_dict=False,
                )[0]

                z = zs[:, idx] if not zs is None else None

                if not (
                    hasattr(self.scheduler.config, "variance_type")
                    and self.scheduler.config.variance_type
                    in ["learned", "learned_range"]
                ):
                    noise_pred, _ = noise_pred.split(xt.shape[1], dim=1)

                if do_cfg:
                    if (
                        mode == ManipulateMode.target_cfg_avg
                        or mode == ManipulateMode.target_cfg_cls
                    ):
                        (out_uncond, out_target, out_positive, out_negative) = (
                            noise_pred.chunk(4)
                        )
                        out_target_cfg = out_target + manipulation_scale * (
                            out_positive - out_negative
                        )
                        noise_pred = out_uncond + guidance_scale * (
                            out_target_cfg - out_uncond
                        )
                    else:
                        out_uncond, out_cond = noise_pred.chunk(2)
                        noise_pred = out_uncond + guidance_scale * (
                            out_cond - out_uncond
                        )

                        # compute approximate x0_cf
                        # alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        # x0_cf = (
                        #     xt - torch.sqrt(1 - alpha_prod_t) * noise_pred
                        # ) / torch.sqrt(alpha_prod_t)
                        # x_in = x0_cf.requires_grad_(True)

                        # # compute distance guidance
                        # dist_scale = 0.5
                        # # distance_loss = self.compute_fs_loss(
                        # #     cf_latents=x0_cf, input_image=image, t=t
                        # # )
                        # distance_loss = self.compute_perceptual_loss(
                        #     cf_latents=x_in, input_image=image, t=t
                        # )
                        # dist_grad = torch.autograd.grad(distance_loss, x_in)[0]
                        # dist_grad_scaled = dist_grad / torch.norm(dist_grad) * torch.norm(noise_pred)

                        # # gradient ascent / descent
                        # noise_pred = (
                        #     noise_pred
                        #     - dist_scale * torch.sqrt(1 - alpha_prod_t) * dist_grad_scaled
                        # )
                        # # out_cond = out_cond - dist_scale * torch.sqrt(1 - alpha_prod_t) * dist_grad_scaled
                        # # noise_pred = out_uncond + dist_scale * torch.sqrt(1 - alpha_prod_t) * dist_grad_scaled

                        # del distance_loss

                elif (
                    mode == ManipulateMode.target_cfg_avg
                    or mode == ManipulateMode.target_cfg_cls
                ):
                    out_target, out_positive, out_negative = noise_pred.chunk(3)
                    noise_pred = out_target + manipulation_scale * (
                        out_positive - out_negative
                    )

            # 2. compute less noisy image and set x_t -> x_t-1
            xt = self.reverse_step(noise_pred, t, xt, eta=etas[idx], variance_noise=z)

            if controller is not None:
                xt = controller.step_callback(xt)
        return xt, zs

    def ef_ddpm_inversion(
        self,
        source_embeds: torch.Tensor,
        image: PipelineImageInput = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        eta: Union[float, List[float]] = 1.0,
        generator: Optional[torch.Generator] = None,
    ):
        device = self._execution_device

        # Preprocess image (x0)
        image = self.movq_processor.preprocess(image, height=height, width=width).to(
            self.device
        )

        # Prepare image latent variables (x0 -> w0)
        w0 = self.movq.encode(image).latents

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # find Zs and wts - forward process
        wt, zs, wts = self.inversion_forward_process(
            w0,
            etas=eta,
            image_embeds=source_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale_src=guidance_scale,
            generator=generator,
        )

        return wts, zs

    def sample_ddpm(
        self,
        latents: torch.Tensor,
        zs: torch.Tensor,
        target_embeds: torch.Tensor,
        negative_image_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        eta: Union[float, List[float]] = 1.0,
        T_skip: Union[int, float] = 0.36,
        output_type: str = "pil",
        return_dict: bool = True,
        controller: Optional[Any] = None,
        mask: Optional[torch.Tensor] = None,
        mode: Optional[ManipulateMode] = None,
        manipulation_scale: Optional[float] = None,
        image: Optional[torch.Tensor] = None,
    ):
        if isinstance(T_skip, float):
            # support T_skip which is a fraction of num_inference_steps
            T_skip = int(T_skip * num_inference_steps)

        # reverse process (via Zs and wT)
        latents, _ = self.inversion_reverse_process(
            xT=latents[:, num_inference_steps - T_skip],
            etas=eta,
            image_embeds=target_embeds,
            negative_image_embeds=negative_image_embeds,
            guidance_scale=guidance_scale,
            prog_bar=True,
            zs=zs[:, : (num_inference_steps - T_skip)],
            controller=controller,
            mask=mask,
            mode=mode,
            manipulation_scale=manipulation_scale,
            image=image,
        )

        if not output_type == "latent":
            image = self.movq.decode(latents, force_not_quantize=True)["sample"]
            # post-processing
            image = self.movq_processor.postprocess(
                image.detach(), output_type=output_type
            )
        else:
            image = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def sample(
        self,
        image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        negative_image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        initial_embeds: Optional[torch.Tensor] = None,
        start_noise: Optional[int] = None,
        **kwargs,
    ):

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        device = self._execution_device

        self._guidance_scale = guidance_scale

        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)
        batch_size = image_embeds.shape[0] * num_images_per_prompt
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = torch.cat(negative_image_embeds, dim=0)

        if self.do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            init_image_embeds = initial_embeds.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            negative_image_embeds = negative_image_embeds.repeat_interleave(
                num_images_per_prompt, dim=0
            )

            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(
                dtype=self.unet.dtype, device=device
            )
            init_image_embeds = torch.cat(
                [negative_image_embeds, init_image_embeds], dim=0
            ).to(dtype=self.unet.dtype, device=device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels

        height, width = self.downscale_height_and_width(
            height, width, self.movq_scale_factor
        )

        # create initial latent
        latents = self.prepare_latents(
            (batch_size, num_channels_latents, height, width),
            image_embeds.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )

        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )

            if t > start_noise:
                added_cond_kwargs = {"image_embeds": init_image_embeds}
            else:
                added_cond_kwargs = {"image_embeds": image_embeds}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            )[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                image_embeds = callback_outputs.pop("image_embeds", image_embeds)
                negative_image_embeds = callback_outputs.pop(
                    "negative_image_embeds", negative_image_embeds
                )

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        if output_type not in ["pt", "np", "pil", "latent"]:
            raise ValueError(
                f"Only the output types `pt`, `pil` and `np` are supported not output_type={output_type}"
            )

        if not output_type == "latent":
            image = self.movq.decode(latents, force_not_quantize=True)["sample"]
            # post-processing
            image = self.movq_processor.postprocess(
                image.detach(), output_type=output_type
            )
        else:
            image = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    @staticmethod
    def normalize(image_embeds: torch.Tensor, data_embeds: torch.Tensor):
        cond_mean = data_embeds.mean(0)
        cond_std = data_embeds.std(0)
        cond = (image_embeds - cond_mean) / cond_std
        return cond

    @staticmethod
    def denormalize(image_embeds: torch.Tensor, data_embeds: torch.Tensor):
        cond_mean = data_embeds.mean(0)
        cond_std = data_embeds.std(0)
        cond = (image_embeds * cond_std) + cond_mean
        return cond

    def compute_perceptual_loss(
        self, cf_latents: torch.Tensor, input_image: torch.Tensor, t
    ) -> torch.Tensor:
        device = self._execution_device
        cf_image = self.movq.decode(cf_latents, force_not_quantize=True)["sample"]
        # post-processing
        cf_image = self.movq_processor.postprocess(cf_image, output_type="pt")

        model = self.lpips_model
        model.to(device)

        # Define a transform to prepare images
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (256, 256)
                ),  # LPIPS model expects images of this size
            ]
        )

        # Load and preprocess the images
        cf_tensor = transform(cf_image)
        f_tensor = transform(input_image)

        # Compute the LPIPS similarity score
        distance = model(cf_tensor, f_tensor)

        a = [to_pil_image(x) for x in cf_image]
        a[0].save(f"lpips_check_x0_cf_{t}.png")

        return distance

    def compute_fs_loss(
        self, cf_latents: torch.Tensor, input_image: torch.Tensor, t
    ) -> torch.Tensor:
        device = self._execution_device
        cf_image = self.movq.decode(cf_latents, force_not_quantize=True)["sample"]
        # post-processing
        cf_image = self.movq_processor.postprocess(cf_image, output_type="pt")

        model = self.facenet_model
        model.to(device)

        transform = transforms.Compose([transforms.Resize((160, 160))])

        # Load and preprocess the images
        cf_tensor = transform(cf_image)
        f_tensor = transform(input_image)
        cf_embedding = model(cf_tensor)
        f_embedding = model(f_tensor)

        similarity = F.cosine_similarity(cf_embedding, f_embedding)
        distance = -similarity

        # distance = F.pairwise_distance(cf_embedding, f_embedding)

        # l1 distance
        # distance = torch.abs(cf_image - input_image).sum(dim=[1,2,3])

        a = [to_pil_image(x) for x in cf_image]
        a[0].save(f"fs_check_x0_cf_{t}.png")

        return distance


class KandinskyV22PipelineWithInversionInpaint(KandinskyV22PipelineWithInversion):
    def inversion_reverse_process(
        self,
        xT: torch.Tensor,
        guidance_scale_target: float,
        negative_image_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        etas: Optional[Union[List, int, float]] = 0,
        image_embeds: Optional[torch.Tensor] = None,
        prog_bar: bool = False,
        zs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        controller: Optional[Any] = None,
    ):
        device = self._execution_device
        bsz = xT.shape[0]

        if mask is not None:
            latent_dim = xT.shape[-1]
            height, width = mask.shape[-2:]
            if not len(mask.shape) == 4:
                mask = mask.expand(1, 1, height, width)

            if not height == width == latent_dim:
                mask = F.interpolate(
                    mask,
                    (latent_dim, latent_dim),
                    mode="bilinear",
                    align_corners=False,
                ).to(torch.float32)

            noise = randn_tensor(shape=x0.shape, device=self.device, dtype=x0.dtype)

        if negative_image_embeds is None:
            self.prior.to(device)
            uncond_embeds = self.prior.get_zero_embed(batch_size=bsz)
        elif isinstance(negative_image_embeds, list):
            uncond_embeds = torch.cat(negative_image_embeds, dim=0)
            # uncond_embeds = self.prior(prompt=[""] * bsz).image_embeds

        if uncond_embeds.shape[0] != bsz:
            raise ValueError(
                f"mismatch in batch_size between condition and negative embeds"
            )

        cfg_scales_tensor = (
            torch.Tensor(guidance_scale_target).view(-1, 1, 1, 1).to(device)
        )

        if etas is None:
            etas = 0
        if type(etas) in [int, float]:
            etas = [etas] * self.scheduler.num_inference_steps
        assert len(etas) == self.scheduler.num_inference_steps
        timesteps = self.scheduler.timesteps.to(device)

        xt = xT.clone()
        new_timesteps = timesteps[-zs.shape[1] :]
        op = tqdm(new_timesteps) if prog_bar else new_timesteps

        t_to_idx = {int(v): k for k, v in enumerate(new_timesteps)}

        for t in op:
            idx = (
                self.scheduler.num_inference_steps
                - t_to_idx[int(t)]
                - (self.scheduler.num_inference_steps - zs.shape[1] + 1)
            )

            with torch.no_grad():
                context = {"image_embeds": uncond_embeds}
                uncond_out = self.unet(
                    xt,
                    timestep=t,
                    encoder_hidden_states=None,
                    added_cond_kwargs=context,
                )
                if image_embeds is not None:
                    context = {"image_embeds": image_embeds}
                    cond_out = self.unet(
                        xt,
                        timestep=t,
                        encoder_hidden_states=None,
                        added_cond_kwargs=context,
                    )

            z = zs[:, idx] if not zs is None else None

            if image_embeds is not None:
                ## classifier free guidance
                noise_pred = uncond_out.sample + cfg_scales_tensor * (
                    cond_out.sample - uncond_out.sample
                )

            else:
                noise_pred = uncond_out.sample

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(xt.shape[1], dim=1)

            # 2. compute less noisy image and set x_t -> x_t-1
            xt = self.reverse_step(noise_pred, t, xt, eta=etas[idx], variance_noise=z)

            # masking
            if mask is not None:
                mask_latent = mask.expand(-1, 4, -1, -1)
                init_latents = x0
                # if idx < len(new_timesteps) - 1:
                #     noise_timestep = new_timesteps[idx]
                #     init_latents = self.scheduler.add_noise(
                #         init_latents, noise, torch.tensor(noise_timestep)
                #     )
                # apply mask on latents
                # xt = mask_latent * init_latents + (1 - mask_latent) * xt
                xt = (1 - mask_latent) * init_latents + mask_latent * xt

            if controller is not None:
                xt = controller.step_callback(xt)
        return xt, zs

    def ef_ddpm_inversion(
        self,
        source_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        negative_image_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        image: PipelineImageInput = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale_src: float = 4.0,
        guidance_scale_target: float = 4.0,
        eta: Union[float, List[float]] = 1.0,
        T_skip: Union[int, float] = 0.36,
        output_type: str = "pil",
        return_dict: bool = True,
        controller: Optional[Any] = None,
        mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        device = self._execution_device
        if isinstance(T_skip, float):
            # support T_skip which is a fraction of num_inference_steps
            T_skip = int(T_skip * num_inference_steps)

        # Preprocess image (x0)
        image = self.movq_processor.preprocess(image, height=height, width=width).to(
            self.device
        )

        # Prepare image latent variables (x0 -> w0)
        w0 = self.movq.encode(image).latents

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # find Zs and wts - forward process
        wt, zs, wts = self.inversion_forward_process(
            w0,
            etas=eta,
            image_embeds=source_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale_src=guidance_scale_src,
            generator=generator,
        )

        # reverse process (via Zs and wT)
        x0 = None
        if mask is not None:
            x0 = w0
        latents, _ = self.inversion_reverse_process(
            xT=wts[:, num_inference_steps - T_skip],
            etas=eta,
            image_embeds=target_embeds,
            negative_image_embeds=negative_image_embeds,
            guidance_scale_target=[guidance_scale_target],
            prog_bar=True,
            zs=zs[:, : (num_inference_steps - T_skip)],
            controller=controller,
            x0=x0,
            mask=mask,
        )

        if not output_type == "latent":
            image = self.movq.decode(latents, force_not_quantize=True)["sample"]
            # post-processing
            image = self.movq_processor.postprocess(
                image.detach(), output_type=output_type
            )
        else:
            image = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class Pix2PixZeroL2Loss:
    def __init__(self):
        self.loss = 0.0

    def compute_loss(self, predictions, targets):
        self.loss += ((predictions - targets) ** 2).sum((1, 2)).mean(0)


class Pix2PixZeroAttnProcessor:
    """An attention processor class to store the attention weights.
    In Pix2Pix Zero, it happens during computations in the cross-attention blocks."""

    def __init__(self, is_pix2pix_zero=False):
        self.is_pix2pix_zero = is_pix2pix_zero
        if self.is_pix2pix_zero:
            self.reference_cross_attn_map = {}

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        timestep=None,
        loss=None,
    ):
        batch_size, sequence_length, *_ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if self.is_pix2pix_zero and timestep is not None:
            # new bookkeeping to save the attention weights.
            if loss is None:
                self.reference_cross_attn_map[timestep.item()] = (
                    attention_probs.detach().cpu()
                )
            # compute loss
            elif loss is not None:
                prev_attn_probs = self.reference_cross_attn_map.pop(timestep.item())
                loss.compute_loss(
                    attention_probs, prev_attn_probs.to(attention_probs.device)
                )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class KandinskyV22PipelineXAGuidance(KandinskyV22PipelineWithInversion):

    @staticmethod
    def construct_direction(
        source_embeds: torch.Tensor, target_embeds: torch.Tensor
    ) -> torch.Tensor:
        edit_direction = (source_embeds.mean(0) - target_embeds.mean(0)).unsqueeze(0)
        return edit_direction

    @torch.no_grad()
    def __call__(
        self,
        image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        source_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        negative_image_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        cross_attention_guidance_amount: float = 0.15,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image_embeds (`torch.Tensor` or `List[torch.Tensor]`):
                The clip image embeddings for text prompt, that will be used to condition the image generation.
            negative_image_embeds (`torch.Tensor` or `List[torch.Tensor]`):
                The clip image embeddings for negative text prompt, will be used to condition the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        device = self._execution_device
        self._guidance_scale = guidance_scale
        bsz = image_embeds.shape[0]

        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)
        batch_size = image_embeds.shape[0] * num_images_per_prompt
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = torch.cat(negative_image_embeds, dim=0)

        if negative_image_embeds is None:
            negative_image_embeds = self.prior.get_zero_embed(batch_size=bsz).to(device)

        if self.do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_image_embeds = negative_image_embeds.repeat_interleave(
                num_images_per_prompt, dim=0
            )

            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(
                dtype=self.unet.dtype, device=device
            )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels

        height, width = self.downscale_height_and_width(
            height, width, self.movq_scale_factor
        )

        # create initial latent
        latents = self.prepare_latents(
            (batch_size, num_channels_latents, height, width),
            image_embeds.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )
        latents_init = latents.clone()

        # Rejig the UNet so that we can obtain the cross-attenion maps and
        # use them for guiding the subsequent image generation.
        self.unet = prepare_unet(self.unet)

        # Denoising loop where we obtain the cross-attention maps.
        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )

            added_cond_kwargs = {"image_embeds": image_embeds}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs={"timestep": t},
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            )[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                image_embeds = callback_outputs.pop("image_embeds", image_embeds)
                negative_image_embeds = callback_outputs.pop(
                    "negative_image_embeds", negative_image_embeds
                )

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        # Compute the edit directions.
        edit_direction = self.construct_direction(source_embeds, target_embeds).to(
            image_embeds.device
        )

        # Edit the prompt embeddings as per the edit directions discovered.
        image_embeds_edit = image_embeds.clone()
        # only add edit direction to positive embeds
        image_embeds_edit[batch_size:] += edit_direction

        # Second denoising loop to generate the edited image.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latents = latents_init
        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )

            # we want to learn the latent such that it steers the generation
            # process towards the edited direction, so make the make initial
            # noise learnable
            x_in = latent_model_input.detach().clone()
            x_in.requires_grad = True

            # optimizer
            opt = torch.optim.SGD([x_in], lr=cross_attention_guidance_amount)

            with torch.enable_grad():
                # initialize loss
                loss = Pix2PixZeroL2Loss()

                # predict the noise residual
                added_cond_kwargs = {"image_embeds": image_embeds_edit.detach()}
                noise_pred = self.unet(
                    x_in,
                    t,
                    encoder_hidden_states=None,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs={"timestep": t, "loss": loss},
                ).sample

                loss.loss.backward(retain_graph=False)
                opt.step()

            # recompute the noise
            added_cond_kwargs = {"image_embeds": image_embeds_edit}
            noise_pred = self.unet(
                x_in.detach(),
                t,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs={"timestep": None},
            ).sample

            latents = x_in.detach().chunk(2)[0]

            if self.do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            )[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                image_embeds = callback_outputs.pop("image_embeds", image_embeds)
                negative_image_embeds = callback_outputs.pop(
                    "negative_image_embeds", negative_image_embeds
                )

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        if output_type not in ["pt", "np", "pil", "latent"]:
            raise ValueError(
                f"Only the output types `pt`, `pil` and `np` are supported not output_type={output_type}"
            )

        if not output_type == "latent":
            # post-processing
            image = self.movq.decode(latents, force_not_quantize=True)["sample"]
            if output_type in ["np", "pil"]:
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            if output_type == "pil":
                image = self.numpy_to_pil(image)
        else:
            image = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


def prepare_unet(unet: UNet2DConditionModel):
    """Modifies the UNet (`unet`) to perform Pix2Pix Zero optimizations."""
    pix2pix_zero_attn_procs = {}
    for name in unet.attn_processors.keys():
        module_name = name.replace(".processor", "")
        module = unet.get_submodule(module_name)
        if "attn2" in name:
            pix2pix_zero_attn_procs[name] = Pix2PixZeroAttnProcessor(
                is_pix2pix_zero=True
            )
            module.requires_grad_(True)
        else:
            pix2pix_zero_attn_procs[name] = Pix2PixZeroAttnProcessor(
                is_pix2pix_zero=False
            )
            module.requires_grad_(False)

    unet.set_attn_processor(pix2pix_zero_attn_procs)
    return unet
