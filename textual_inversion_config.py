import os
from typing import Optional, Union, List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class EmbeddingManagerConfig:
    placeholder_strings: list
    initializer_words: list
    per_image_tokens: bool
    num_vectors_per_token: int
    progressive_words: bool


@dataclass
class DatasetConfig:
    image_dir: Path
    name: Optional[str] = None
    image_edits_dir: Optional[Path] = None
    synset_ids: Optional[List[str]] = None
    eval_dir: Optional[Path] = None
    repeats: Optional[int] = 1
    img_size: Optional[int] = 512
    subset: Optional[int] = None
    use_prefix: Optional[bool] = False
    classes: Optional[list] = None
    difficulty: Optional[str] = None
    index: Optional[int] = None
    bbox_dir: Optional[Path] = None
    num_samples: Optional[int] = None
    train_subset: Optional[str] = None



@dataclass
class LDMConfig:
    embedding_config: EmbeddingManagerConfig


@dataclass
class TextualInversionConfig:
    diffusion: LDMConfig
    dataset: DatasetConfig
    batch_size: int
    epochs: int
    learning_rate: float
    device: int
    output_dir: Path
    num_inference_steps: int
    guidance_scale: float
    ckpt_path: Optional[Path] = None
    clf_weights_path: Optional[Path] = None

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class BPTTConfig:
    # based on AlignProp config.train
    # optimizer
    lr_text_embed: float
    lr_lora: float

    # flags
    use_lora: bool
    grad_checkpoint: bool
    truncated_backprop: bool

    batch_size: int  # per gpu
    num_gpus: Optional[int] = None

    adam_beta1: Optional[float] = 0.9
    adam_beta2: Optional[float] = 0.99
    adam_weight_decay: Optional[float] = 1e-2
    adam_epsilon: Optional[float] = 1e-8
    scale_lr: Optional[bool] = True

    truncated_backprop_minmax: Union[tuple, list] = (35, 45)
    gradient_accumulation_steps: Optional[int] = None

    # interpolation
    temperature: Optional[float] = 1.0
    alpha_margin: Optional[float] = 0.0
    sample_sub_alphas: Optional[bool] = True

    # free-guidance drop prob
    conditioning_dropout_prob: Optional[float] = None


@dataclass
class InstructInversionBPTTConfig:
    diffusion: LDMConfig
    dataset: DatasetConfig
    train: BPTTConfig
    log_dir: Path
    epochs: int
    num_inference_steps: int
    mixed_precision: Optional[str] = "no"
    device: Optional[int] = 0
    seed: Optional[int] = 42
    c_reg: Optional[float] = 1.0
    allow_tf32: Optional[bool] = True
    guidance_scale: Optional[float] = 7.5
    image_guidance_scale: Optional[float] = 1.5
    use_clip_templates: Optional[bool] = False
    log_interpretability: Optional[bool] = False
    debug: Optional[bool] = False
    ckpt_path: Optional[Union[Path, list]] = None
    clf_weights_path: Optional[Path] = None

    def __post_init__(self):
        self.log_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class EvalConfig:
    diffusion: LDMConfig
    dataset: DatasetConfig
    ckpt_path: Union[Path, list[Path]]
    output_dir: Path
    batch_size: int
    num_inference_steps: int
    device: int
    guidance_scale: Optional[float] = 7.5
    image_guidance_scale: Optional[float] = 1.5
    clf_weights: Optional[Path] = None
    lora_path: Optional[Path] = None

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class ClassifierTrainConfig:
    dataset: DatasetConfig
    batch_size: int
    log_dir: Path
    lr: float
    epochs: int
    patience: int
    device: int
    clf_weights: Optional[Path] = None
    classes: Optional[list] = None
    file_list_paths: Optional[list[Path]] = None
    clip_image_embeds_dir: Optional[Path] = None


    def __post_init__(self):
        self.log_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class ZeroInversionBPTTConfig:
    diffusion: LDMConfig
    dataset: DatasetConfig
    train: BPTTConfig
    epochs: int
    log_dir: Path
    num_inference_steps: int
    guidance_scale: float
    device: Optional[int] = 0
    ckpt_path: Optional[Path] = None
    clf_weights_path: Optional[Path] = None

    def __post_init__(self):
        self.log_dir.mkdir(exist_ok=True, parents=True)




@dataclass
class KandinskyEvalConfig:
    dataset: DatasetConfig
    output_dir: Path
    batch_size: int
    device: Optional[int] = 0
    num_inference_steps: Optional[int] = 100
    num_validation_images: Optional[int] = 8
    num_images: Optional[int] = 10000
    ckpt: Optional[int] = 0
    guidance_scale: Optional[float] = 7.5
    clf_weights: Optional[Path] = None
    clip_image_embeds_dir: Optional[Path] = None
    embed_filenames: Optional[list[str]] = None
    lora_weights_dir: Optional[Path] = None
    eval_clf_weights: Optional[Path] = None
    file_list_paths: Optional[list[Path]] = None
    analyze_results: Optional[bool] = True

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
@dataclass
class P2PZeroEvalConfig:
    dataset: DatasetConfig
    output_dir: Path
    source_prompt: str  
    target_prompt: str
    batch_size: int
    target: int
    device: Optional[int] = 0
    num_inference_steps: Optional[int] = 100
    guidance_scale: Optional[float] = 7.5
    cross_attention_guidance_amount: Optional[float] = 0.10
    clf_weights: Optional[Path] = None
    clip_image_embeds_dir: Optional[Path] = None
    debug: Optional[bool] = False
    cfg_path: Optional[Path] = None
    

    llava: bool = False
    
    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)

@dataclass
class DDPMEFEvalConfig:
    skip: int
    cfg_tar: float
    cfg_src: float
    source_prompt: str
    target_prompt: str
    num_diffusion_steps: int
    xa: float
    sa: float
    batch_size: int
    eta: int
    target: int
    mode: str
    
    dataset: DatasetConfig
    output_dir: Path
    batch_size: int
    device: Optional[int] = 0
    num_inference_steps: Optional[int] = 100
    guidance_scale: Optional[float] = 7.5
    clf_weights: Optional[Path] = None
    clip_image_embeds_dir: Optional[Path] = None
    debug: Optional[bool] = False
    llava: bool = False
    cfg_path: Optional[Path] = None
    
    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)

@dataclass
class InstrP2PEvalConfig:
    batch_size: int
    guidance_scale: float
    image_guidance_scale: float
    source_prompt: str
    target_prompt: str
    dataset: DatasetConfig
    output_dir: Path
    target: int
    device: int = 0
    num_inference_steps: int = 100
    clf_weights: Optional[Path] = None
    clip_image_embeds_dir: Optional[Path] = None
    debug: bool = False
    llava: bool = False
    

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)

@dataclass
class KandinskyLoRAConfig:
    dataset: DatasetConfig
    train_batch_size: int
    num_train_epochs: int
    learning_rate: float
    output_dir: Path

    logging_dir: Optional[Union[Path, str]] = "logs"
    seed: Optional[int] = None
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = 1
    gradient_checkpointing: Optional[bool] = True
    pretrained_decoder_model_name_or_path: Optional[str] = (
        "kandinsky-community/kandinsky-2-2-decoder"
    )
    pretrained_prior_model_name_or_path: Optional[str] = (
        "kandinsky-community/kandinsky-2-2-prior"
    )
    snr_gamma: Optional[float] = None
    allow_tf32: Optional[bool] = True
    dataloader_num_workers: Optional[int] = 0
    num_validation_images: Optional[int] = 4

    adam_beta1: Optional[float] = 0.9
    adam_beta2: Optional[float] = 0.999
    adam_weight_decay: Optional[float] = 0.0
    adam_epsilon: Optional[float] = 1e-08

    lr_scheduler: Optional[str] = "constant"
    lr_warmup_steps: Optional[int] = 500
    max_grad_norm: Optional[float] = 1.0
    mixed_precision: Optional[str] = "no"
    report_to: Optional[str] = "wandb"
    checkpointing_steps: Optional[int] = 100
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    local_rank: Optional[int] = 1
    rank: Optional[int] = 4
    lora_alpha: Optional[float] = 1.0

    push_to_hub: Optional[bool] = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None

    num_gpus: Optional[int] = None
    
    # sliders
    guidance_scale: Optional[int] = 1.0
    max_denoising_steps: Optional[int] = 50
    clf_weights: Optional[Path] = None
    clip_embeds_dir: Optional[Path] = None
    embed_filenames: Optional[list[str]] = None
    validation_epochs: Optional[int] = 100
    file_list_paths: Optional[list[Path]] = None
    lora_weights_dir: Optional[Path] = None

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)

@dataclass
class SDLoRAConfig:
    dataset: DatasetConfig
    train_batch_size: int
    num_train_epochs: int
    learning_rate: float
    output_dir: Path

    logging_dir: Optional[Union[Path, str]] = "logs"
    seed: Optional[int] = None
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = 1
    gradient_checkpointing: Optional[bool] = True
    pretrained_model_name_or_path: Optional[str] = "CompVis/stable-diffusion-v1-4"
    snr_gamma: Optional[float] = None
    allow_tf32: Optional[bool] = True
    dataloader_num_workers: Optional[int] = 0
    num_validation_images: Optional[int] = 8
    validation_epochs: Optional[int] = 100

    adam_beta1: Optional[float] = 0.9
    adam_beta2: Optional[float] = 0.999
    adam_weight_decay: Optional[float] = 0.0
    adam_epsilon: Optional[float] = 1e-08

    lr_scheduler: Optional[str] = "constant"
    lr_warmup_steps: Optional[int] = 500
    max_grad_norm: Optional[float] = 1.0
    mixed_precision: Optional[str] = "no"
    report_to: Optional[str] = "wandb"
    checkpointing_steps: Optional[int] = 500
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    local_rank: Optional[int] = 1
    rank: Optional[int] = 4
    lora_alpha: Optional[float] = 1.0
    noise_offset: Optional[float] = 0.0
    prediction_type: Optional[str] = None
    scale_lr: Optional[bool] = True

    push_to_hub: Optional[bool] = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None

    num_gpus: Optional[int] = None

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class UnifiedEvalConfig:
    ddpmef: Optional[DDPMEFEvalConfig]
    instrp2p: Optional[InstrP2PEvalConfig]
    pix2pixzero: Optional[P2PZeroEvalConfig]
    ours: Optional[KandinskyEvalConfig]
    method: str
    device: int

@dataclass
class SDEvalConfig:
    dataset: DatasetConfig
    output_dir: Path
    batch_size: int
    num_vectors: int
    device: Optional[int] = 0
    num_inference_steps: Optional[int] = 100
    num_validation_images: Optional[int] = 8
    text_embeds_path: Optional[Path] = None
    lora_weights_dir: Optional[Path] = None
    eval_clf_weights: Optional[Path] = None
    file_list_paths: Optional[list[Path]] = None
    analyze_results: Optional[bool] = True

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)

@dataclass
class TrainSDTextualInversionConfig:
    dataset: DatasetConfig
    train_batch_size: int
    num_train_epochs: int
    learning_rate: float
    output_dir: Path
    placeholder_token: list[str]
    init_words: list[str]
    train_unet: bool

    # How many textual inversion vectors shall be used to learn the concept
    num_vectors: Optional[int] = 1
    pretrained_model_name_or_path: Optional[str] = "CompVis/stable-diffusion-v1-4"
    seed: Optional[int] = None
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = 1
    gradient_checkpointing: Optional[bool] = True
    # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size
    scale_lr: Optional[bool] = True
    lr_scheduler: Optional[str] = "constant"
    lr_warmup_steps: Optional[int] = 500
    lr_num_cycles: Optional[int] = 1
    dataloader_num_workers: Optional[int] = 0
    logging_dir: Optional[Union[Path, str]] = "logs"
    mixed_precision: Optional[str] = "no"
    allow_tf32: Optional[bool] = True
    report_to: Optional[str] = "wandb"
    num_validation_images: Optional[int] = 8
    validation_epochs: Optional[int] = 1
    validation_steps: Optional[int] = None
    checkpointing_steps: Optional[int] = 500
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

    adam_beta1: Optional[float] = 0.9
    adam_beta2: Optional[float] = 0.999
    adam_weight_decay: Optional[float] = 0.0
    adam_epsilon: Optional[float] = 1e-08

    push_to_hub: Optional[bool] = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None
    no_safe_serialization: Optional[bool] = True
    save_steps: Optional[int] = 500
    save_as_full_pipeline: Optional[bool] = False

    num_gpus: Optional[int] = None

    # LoRA
    rank: Optional[int] = 4
    lora_alpha: Optional[float] = 4.0

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
