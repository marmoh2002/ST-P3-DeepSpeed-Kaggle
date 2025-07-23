
import sys
sys.path.append('/kaggle/working/grad_proj')
sys.path.append('/kaggle/working/nuscenes-devkit/python-sdk')
# To ignore warinings
import warnings
warnings.filterwarnings('ignore')

import os
import time
import socket
import torch
# import pytorch_lightning as pl
import lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint

from stp3.config import get_parser, get_cfg
from stp3.datas.dataloaders import prepare_dataloaders
from stp3.trainer import TrainingModule
from torchmetrics.classification import StatScores
from torchmetrics import Metric
from torchmetrics.functional import stat_scores
from torchmetrics.utilities import reduce
from nuscenes.map_expansion.map_api import NuScenesMap
import nuscenes
import gc
from nuscenes.nuscenes import NuScenes
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.strategies import DeepSpeedStrategy
# This is crucial for reproducibility.  Set these *before* importing your modules.
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"  # or "nccl" if appropriate
from huggingface_hub import hf_hub_download
from huggingface_hub import upload_file
from huggingface_hub import HfApi
api = HfApi()
from kaggle_secrets import UserSecretsClient
secret_label = "hf"
secret_value = UserSecretsClient().get_secret(secret_label)
token = secret_value

gc.collect()
torch.cuda.empty_cache()

######################
#     DEEPSPEED     #
######################
def training_loop(epochs=None):
    checkpoint_parser = get_parser()
    checkpoint_parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    
    # Checkpoint mode arguments (mutually exclusive)
    # checkpoint_group = checkpoint_parser.add_mutually_exclusive_group(required=True)
    checkpoint_parser.add_argument('--checkpoint_disable', action='store_true', 
                                 help='Disable checkpoint loading')
    checkpoint_parser.add_argument('--checkpoint_huggingface', type=str, metavar='REPO_NAME',
                                 help='Load checkpoint from HuggingFace repository (provide repo name)')
    checkpoint_parser.add_argument('--checkpoint_path', type=str, metavar='PATH',
                                 help='Load checkpoint from local path')
    
    # Parse checkpoint arguments first
    cfg_args, remaining_args = checkpoint_parser.parse_known_args(
        args=[] if 'ipykernel' in sys.modules else sys.argv[1:]
    )
    
    
    print(f'Checkpoint args: {cfg_args}')
    print(f'Config args: {cfg_args}')
    
    # Handle checkpoint loading based on mode
    checkpoint_path = None
    load_weights = False
    if cfg_args.checkpoint_disable:
        print("******************Checkpoint loading disabled******************")
        load_weights = False
        
    elif cfg_args.checkpoint_huggingface:
        print(f"******************Loading checkpoint from HuggingFace repo: {cfg_args.checkpoint_huggingface}******************")
        try:
            checkpoint_path = hf_hub_download(
                repo_id=cfg_args.checkpoint_huggingface,
                filename="checkpoints/last.ckpt",  # You might want to make this configurable too
                repo_type="model", 
                token=token
            )
            load_weights = True
        except Exception as e:
            print(f"Error loading from HuggingFace: {e}")
            print("Falling back to no checkpoint loading")
            load_weights = False

    elif cfg_args.checkpoint_path:
        print(f"******************Loading checkpoint from path: {cfg_args.checkpoint_path}******************")
        if os.path.exists(cfg_args.checkpoint_path):
            checkpoint_path = cfg_args.checkpoint_path
            load_weights = True
        else:
            print(f"Error: Checkpoint path {cfg_args.checkpoint_path} does not exist")
            print("Falling back to no checkpoint loading")
            load_weights = False
    
    cfg = get_cfg(cfg_args)
    print(f'Config: {cfg.convert_to_dict()}')
    if load_weights and checkpoint_path:
        cfg.PRETRAINED.PATH = checkpoint_path
        cfg.PRETRAINED.LOAD_WEIGHTS =True
    else:
        cfg.PRETRAINED.PATH = None
        cfg.PRETRAINED.LOAD_WEIGHTS = False
    # cfg.PLANNING.ENABLED = True
    strategy = DeepSpeedStrategy()
    trainloader, valloader = prepare_dataloaders(cfg)
    model = TrainingModule(cfg.convert_to_dict())
    # model.profiler=profiler
    if cfg.PRETRAINED.LOAD_WEIGHTS and checkpoint_path:
        # Load single-image instance segmentation model.
        pretrained_model_weights = torch.load(
            cfg.PRETRAINED.PATH, map_location='cpu'
        )['state_dict']
        state = model.state_dict()
        pretrained_model_weights = {k: v for k, v in pretrained_model_weights.items() if k in state and 'decoder' not in k}
        model.load_state_dict(pretrained_model_weights, strict=False)
        print(f'******************Loaded single-image model weights from {cfg.PRETRAINED.PATH}******************')
    else:
        print(f'******************haven\'t loaded single-image model weights from {cfg.PRETRAINED.PATH} because cfg.PRETRAINED.LOAD_WEIGHTS is set to {cfg.PRETRAINED.LOAD_WEIGHTS}******************')

    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H_%M_%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor='step_val_seg_iou_dynamic',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
        mode='min'
    ) 
    ep= cfg_args.epochs if cfg_args.epochs is not None else 2
    print(f'******************Training for {ep} epochs******************')
    # print(f'******************{cfg.EPOCHS}******************')
    trainer = pl.Trainer(
        strategy=strategy,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.GPUS,
        # precision=cfg.PRECISION,
        precision = "32",
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs= ep,#cfg.EPOCHS,
        logger=tb_logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        # profiler=profiler,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, trainloader, valloader)

if __name__ == "__main__":
    training_loop()