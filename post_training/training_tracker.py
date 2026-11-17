#!/usr/bin/env python3
"""GPU setup and CSV/WandB logging helpers."""

import csv
import shutil
import subprocess
from pathlib import Path

import torch

POST_TRAINING_DIR = Path(__file__).resolve().parent


def detect_nvidia_smi_gpu():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=5,
        )
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


def configure_nvidia_gpu(enable_amp=True, auto_install_cuda=False):
    hw_gpu = detect_nvidia_smi_gpu()

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        bf16_ok = bool(torch.cuda.is_bf16_supported())

        # Enable TF32 and cudnn autotuner on Ampere/Ada GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        use_amp = bool(enable_amp)
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        info = {
            "device": "cuda",
            "gpu_name": props.name,
            "vram_gb": round(props.total_memory / (1024**3), 2),
            "cuda_version": torch.version.cuda,
            "compute_capability": f"{props.major}.{props.minor}",
            "tf32_enabled": True,
            "amp_enabled": use_amp,
            "amp_dtype": "bfloat16" if bf16_ok else "float16",
            "nvidia_smi_detected": hw_gpu,
        }
        print(f"[gpu] {props.name} ({info['vram_gb']} GB, CUDA {torch.version.cuda}, amp={info['amp_dtype']})")
        return "cuda", use_amp, amp_dtype, info

    if hw_gpu:
        print(f"[warning] GPU detected ({hw_gpu}) but torch ({torch.__version__}) has no CUDA. Run setup_env.ps1.")

    info = {
        "device": "cpu",
        "gpu_name": None,
        "vram_gb": 0.0,
        "cuda_version": None,
        "compute_capability": None,
        "tf32_enabled": False,
        "amp_enabled": False,
        "amp_dtype": "float32",
        "nvidia_smi_detected": hw_gpu,
    }
    return "cpu", False, torch.float32, info


def current_gpu_mem_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated(0) / (1024**2), 1)
    return 0.0


class TrainingTracker:
    def __init__(self, stage_name, log_dir=POST_TRAINING_DIR / "logs", use_wandb=False,
                 wandb_project="chess-post-training", config=None):
        self.stage_name = stage_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.steps_csv = self.log_dir / f"{stage_name}_steps.csv"
        self.val_csv = self.log_dir / f"{stage_name}_val.csv"
        self.steps_csv.unlink(missing_ok=True)
        self.val_csv.unlink(missing_ok=True)

        self._step_fields = None
        self._val_fields = None
        self.wandb_run = None

        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=wandb_project, name=stage_name, config=config or {})
            except Exception as e:
                print(f"[tracker] wandb init failed ({e}), logging to CSV only")

    def _append_csv(self, path, fields_attr, row):
        fields = getattr(self, fields_attr)
        mode = "w" if fields is None else "a"
        if fields is None:
            fields = list(row.keys())
            setattr(self, fields_attr, fields)
        with open(path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if mode == "w":
                writer.writeheader()
            writer.writerow(row)

    def log_step(self, record):
        self._append_csv(self.steps_csv, "_step_fields", record)
        if self.wandb_run is not None:
            self.wandb_run.log({f"train/{k}": v for k, v in record.items() if isinstance(v, (int, float))})

    def log_val(self, record):
        flat = {k: v for k, v in record.items() if not isinstance(v, (dict, list))}
        self._append_csv(self.val_csv, "_val_fields", flat)
        if self.wandb_run is not None:
            self.wandb_run.log({f"val/{k}": v for k, v in flat.items() if isinstance(v, (int, float))})

    def close(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()
