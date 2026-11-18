#!/usr/bin/env python3
"""Check base model weights, tokenizer, GPU setup, and Stockfish binary."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

import chess

POST_TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = POST_TRAINING_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(POST_TRAINING_DIR))

from training_tracker import configure_nvidia_gpu

SPACE_REPO_ID = "nathanael-fijalkow/chess-post-training"


def ensure_repo_files(root=REPO_ROOT):
    base_weights = root / "base_model" / "model.safetensors"
    openings_file = root / "chess_hw" / "openings.txt.gz"

    needs_dl = not base_weights.exists() or not openings_file.exists()
    if not needs_dl:
        for p in (base_weights, openings_file):
            with open(p, "rb") as f:
                if f.read(24) == b"version https://git-lfs.":
                    needs_dl = True
                    break

    if needs_dl:
        print(f"[inspect] downloading repo files from {SPACE_REPO_ID}...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=SPACE_REPO_ID, repo_type="space", local_dir=str(root))


def _verify_stockfish(path):
    try:
        p = subprocess.run([path], input="uci\nquit\n", text=True, capture_output=True, timeout=5)
        return "uciok" in p.stdout
    except Exception:
        return False


def download_stockfish_binary(bin_dir):
    bin_dir.mkdir(parents=True, exist_ok=True)
    sys_name = platform.system().lower()
    headers = {"User-Agent": "Mozilla/5.0"}

    if sys_name == "windows":
        urls = [
            "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-windows-x86-64-avx2.zip",
            "https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-windows-x86-64-avx2.zip",
        ]
        archive = bin_dir / "stockfish_win.zip"
        for url in urls:
            try:
                print(f"[inspect] downloading Stockfish from {url}")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp, open(archive, "wb") as out:
                    shutil.copyfileobj(resp, out)
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(bin_dir)
                archive.unlink(missing_ok=True)
                for p in bin_dir.rglob("stockfish*.exe"):
                    target = bin_dir / "stockfish.exe"
                    if p.resolve() != target.resolve():
                        shutil.copy2(p, target)
                    return target
            except Exception as e:
                print(f"[inspect] download failed ({url}): {e}")
                archive.unlink(missing_ok=True)

    elif sys_name == "linux":
        urls = [
            "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-ubuntu-x86-64-avx2.tar",
            "https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-ubuntu-x86-64-avx2.tar",
        ]
        archive = bin_dir / "stockfish_linux.tar"
        for url in urls:
            try:
                print(f"[inspect] downloading Stockfish from {url}")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp, open(archive, "wb") as out:
                    shutil.copyfileobj(resp, out)
                with tarfile.open(archive, "r") as tf:
                    tf.extractall(bin_dir)
                archive.unlink(missing_ok=True)
                for p in bin_dir.rglob("stockfish*"):
                    if p.is_file() and not p.name.endswith(".tar"):
                        target = bin_dir / "stockfish"
                        if p.resolve() != target.resolve():
                            shutil.copy2(p, target)
                        target.chmod(target.stat().st_mode | stat.S_IEXEC)
                        return target
            except Exception as e:
                print(f"[inspect] download failed ({url}): {e}")
                archive.unlink(missing_ok=True)

    return None


def resolve_stockfish_path(custom_path=None, auto_download=False):
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    if os.environ.get("STOCKFISH_PATH"):
        candidates.append(os.environ["STOCKFISH_PATH"])

    for marker in (POST_TRAINING_DIR / ".stockfish_path", REPO_ROOT / ".stockfish_path"):
        if marker.exists():
            saved = marker.read_text(encoding="utf-8").strip()
            if saved:
                candidates.append(saved)

    if shutil.which("stockfish"):
        candidates.append(shutil.which("stockfish"))

    bin_dir = POST_TRAINING_DIR / "bin"
    candidates.extend([
        str(bin_dir / "stockfish.exe"),
        str(bin_dir / "stockfish"),
        str(REPO_ROOT / "bin" / "stockfish.exe"),
        str(REPO_ROOT / "bin" / "stockfish"),
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
    ])
    if bin_dir.exists():
        for p in bin_dir.rglob("stockfish*"):
            if p.is_file() and not p.name.endswith((".zip", ".tar")):
                candidates.append(str(p))

    for cand in candidates:
        if cand and Path(cand).is_file() and _verify_stockfish(cand):
            return str(Path(cand).resolve())

    if auto_download:
        dl = download_stockfish_binary(bin_dir)
        if dl and _verify_stockfish(str(dl)):
            (POST_TRAINING_DIR / ".stockfish_path").write_text(str(dl), encoding="utf-8")
            return str(dl)

    return None


def inspect_all(auto_download_sf=False, install_cuda_torch=False, save_json=None):
    ensure_repo_files(REPO_ROOT)
    device, _, _, gpu_info = configure_nvidia_gpu(enable_amp=True, auto_install_cuda=install_cuda_torch)

    from chess_hw import CONFIG, ChessTokenizer, Engine, ModelPlayer, load_model, validate_state_dict
    from chess_hw.checks import check_safetensors_file
    from chess_hw.match import MAX_PLIES, load_openings

    base_weights = REPO_ROOT / "base_model" / "model.safetensors"
    sf_errors = check_safetensors_file(base_weights)
    if sf_errors:
        raise RuntimeError(f"Base model check failed: {sf_errors}")

    base_sha = hashlib.sha256(base_weights.read_bytes()).hexdigest()
    model = load_model(base_weights, device=device)
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    sd_err = validate_state_dict(sd)
    if sd_err:
        raise RuntimeError(f"Invalid base state_dict: {sd_err}")

    tok = ChessTokenizer()
    openings = load_openings()
    player = ModelPlayer(model, name="base_model")

    board = chess.Board()
    start_cands = player.candidates([])
    start_choice = player.choose(board)

    op_board = chess.Board()
    for u in openings[0]:
        op_board.push_uci(u)
    op_cands = player.candidates([m.uci() for m in op_board.move_stack])
    op_choice = player.choose(op_board)

    sf_path = resolve_stockfish_path(auto_download=auto_download_sf)
    eng_move = Engine(level=2, seed=0).best_move(op_board)

    shapes = {k: list(v.shape) for k, v in sd.items()}
    n_params = model.num_parameters()

    report = {
        "space_repo": SPACE_REPO_ID,
        "gpu_info": gpu_info,
        "cpu_count": os.cpu_count(),
        "stockfish_path": sf_path,
        "base_model": {
            "path": str(base_weights),
            "sha256": base_sha,
            "num_parameters": n_params,
            "config": asdict(CONFIG),
            "num_tensors": len(shapes),
            "tensor_shapes": shapes,
            "tied_output_head": "F.linear(hidden_states, self.wte.weight)",
        },
        "tokenizer": {
            "vocab_size": tok.vocab_size,
            "special_tokens": {"PAD": tok.pad_token_id, "BOS": tok.bos_token_id, "EOS": tok.eos_token_id},
            "num_uci_moves": tok.vocab_size - tok.n_special,
        },
        "evaluation_harness": {
            "max_plies": MAX_PLIES,
            "max_attempts": 3,
            "num_openings_in_book": len(openings),
            "sample_opening_0": openings[0],
            "start_pos_top3": start_cands,
            "start_pos_chosen": start_choice.move.uci() if start_choice.move else None,
            "opening_0_top3": op_cands,
            "opening_0_chosen": op_choice.move.uci() if op_choice.move else None,
            "opening_0_chosen_attempt": op_choice.attempt,
            "engine_L2_reply": eng_move.uci() if eng_move else None,
        },
    }

    print(f"device={gpu_info['device']} ({gpu_info['gpu_name'] or 'cpu'}) | cpus={report['cpu_count']}")
    print(f"stockfish={sf_path or 'not found (fallback to internal Engine)'}")
    print(f"model=ChessGPT ({n_params:,} params, {len(shapes)} tensors) | sha256={base_sha[:16]}...")
    print(f"vocab={tok.vocab_size} | openings={len(openings):,} | start_top3={start_cands}")

    if save_json:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-stockfish", action="store_true")
    ap.add_argument("--install-cuda-torch", action="store_true")
    ap.add_argument("--out", default=str(POST_TRAINING_DIR / "logs" / "repo_inspection.json"))
    args = ap.parse_args()
    inspect_all(
        auto_download_sf=args.download_stockfish,
        install_cuda_torch=args.install_cuda_torch,
        save_json=Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    main()
