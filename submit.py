#!/usr/bin/env python3
"""Submit your post-trained model.

    uv run hf auth login             # once (or export HF_TOKEN=...)
    uv run submit.py --model my_model/ --writeup WRITEUP.md

What it does:
  1. checks that my_model/model.safetensors has exactly the reference
     architecture (same tensor names / shapes / float32) and is not the
     unchanged base model,
  2. checks that your write-up exists,
  3. zips {model.safetensors, WRITEUP.md, manifest.json}, ENCRYPTS it with
     the instructor's public key (only the instructor can decrypt it), and
  4. uploads `submission.enc` to https://huggingface.co/<ORG>/chess-<your-hf-username>.

You can submit as many times as you want; each upload is a new commit and
the last commit before the deadline is graded.
Only the weights are submitted: the evaluation always uses the reference
model.py / tokenizer.py / play.py, so changes to those files have no effect.
"""

import argparse
import hashlib
import sys
from pathlib import Path

from chess_hw.checks import check_safetensors_file
from chess_hw.crypto import encrypt, manifest, pack

ORG = "llm-course-2026"
BASE_SHA256_FILE = Path(__file__).resolve().parent / "base_model" / "model.safetensors"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="directory containing model.safetensors (or the file)")
    ap.add_argument("--writeup", required=True, help="markdown file describing your method (see TEMPLATE_WRITEUP.md)")
    ap.add_argument("--org", default=ORG)
    ap.add_argument("--dry_run", action="store_true", help="run all checks and build submission.enc, do not upload")
    args = ap.parse_args()

    weights = Path(args.model)
    if weights.is_dir():
        weights = weights / "model.safetensors"
    print(f"[1/4] checking {weights}")
    if not weights.exists():
        sys.exit(f"  {weights} not found")
    errors = check_safetensors_file(weights)
    if errors:
        sys.exit("  INVALID weights:\n    " + "\n    ".join(errors))
    digest = sha256(weights)
    if BASE_SHA256_FILE.exists() and digest == sha256(BASE_SHA256_FILE):
        sys.exit("  these are the unchanged base weights: nothing to submit")
    print(f"  ok (sha256 {digest[:16]}...)")

    print(f"[2/4] checking write-up {args.writeup}")
    writeup = Path(args.writeup)
    if not writeup.exists() or len(writeup.read_text().split()) < 100:
        sys.exit("  write-up missing or shorter than 100 words")
    print("  ok")

    print("[3/4] Hugging Face login")
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        username = api.whoami()["name"]
    except Exception:
        if not args.dry_run:
            sys.exit("  not logged in: run `uv run hf auth login` (token with write access) or set HF_TOKEN")
        username = "dry-run-user"
        print("  not logged in (fine for a dry run)")
    repo_id = f"{args.org}/chess-{username}"
    print(f"  {username} -> {repo_id}")

    payload = pack({
        "model.safetensors": weights.read_bytes(),
        "WRITEUP.md": writeup.read_bytes(),
        "manifest.json": manifest(username, repo_id, digest),
    })
    blob = encrypt(payload, repo_id)
    out = weights.parent / "submission.enc"
    out.write_bytes(blob)
    card = (f"---\ntags: [chess, llm-course-2026]\n---\n# chess-{username}\n\n"
            f"Encrypted submission to the LLM course 2026 chess post-training homework.\n\n"
            f"- submitted by: {username}\n- weights sha256: `{digest}`\n")

    if args.dry_run:
        print(f"[4/4] dry run: wrote {out}, nothing uploaded")
        return
    print("[4/4] uploading")
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=blob, path_in_repo="submission.enc", repo_id=repo_id,
                    commit_message=f"submission {digest[:12]}")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id,
                    commit_message="model card")
    commit = api.list_repo_commits(repo_id)[0]
    print(f"\nDone: https://huggingface.co/{repo_id}\ncommit {commit.commit_id} at {commit.created_at}")
    print("Keep this commit id: it identifies your submission.")


if __name__ == "__main__":
    main()
