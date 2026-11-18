"""Submission encryption (hybrid RSA-OAEP + AES-256-GCM).

Students encrypt their submission with the instructor's PUBLIC key before
uploading, so nobody but the instructor can read uploaded weights, even if
the Hub repo is visible to other students.

The ciphertext is bound to the repository id (AES-GCM associated data): a
blob copied into another student's repo fails to decrypt.
"""

from __future__ import annotations

import io
import json
import os
import struct
import zipfile
from pathlib import Path
from typing import Dict

PUBLIC_KEY_FILE = Path(__file__).resolve().parent / "instructor_public.pem"
MAGIC = b"CHW1"
PAYLOAD_FILES = ("model.safetensors", "WRITEUP.md", "manifest.json")
MAX_PAYLOAD = 32 * 1024 * 1024


def pack(files: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in PAYLOAD_FILES:
            z.writestr(name, files[name])
    return buf.getvalue()


def unpack(data: bytes) -> Dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = set(z.namelist())
        if names != set(PAYLOAD_FILES):
            raise ValueError(f"unexpected payload content: {sorted(names)}")
        if sum(i.file_size for i in z.infolist()) > MAX_PAYLOAD:
            raise ValueError("payload too large")
        return {n: z.read(n) for n in PAYLOAD_FILES}


def encrypt(plaintext: bytes, repo_id: str, public_key_pem: bytes | None = None) -> bytes:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    pub = serialization.load_pem_public_key(public_key_pem or PUBLIC_KEY_FILE.read_bytes())
    key, nonce = AESGCM.generate_key(256), os.urandom(12)
    wrapped = pub.encrypt(key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    ct = AESGCM(key).encrypt(nonce, plaintext, repo_id.encode())
    return MAGIC + struct.pack(">I", len(wrapped)) + wrapped + nonce + ct


def decrypt(blob: bytes, repo_id: str, private_key_pem: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if blob[:4] != MAGIC:
        raise ValueError("not a submission blob")
    n = struct.unpack(">I", blob[4:8])[0]
    wrapped, nonce, ct = blob[8:8 + n], blob[8 + n:20 + n], blob[20 + n:]
    priv = serialization.load_pem_private_key(private_key_pem, password=None)
    key = priv.decrypt(wrapped, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    return AESGCM(key).decrypt(nonce, ct, repo_id.encode())  # raises InvalidTag if repo_id differs


def manifest(username: str, repo_id: str, weights_sha256: str, extra: dict | None = None) -> bytes:
    import datetime

    m = {"username": username, "repo_id": repo_id, "weights_sha256": weights_sha256,
         "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **(extra or {})}
    return json.dumps(m, indent=2).encode()
