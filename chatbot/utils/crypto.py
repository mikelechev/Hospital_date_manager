import os
import json
import base64
from pathlib import Path
from typing import Dict

try:
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
        backend=default_backend(),
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def encrypt_dict(password: str, data: Dict[str, str]) -> Dict[str, str]:
    if not HAS_CRYPTO:
        raise RuntimeError("cryptography library not available")
    salt = os.urandom(16)
    key = _derive_key(password, salt)
    f = Fernet(key)
    token = f.encrypt(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    return {
        "salt": base64.b64encode(salt).decode(),
        "token": base64.b64encode(token).decode(),
    }


def decrypt_file(password: str, filepath: str) -> Dict[str, str]:
    if not HAS_CRYPTO:
        raise RuntimeError("cryptography library not available")
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(str(p))
    j = json.loads(p.read_text(encoding="utf-8"))
    salt_b64 = j.get("salt")
    token_b64 = j.get("token")
    if not salt_b64 or not token_b64:
        raise ValueError("Invalid encrypted file format")
    salt = base64.b64decode(salt_b64)
    token = base64.b64decode(token_b64)
    key = _derive_key(password, salt)
    f = Fernet(key)
    try:
        decrypted = f.decrypt(token)
        return json.loads(decrypted.decode("utf-8"))
    except Exception as e:
        raise RuntimeError("Decryption failed or wrong passphrase") from e
