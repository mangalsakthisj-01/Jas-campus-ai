import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
import jwt

ALGORITHM = "HS256"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return "$".join(("scrypt", str(_SCRYPT_N), str(_SCRYPT_R), str(_SCRYPT_P), base64.urlsafe_b64encode(salt).decode("ascii"), base64.urlsafe_b64encode(digest).decode("ascii")))

def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_text, digest_text = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, UnicodeError):
        return False

def create_access_token(user_id: int) -> str:
    secret = os.getenv("JAS_JWT_SECRET", "dev-only-change-me-before-deployment")
    expire = datetime.now(timezone.utc) + timedelta(hours=12)
    return jwt.encode({"sub": str(user_id), "exp": expire}, secret, algorithm=ALGORITHM)

def decode_access_token(token: str) -> int | None:
    secret = os.getenv("JAS_JWT_SECRET", "dev-only-change-me-before-deployment")
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, TypeError, ValueError):
        return None
