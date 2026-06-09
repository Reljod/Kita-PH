import os
from cryptography.fernet import Fernet

def get_fernet() -> Fernet:
    secret_key = os.getenv("API_KEY_ENCRYPTION_KEY")
    if not secret_key:
        raise ValueError("API_KEY_ENCRYPTION_KEY environment variable is not set")
    return Fernet(secret_key.encode())

def encrypt_key(plain_key: str) -> str:
    """Encrypt a plaintext string using the secret key."""
    f = get_fernet()
    return f.encrypt(plain_key.encode('utf-8')).decode('utf-8')

def decrypt_key(encrypted_key: str) -> str:
    """Decrypt a ciphertext string back to plaintext."""
    f = get_fernet()
    return f.decrypt(encrypted_key.encode('utf-8')).decode('utf-8')
