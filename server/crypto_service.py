import os
import struct
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Load keys from environment or use secure prototype defaults
AES_SECRET_HEX = os.getenv(
    "AES_SECRET_KEY", 
    "94c8e763a8a3a31e2474db62c82e0fb58cc2a77ef7cb73f1d8c117b4abdc3d9d"
)
DEVICE_SECRET_RAW = os.getenv(
    "DEVICE_SECRET_KEY",
    "d3v1c3_s3cr3t_ke"  # Must be exactly 16 bytes
)

# Initialize AES-GCM
AES_KEY_BYTES = bytes.fromhex(AES_SECRET_HEX)
aesgcm = AESGCM(AES_KEY_BYTES)

# Normalize device key to 16 bytes
XTEA_KEY_BYTES = DEVICE_SECRET_RAW.encode("utf-8")[:16].ljust(16, b"\x00")

# =====================================================================
# AES-GCM (Browser -> Server)
# =====================================================================

def decrypt_aes_gcm_vector(ciphertext_b64: str, iv_b64: str) -> list[float]:
    """
    Decrypt an AES-GCM encrypted 128D float32 vector from the browser.
    Ciphertext should contain the 16-byte authentication tag at the end (standard Web Crypto GCM output).
    """
    try:
        ciphertext = base64.b64decode(ciphertext_b64)
        iv = base64.b64decode(iv_b64)
        
        # Decrypt using cryptography's AESGCM (expects tag appended to ciphertext)
        decrypted_bytes = aesgcm.decrypt(iv, ciphertext, None)
        
        if len(decrypted_bytes) != 512:
            raise ValueError(f"Decrypted payload length is {len(decrypted_bytes)} bytes, expected 512 bytes (128 floats)")
            
        # Unpack from little-endian float32 array
        floats = list(struct.unpack("<128f", decrypted_bytes))
        return floats
    except Exception as e:
        raise ValueError(f"AES-GCM Decryption failed: {str(e)}")

# =====================================================================
# XTEA-CTR (Server <-> Edge)
# =====================================================================

def _xtea_encrypt_block(key: tuple[int, int, int, int], block: tuple[int, int]) -> tuple[int, int]:
    y, z = block
    sum_val = 0
    delta = 0x9E3779B9
    mask = 0xFFFFFFFF
    for _ in range(32):
        y = (y + (((z << 4 ^ z >> 5) + z) ^ (sum_val + key[sum_val & 3]))) & mask
        sum_val = (sum_val + delta) & mask
        z = (z + (((y << 4 ^ y >> 5) + y) ^ (sum_val + key[(sum_val >> 11) & 3]))) & mask
    return y, z

def xtea_crypt_ctr(key_bytes: bytes, nonce_bytes: bytes, data_bytes: bytes) -> bytes:
    """
    Encrypt/Decrypt data using XTEA in CTR mode.
    Pure-Python implementation with no external dependencies.
    """
    key = struct.unpack(">4I", key_bytes)
    nonce = struct.unpack(">2I", nonce_bytes)
    
    out = bytearray()
    num_blocks = (len(data_bytes) + 7) // 8
    
    for i in range(num_blocks):
        # Generate block counter: increment nonce[1]
        ctr_block = (nonce[0], (nonce[1] + i) & 0xFFFFFFFF)
        # Encrypt the counter block to get keystream bytes
        keystream_block = _xtea_encrypt_block(key, ctr_block)
        keystream_bytes = struct.pack(">2I", *keystream_block)
        
        # XOR keystream with data
        chunk = data_bytes[i*8 : (i+1)*8]
        for b, k in zip(chunk, keystream_bytes):
            out.append(b ^ k)
            
    return bytes(out)

def encrypt_xtea_vector(vector: list[float]) -> tuple[str, str]:
    """
    Encrypt a 128D float32 vector using XTEA-CTR.
    Returns (ciphertext_b64, nonce_b64)
    """
    try:
        # Pack to little-endian float32 bytes
        data_bytes = struct.pack("<128f", *vector)
        # Generate random 8-byte nonce
        nonce_bytes = os.urandom(8)
        # Encrypt
        encrypted_bytes = xtea_crypt_ctr(XTEA_KEY_BYTES, nonce_bytes, data_bytes)
        
        ciphertext_b64 = base64.b64encode(encrypted_bytes).decode("utf-8")
        nonce_b64 = base64.b64encode(nonce_bytes).decode("utf-8")
        return ciphertext_b64, nonce_b64
    except Exception as e:
        raise ValueError(f"XTEA-CTR Encryption failed: {str(e)}")

def decrypt_xtea_vector(ciphertext_b64: str, nonce_b64: str) -> list[float]:
    """
    Decrypt an XTEA-CTR encrypted 128D float32 vector.
    Returns list of 128 floats.
    """
    try:
        ciphertext = base64.b64decode(ciphertext_b64)
        nonce = base64.b64decode(nonce_b64)
        
        # Decrypt
        decrypted_bytes = xtea_crypt_ctr(XTEA_KEY_BYTES, nonce, ciphertext)
        
        if len(decrypted_bytes) != 512:
            raise ValueError(f"Decrypted payload length is {len(decrypted_bytes)} bytes, expected 512 bytes")
            
        # Unpack from little-endian float32 array
        floats = list(struct.unpack("<128f", decrypted_bytes))
        return floats
    except Exception as e:
        raise ValueError(f"XTEA-CTR Decryption failed: {str(e)}")
