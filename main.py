"""
Go to end for example_usage function

Requires numpy

Thomas Zarri
"""

import numpy as np
import random
import time

class PaddingError(Exception):
    def __init__(self, message=None):
        super().__init__(message or "Padding error")

INTERNAL_ORACLE_KEY = bytes.fromhex("deadbeef01234567")
INTERNAL_IV = bytes.fromhex("0000000000000000") # we pretend we know the IV as well
INTERNAL_HASH_IV = bytes.fromhex("01234567beefdead")

sbox = np.array([142, 79, 87, 120, 121, 106, 107, 92, 93, 80, 81, 114, 115, 148, 101, 134, 158, 95, 103, 136, 137, 122, 123, 108, 109, 96, 97, 130, 131, 164, 117, 150, 174, 111, 119, 152, 153, 138, 139, 124, 125, 112, 113, 146, 147, 180, 133, 166, 154, 155, 140, 141, 128, 129, 162, 163, 196, 149, 182, 190, 127, 135, 168, 169, 198, 206, 143, 151, 184, 185, 170, 171, 156, 157, 144, 145, 178, 179, 212, 165, 214, 222, 159, 167, 200, 201, 186, 187, 172, 173, 160, 161, 194, 195, 228, 181, 216, 217, 202, 203, 188, 189, 176, 177, 210, 211, 244, 197, 230, 238, 175, 183, 232, 233, 218, 219, 204, 205, 192, 193, 226, 227, 4, 213, 246, 254, 191, 199, 14, 207, 215, 248, 249, 234, 235, 220, 221, 208, 209, 242, 243, 20, 229, 6, 231, 8, 9, 250, 251, 236, 237, 224, 225, 2, 3, 36, 245, 22, 30, 223, 247, 24, 25, 10, 11, 252, 253, 240, 241, 18, 19, 52, 5, 38, 46, 239, 26, 27, 12, 13, 0, 1, 34, 35, 68, 21, 54, 62, 255, 7, 40, 41, 56, 57, 42, 43, 28, 29, 16, 17, 50, 51, 84, 37, 70, 78, 15, 23, 72, 73, 58, 59, 44, 45, 32, 33, 66, 67, 100, 53, 86, 94, 31, 39, 116, 69, 102, 110, 47, 55, 88, 89, 74, 75, 60, 61, 48, 49, 82, 83, 132, 85, 118, 126, 63, 71, 104, 105, 90, 91, 76, 77, 64, 65, 98, 99], dtype=np.uint8)

# convert string -> bytes
# convert uint64 -> bytes
# convert np array[uint64] -> bytes
# convert np array[uint8] -> bytes
# convert uint8 -> bytes

# convert bytes -> uint64
# convert bytes -> np array[uint8]
# convert bytes -> np array[uint64]

# convert np array[uint8] -> np array [uint64]
# convert np array[uint64] -> np array [uint8]

def _uint8_bytes(x : np.uint8) -> bytes:
    return np.array([x], dtype=np.uint8).tobytes()

def _uint64_bytes(x : np.uint64) -> bytes:
    return np.array([x], dtype=np.uint64).tobytes()

def _vuint8_bytes(x : np.ndarray) -> bytes:
    return x.tobytes()

def _vuint64_bytes(x : np.ndarray) -> bytes:
    return x.astype(">u8").tobytes()

def _bytes_uint64(x : bytes) -> np.uint64:
    return np.frombuffer(x, dtype=">u8")[0]

def _bytes_vuint8(x : bytes) -> np.ndarray:
    return np.frombuffer(x, dtype=np.uint8)

def _bytes_vuint64(x : bytes) -> np.ndarray:
    return np.frombuffer(x, dtype=">u8")

def _vuint8_vuint16(x : np.ndarray) -> np.ndarray:
    return x.view(">u2").copy()

def _vuint16_vuint8(x : np.ndarray) -> np.ndarray:
    return x.astype(">u2").view(np.uint8).copy()

def _vuint8_vuint64(x : np.ndarray) -> np.ndarray:
    return x.view(">u8").copy()

def _vuint8_uint64(x : np.ndarray) -> np.uint64:
    return x.view(">u8")[0]

def _vuint8_uint8(x : np.ndarray) -> np.uint8:
    return x[0]

def _vuint64_vuint8(x : np.ndarray) -> np.ndarray:
    return x.astype(">u8").view(np.uint8).copy()

def _uint8_vuint8(x : np.uint8) -> np.ndarray:
    return np.array([x], dtype=np.uint8)

def _uint64_vuint8(x : np.uint64) -> np.ndarray:
    return np.array([x], dtype=">u8").view(np.uint8).copy()

def _rotate_left64(x : np.uint64, n : int) -> np.uint64:
    x = np.uint64(x)
    l = np.uint64((x << n) & np.uint64(0xFFFF_FFFF_FFFF_FFFF))
    r = np.uint64(x >> (64 - n)) 

    return np.uint64(l | r)

def _feistel_round(v : np.uint32, round_key : np.uint32) -> np.uint32:
    b0 = sbox[(v >> 0) & 0xFF]
    b1 = sbox[(v >> 8) & 0xFF]
    b2 = sbox[(v >> 16) & 0xFF]
    b3 = sbox[(v >> 24) & 0xFF]
    t = np.uint32((b3 << 24) | (b2 << 16) | (b1 << 8) | b0)

    return np.uint32(t ^ round_key)

def _get_round_keys(key : np.uint64) -> np.ndarray:
    res = np.zeros(6, dtype=np.uint32)

    for i in range(6):
        key = _rotate_left64(key, 33)
        # TODO: first 32 bits of K_i could be most significant/least significant
        round_key = np.uint32(key >> 32)

        res[i] = round_key

    return res

def _feistel_encrypt(block : np.uint64, key : np.uint64) -> np.uint64:
    # break into blocks
    L = np.uint32(block >> 32)
    R = np.uint32(block & np.uint32((1 << 32) - 1))

    for k in _get_round_keys(key):
        # TODO: unnecessary uint32?
        L, R = R, np.uint32(L ^ _feistel_round(R, k))

    return (np.uint64(L) << 32) | np.uint64(R)

def _feistel_decrypt(block : np.uint64, key : np.uint64) -> np.uint64:
    L = np.uint32(block >> 32)
    R = np.uint32(block & np.uint32((1 << 32) - 1))

    for k in reversed(_get_round_keys(key)):
        L, R = np.uint32(R ^ _feistel_round(L, k)), L

    return (np.uint64(L) << 32) | np.uint64(R)

def _convert_np_array(data : str | bytes | np.ndarray) -> np.ndarray:
    if isinstance(data, np.ndarray) and data.dtype == np.uint8:
        return data

    if isinstance(data, np.ndarray) and data.dtype == np.uint64:
        return _vuint8_vuint64(data)

    if isinstance(data, bytes):
        return _bytes_vuint8(data)

    # TODO: look for hex string?
    if isinstance(data, str):
        return _bytes_vuint8(data.encode("utf-8"))

    raise TypeError(f"Cannot convert type {type(data)} to np.uint8 buffer, try a utf-8 string or any bitstring")

def _constant_time_check(data, val):
    result = np.uint8(0)

    # result will stay 0 as long as no difference between element and value
    for elem in data:
        result |= elem ^ val

    return result == 0

def _pad_pkcs7(data, block_size_bytes=8):
    pad_length = block_size_bytes - (len(data) % block_size_bytes)
    # array of pad length, entries pad_length
    pad = np.full(pad_length, pad_length, dtype=np.uint8)
    # TODO: not the nicest construction
    return np.concatenate([data, pad])

def _unpad_pkcs7(data, block_size_bytes=8):
    pad_length = int(data[-1])

    within_bounds = 0 < pad_length <= block_size_bytes
    padding_valid = _constant_time_check(data[-pad_length:], pad_length)

    if within_bounds and padding_valid:
        return data[:-pad_length]

    raise PaddingError()

    # invalid/fail
    return np.zeros(1, dtype=np.uint8)

def _pad_md(m : np.ndarray) -> np.ndarray:
    ml = len(m)
    ml_bytes = _uint64_vuint8(np.uint64(ml))

    pad_len = (8 - (ml + 1 + 8) % 8) % 8
    
    return np.concatenate([
        m,
        _uint8_vuint8(np.uint8(0x80)), # 1000 0000
        np.zeros(pad_len, dtype=np.uint8),
        ml_bytes
    ])

# TODO: unsure if even needed
def _unpad_md(m : np.ndarray) -> np.ndarray:
    if m.size <= 8 or arr.size % 8 != 0:
        raise ValueError("Invalid MD padded input length")

    msg_len = _vuint8_uint64(m[-8:])

    if msg_len >= arr.size - 8:
        raise ValueError("Invalid MD length failed")

    msg_end = msg_len

    if arr[msg_end] != 0x80:
        raise ValueError("Invalid MD padding")

    zeros_region = arr[msg_end + 1:-8]
    if zeros_region.size > 0 and not np.all(zeros_region == 0):
        raise ValueError("Invalid MD padding zeros")

    return arr[:msg_end].copy()

def _encrypt_ecb(plaintext : np.ndarray, key : np.uint64) -> np.ndarray:
    padded = _pad_pkcs7(plaintext, 8)
    blocks = _vuint8_vuint64(padded)
    encrypted = np.array([_feistel_encrypt(b, key) for b in blocks], dtype=np.uint64)
    ciphertext = _vuint64_vuint8(encrypted)

    return ciphertext

def _decrypt_ecb(ciphertext : np.ndarray, key : np.uint64) -> np.ndarray:
    blocks = _vuint8_vuint64(ciphertext)
    decrypted = np.array([_feistel_decrypt(b, key) for b in blocks], dtype=np.uint64)
    padded = _vuint64_vuint8(decrypted)
    plaintext = _unpad_pkcs7(padded)

    return plaintext 

def _encrypt_cbc(plaintext : np.ndarray, key : np.uint64) -> np.ndarray:
    iv = _bytes_uint64(INTERNAL_IV)
    iv_block = _bytes_vuint64(INTERNAL_IV)

    padded = _pad_pkcs7(plaintext, 8)
    blocks = _vuint8_vuint64(padded)
    num_blocks = blocks.size

    encrypted_blocks = np.empty_like(blocks, dtype=np.uint64)
    prev = iv

    for i in range(num_blocks):
        encrypted_block = _feistel_encrypt(blocks[i] ^ prev, key)
        encrypted_blocks[i] = encrypted_block
        prev = encrypted_block

    return _vuint64_vuint8(np.concatenate([iv_block, encrypted_blocks]))

def _decrypt_cbc(ciphertext : np.ndarray, key : np.uint64) -> np.ndarray:
    iv_blocks = _vuint8_vuint64(ciphertext)

    iv = iv_blocks[0]
    blocks = iv_blocks[1:].copy()

    num_blocks = blocks.size
    decrypted_blocks = np.empty_like(blocks, dtype=np.uint64)

    prev = iv

    for i in range(num_blocks):
        decrypted_block = _feistel_decrypt(blocks[i], key) ^ prev
        decrypted_blocks[i] = decrypted_block
        prev = blocks[i]

    decrypted = _vuint64_vuint8(decrypted_blocks)
    plaintext = _unpad_pkcs7(decrypted, 8)

    return plaintext

def Encrypt_ECB(plaintext : bytes, key : bytes) -> bytes:
    return _vuint8_bytes(_encrypt_ecb(_convert_np_array(plaintext), _bytes_uint64(key)))

def Decrypt_ECB(ciphertext : bytes, key : bytes) -> bytes:
    ciphertext = _convert_np_array(ciphertext)

    if len(ciphertext) % 8 != 0:
        raise ValueError("Ciphertext must be multiple of block size (8 bytes)")

    return _vuint8_bytes(_decrypt_ecb(ciphertext, _bytes_uint64(key)))

def Encrypt_CBC(plaintext : bytes, key : bytes) -> bytes:
    return _vuint8_bytes(_encrypt_cbc(_convert_np_array(plaintext), _bytes_uint64(key)))

def Decrypt_CBC(ciphertext : bytes, key : bytes) -> bytes:
    # print(f"Cipher before: {ciphertext}")
    ciphertext = _convert_np_array(ciphertext)
    # print(f"Cipher after : {_vuint8_bytes(ciphertext)}")

    if len(ciphertext) % 8 != 0:
        raise ValueError("ciphertext must be multiple of block size (8 bytes)")

    return _vuint8_bytes(_decrypt_cbc(ciphertext, _bytes_uint64(key)))

def _compression(k : np.uint64, x : np.uint64) -> np.uint64:
    e = _feistel_encrypt(x, k)
    h = e ^ x

    return h

def Compression(k : bytes, x : bytes) -> bytes:
    # TODO: assert size of both is 64? 8 bytes
    # TODO: use function or define new one
    # TODO: or use np array
    k = _bytes_uint64(k)
    x = _bytes_uint64(x)

    h = _compression(k, x)

    return _uint64_bytes(h)

def _hash(iv : np.uint64, m : np.ndarray) -> np.uint64:
    padded = _pad_md(m) 
    blocks = _vuint8_vuint64(padded)
    h = iv

    # compression takes n+n' -> n, k,
    #   we take blocks of size n'
    #   iv is of size n

    # set z_0 = IV
    #   z_i = G(z_{i-1}||x_i)
    # x are message blocks after padding
    # z are chaining variables

    # compression function of davis meyer
    # n'-bit key length
    # n-bit block length
    # G(k,m)=F_k(m) ^ m

    # k usually message block
    # x usually chain varible

    for block in blocks:
        h = _compression(block, h)

    return h

def Hash(iv : bytes, m : bytes) -> bytes:
    if len(iv) != 8:
        raise ValueError("Expected IV to be 8 bytes")

    iv = _bytes_uint64(iv)
    m = _convert_np_array(m)

    h = _hash(iv, m)

    return _uint64_bytes(h)


def MAC(k : bytes, m : bytes) -> bytes:
    m_hash = Hash(INTERNAL_HASH_IV, m)
    t = Encrypt_ECB(m_hash, k)

    return t

def Verify(k : bytes, m : bytes, t : bytes) -> bool:
    expected = MAC(k, m)

    if len(t) != len(expected): return False

    # NOTE: intentionally introducing a timing-based attack
    for i, byte in enumerate(expected):
        if byte != t[i]:
            return False

        time.sleep(0.03)

    return True

def AuthEncrypt(k : bytes, plaintext : bytes) -> bytes:
    t = MAC(k, plaintext)
    # NOTE: t is 16 bytes
    c = Encrypt_CBC(plaintext + t, k)

    return c

# NOTE: requires python 3.10+
def AuthDecrypt(k : bytes, ciphertext : bytes) -> bytes | bool:
    # print(f"Cipher: {ciphertext}")
    plaintext_tag = Decrypt_CBC(ciphertext, k)
    # print(len(plaintext_tag))
    # NOTE: tag is 16 bytes
    plaintext = plaintext_tag[:-16] 
    tag = plaintext_tag[-16:]

    if not Verify(k, plaintext, tag):
        return False

    return plaintext

def test_feistel():
    # do some testing
    count = 10
    values = [np.uint64(random.randint(0, (1 << 64) - 1)) for _ in range(count)]

    for value in values:
        key = values[random.randint(0, count - 1)]

        enc = _feistel_encrypt(value, key)
        dec = _feistel_decrypt(enc, key)

        print(value, enc, dec)

        assert(value == dec)
    
    print("Feistel test passed")

test_strings = [
    b"hello world",
    b"My name is 1-4220141 thomas",
    b"I have some \0\0\1\2\3 binary data blah",
    b"hello again",
    b"hello world"
]

def test_block_cipher(cipher_enc, cipher_dec):
    key = np.random.randint(0, 2**64, dtype=np.uint64)

    for string in test_strings:
        encrypted = cipher_enc(string, bytes.fromhex("deadbeefdeafbeed"))
        decrypted = cipher_dec(encrypted, bytes.fromhex("deadbeefdeafbeed"))

        print(_convert_np_array(string))
        print(encrypted)
        print(decrypted)

        # TODO: just check equal?
        assert(np.array_equal(_convert_np_array(string), _convert_np_array(decrypted)))

    print("Block cipher test passed")

def hash_function_runs(function):
    # TODO: test value is reasonable

    for string in test_strings:
        print(_convert_np_array(string))
        res = function(b"\0\0\0\0\0\0\0\0", string)
        print(res)

    print("Hash done")

def test_auth():
    for string in test_strings:
        print(_convert_np_array(string))
        encrypted = AuthEncrypt(bytes.fromhex("deadbeefdeadbeef"), string)
        decrypted = AuthDecrypt(bytes.fromhex("deadbeefdeadbeef"), encrypted)
        print(_convert_np_array(encrypted))

        if decrypted is False:
            raise RuntimeError("Failed!!")
        else:
            print(_convert_np_array(decrypted))

def example_usage():
    # We take any bytes-like object, so we can do bytes.from*
    # or directly take some b"Byte string",
    # or encode some regular string as "Regular string".encode("utf-8")
    key = bytes.fromhex("0000111122223333")
    plaintext = bytes.fromhex("0123456789abcdef")
    # or
    plaintext = b"Hello world"

    # plaintext, key (8-bytes)
    enc_ecb = Encrypt_ECB(plaintext, key)
    dec_ecb = Decrypt_ECB(enc_ecb, key)
    
    if dec_ecb != plaintext:
        raise RuntimeError("ECB failed!")
    
    enc_cbc = Encrypt_CBC(plaintext, key)
    dec_cbc = Decrypt_CBC(enc_cbc, key)

    if dec_cbc != plaintext:
        raise RuntimeError("CBC failed!")

    # 64 + 64 -> 64
    # key, m
    c = Compression(bytes.fromhex("0011223300112233"), bytes.fromhex("deadbeefdeadbeef"))

    # IV (8 bytes), message
    h = Hash(b"\0\0\0\0\0\0\0\0", b"Hello world please hash this message")

    # key (8 bytes), message
    tag = MAC(key, b"Message to MAC")

    # key (8 bytes), message, tag
    if not Verify(key, b"Message to MAC", tag):
        raise RuntimeError("Failed to verify MAC")

    # key, plaintext
    auth_enc = AuthEncrypt(key, plaintext)
    auth_dec = AuthDecrypt(key, auth_enc)

    if auth_dec != plaintext:
        raise RuntimeError("Auth encryption failed!")

def _attack_oracle(ciphertext : np.ndarray) -> bool:
    try:
        plaintext = _decrypt_cbc(ciphertext, _bytes_uint64(INTERNAL_ORACLE_KEY))
    except PaddingError:
        return False

    return True

def _recover_block(target_block, prev_block, oracle):
    recovered_bytes = np.zeros(8, dtype=np.uint8) # array of block_size bytes

    prev_block = _uint64_vuint8(prev_block)
    crafted_prev = prev_block.copy()

    target_block = _uint64_vuint8(target_block)

    for pos in range(7, -1, -1):
        pad_value = 8 - pos

        for j in range(pos + 1, 7 + 1):
            crafted_prev[j] = prev_block[j] ^ recovered_bytes[j] ^ pad_value

        # values for byte
        for guess in range(256):
            crafted_prev[pos] = prev_block[pos] ^ guess ^ pad_value

            chosen_ciphertext = np.concatenate([crafted_prev, target_block])

            if oracle(chosen_ciphertext):
                recovered_bytes[pos] = guess
                break

    return recovered_bytes

def PaddingOracleAttack(ciphertext : bytes) -> bytes:
    ciphertext = _convert_np_array(ciphertext)
    ciphertext_blocks = _vuint8_vuint64(ciphertext)

    plaintext_blocks = np.zeros(ciphertext_blocks.size - 1, dtype=np.uint64)
    plaintext_block_count = 0

    num_blocks = ciphertext_blocks.size
    iv_block = ciphertext_blocks[0]

    # NOTE: we only need the IV to recover the first plaintext block, but we can recover everything else
    #   without the IV

    first_block = _recover_block(ciphertext_blocks[1], iv_block, _attack_oracle)
    plaintext_blocks[plaintext_block_count] = _vuint8_uint64(first_block)
    plaintext_block_count += 1

    for i in range(2, num_blocks):
        p = _recover_block(ciphertext_blocks[i], ciphertext_blocks[i - 1], _attack_oracle)
        plaintext_blocks[plaintext_block_count] = _vuint8_uint64(p)
        plaintext_block_count += 1

    plaintext = _vuint64_vuint8(plaintext_blocks)

    return _vuint8_bytes(plaintext)

def _message_forgery_oracle_decrypt(ciphertext : bytes) -> bytes:
    return AuthDecrypt(b"secretke", ciphertext)

def _message_special_oracle(ciphertext : bytes) -> bytes:
    try:
        Decrypt_CBC(ciphertext, b"secretke")
    except PaddingError:
        return False

    return True

def _message_forgery_padding_oracle(ciphertext : np.ndarray) -> bytes:
    try:
        AuthDecrypt(b"secretke", _vuint8_bytes(ciphertext))
    except PaddingError:
        return False

    # NOTE: we don't care about the validity of the MAC for this
    return True

def _cbc_r_attempt(plaintext : np.ndarray) -> np.ndarray:
    plaintext = _vuint8_vuint64(plaintext)
    n = plaintext.size
    # NOTE: the first block is random bytes (or all 0s for us)
    # so we cannot recover everything
    # but if we control IV we can maybe?
    ciphertext_blocks = np.zeros(n + 1, dtype=np.uint64)
    # ciphertext_blocks[ciphertext_blocks.size - 1] = _bytes_uint64(INTERNAL_IV)
    
    for i in range(n, 0, -1):
        # we differ from the paper here because recover block returns plaintext
        # so we take the xor with the previous ciphertext block to get back to intermediate value
        oracle_res = _vuint8_uint64(_recover_block(ciphertext_blocks[i], ciphertext_blocks[i - 1], _message_forgery_padding_oracle))
        # NOTE: might need to be plaintext[i]?
        ciphertext_blocks[i - 1] = plaintext[i - 1] ^ oracle_res

    return _vuint64_vuint8(ciphertext_blocks)

def MessageForgeryAttack2() -> bytes:
    guessed_tag = np.zeros(16, dtype=np.uint8)
    
    # ASsume we have some forgery already
    forged = b"forged_message_32_bytes_aaabbbcc"
    actual_tag = _bytes_vuint8(MAC(b"secretke", forged))
    plaintext_tag = np.concatenate([_bytes_vuint8(forged), actual_tag])
    ciphertext = _cbc_r_attempt(_pad_pkcs7(plaintext_tag))

    print(_vuint8_bytes(ciphertext).hex())
    plaintext_pray = Decrypt_CBC(_vuint8_bytes(ciphertext), b"secretke")
    print(f"Outplaint: {plaintext_pray.hex()}")
    print(f"Plaintext: {forged.hex()}")

    res = AuthDecrypt(b"secretke", ciphertext)

    print(res)

    return ciphertext

def MessageForgeryAttack() -> bytes:
    forged = b"forged_message_32_bytes_aaabbbcc"
    guessed_tag = np.zeros(16, dtype=np.uint8)
    actual_tag = _bytes_vuint8(MAC(b"secretke", forged))

    correct, fail_byte = Verify(b"secretke", forged, actual_tag)
    print(f"Should be correct; {correct}")

    for i in range(16):
        times_taken = np.zeros(256, dtype=np.float64)

        # Warm it up first
        for guess in range(20):
            guessed_tag[i] = guess

            Verify(b"secretke", forged, _vuint8_bytes(guessed_tag))

        for guess in range(256):
            guessed_tag[i] = guess

            start_time = time.time_ns()
            correct = Verify(b"secretke", forged, _vuint8_bytes(guessed_tag))
            elapsed = time.time_ns() - start_time

            times_taken[guess] = elapsed

        likely_correct = np.argmax(times_taken)

        guessed_tag[i] = likely_correct

        if guessed_tag[i] != actual_tag[i]:
            print(f"Made a mistake, idx {i}, real value was {actual_tag[i]} with timing {times_taken[actual_tag[i]]}, but got instead {guessed_tag[i]} timing {times_taken[guessed_tag[i]]}")

            return b""

        print(f"Guessed byte idx:{i}, to be {likely_correct}")

    was_correct = Verify(b"secretke", forged, guessed_tag)

    print(f"Was correct: {was_correct}")

if __name__ == "__main__":
    # example_usage()
    # test_block_cipher(Encrypt_CBC, Decrypt_CBC)
    # test_block_cipher(Encrypt_ECB, Decrypt_ECB)
    # hash_function_runs(Hash)
    # test_auth()

    # res = Encrypt_ECB(bytes.fromhex("0123456789abcdef"), bytes.fromhex("0000000000000000")) 
    # print(res.hex())

    MessageForgeryAttack2()

    # Padding oracle attack that retrieves plaintext
    ciphertext = Encrypt_CBC(bytes.fromhex("00112233aabbccddeeffaabbccddeeff"), INTERNAL_ORACLE_KEY)
    res = PaddingOracleAttack(ciphertext)
    print(f"{res=}, {res.hex()}")

    # MessageForgeryAttack()
    # MessageForgeryAttack2()

    # example_usage()
