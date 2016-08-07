"""
Solutions to the Cryptopals Crypto Challenge

All crypto functions take and return numpy arrays of uint8; convenience
functions are provided to convert to and from this format.

You will need the datafiles from the challenges to run the tests.
"""

from base64 import b64encode as base64_encode
from base64 import b64decode as base64_decode
from base64 import b16decode

import itertools
from functools import partial
from collections import defaultdict

import numpy as np
from Crypto.Cipher import AES

import pytest


np.set_printoptions(formatter={'int': hex})
skip = pytest.mark.skip


# # # Utilities # # #


hex_decode = partial(b16decode, casefold=True)


def base64_from_hex(hex_str):
    """Set 1 - Challenge 1"""
    return base64_encode(hex_decode(hex_str))


def array_from_hex(hex_str):
    return np.frombuffer(hex_decode(hex_str), dtype=np.uint8)

afh = array_from_hex


def hex_from_array(arr):
    return ''.join(hex(v)[2:] for v in arr)

hfa = hex_from_array


def bytes_from_array(arr):
    return arr.tobytes()

bfa = bytes_from_array


def array_from_bytes(s):
    return np.frombuffer(s, dtype=np.uint8)

afb = array_from_bytes


def line_array_from_hex_file(path):
    """Returns a (potentially) ragged array of arrays."""
    lines = []
    with open(path) as fh:
        for line in fh:
            lines.append(array_from_hex(line.strip()))
    return np.array(lines)


def array_from_base64(s):
    data = base64_decode(s)
    return np.frombuffer(data, np.uint8)

afb64 = array_from_base64


def hamming_distance(d0, d1):
    return np.unpackbits(d0 ^ d1).sum()


# # # Tests for  Utilities # # #

def test_base64_from_hex():
    hex_data = b"49276d206b696c6c696e6720796f757220627261696e206c696b65206120706f69736f6e6f7573206d757368726f6f6d"
    base64_result = b"SSdtIGtpbGxpbmcgeW91ciBicmFpbiBsaWtlIGEgcG9pc29ub3VzIG11c2hyb29t"
    assert base64_from_hex(hex_data) == base64_result


def test_array_from_hex():
    hex_data = b"4927abcd"
    expected = np.array([0x49, 0x27, 0xab, 0xcd], dtype=np.uint8)
    result = array_from_hex(hex_data)
    assert np.all(result == expected)


def test_hex_from_array():
    data = np.array([0x49, 0x27, 0xab, 0xcd], dtype=np.uint8)
    expected = "4927abcd"
    result = hex_from_array(data)
    assert result == expected


def test_bytes_from_array():
    data = np.array([104, 101, 108, 108, 111], dtype=np.uint8)
    expected = b'hello'
    assert bytes_from_array(data) == expected


def test_array_from_bytes():
    data = b'hello'
    expected = np.array([104, 101, 108, 108, 111], dtype=np.uint8)
    assert np.all(array_from_bytes(data) == expected)


def test_hamming_distance():
    s0 = b"this is a test"
    s1 = b"wokka wokka!!!"
    assert hamming_distance(afb(s0), afb(s1)) == 37


# # # Crypto # # #


letters = map(ord, 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
letter_probabilities = \
       [0.0651738, 0.0124248, 0.0217339, 0.0349835, 0.1041442, 0.0197881,
        0.0158610, 0.0492888, 0.0558094, 0.0009033, 0.0050529, 0.0331490,
        0.0202124, 0.0564513, 0.0596302, 0.0137645, 0.0008606, 0.0497563,
        0.0515760, 0.0729357, 0.0225134, 0.0082903, 0.0171272, 0.0013692,
        0.0145984, 0.0007836,
        0.1918182,
        0.0651738, 0.0124248, 0.0217339, 0.0349835, 0.1041442, 0.0197881,
        0.0158610, 0.0492888, 0.0558094, 0.0009033, 0.0050529, 0.0331490,
        0.0202124, 0.0564513, 0.0596302, 0.0137645, 0.0008606, 0.0497563,
        0.0515760, 0.0729357, 0.0225134, 0.0082903, 0.0171272, 0.0013692,
        0.0145984, 0.0007836]

probability_from_char = defaultdict(float, zip(letters, letter_probabilities))


def _score_char(c):
    return probability_from_char[c]

score_char = np.vectorize(_score_char)


def decrypt_single_byte_xor(plaintext, return_score=False):
    """Set 1 - Challenge 3

    Discover the single-byte key and return plaintext.

    If `return_score` is True, also return a 2-element array where element 0
    is the message and element 1 is the score.
    """
    data = plaintext.reshape(1, -1)
    keys = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    messages = data ^ keys
    scores = score_char(messages)
    message_scores = scores.sum(axis=1)
    best_message_index = message_scores.argmax()
    best_message = messages[best_message_index]
    best_message_score = message_scores[best_message_index]
    if return_score:
        return np.array([best_message, best_message_score])
    else:
        return best_message


def detect_single_byte_xor(ciphertext_lines):
    """Set 1 - Challenge 4

    ciphertext_lines: ragged array returned from line_array_from_hex_file
    """
    messages = np.array([decrypt_single_byte_xor(line, return_score=True)
                         for line in ciphertext_lines])
    best_idx = messages[:, 1].argmax()
    return messages[best_idx][0]


def encrypt_repeating_key_xor(data, key):
    """Set 1 - Challenge 5"""
    key_arr = np.array(list(itertools.islice(itertools.cycle(key), len(data))))
    return data ^ key_arr


def normalized_hamming(data, keysize):
    """Hamming distance divided by keysize"""
    h0 = hamming_distance(data[0*keysize:3*keysize], data[3*keysize:6*keysize])
    return h0 / keysize


def find_likely_keysizes(data):
    """Returns a sorted list of (keysize, score), sorted by score"""
    keysizes = range(2, 41)
    norm_distances = []
    for keysize in keysizes:
        norm_distances.append(normalized_hamming(data, keysize))
    size_and_score = list(zip(keysizes, norm_distances))
    return sorted(size_and_score, key=lambda ss: ss[1])


def _decrypt_repeating_key_xor(data):
    keysizes = find_likely_keysizes(data)
    for keysize in keysizes:
        pad_remainder = len(data) % keysize[0]
        pad_len = keysize[0] - pad_remainder
        padded_data = np.pad(data, (0, pad_len), mode='constant')
        padded_data.shape = (-1, keysize[0])
        decrypted = np.empty_like(padded_data)
        for col in range(padded_data.shape[1]):
            decrypted[:, col] = decrypt_single_byte_xor(padded_data[:, col])[0]
        decrypted.shape = (-1,)
        if pad_len > 0:
            decrypted = decrypted[:-pad_len]
        yield decrypted


def decrypt_repeating_key_xor(data):
    """Set 1 - Challenge 6"""
    candidates = _decrypt_repeating_key_xor(data)
    return next(candidates)


def decrypt_aes_ecb(data, key=afb(b'YELLOW SUBMARINE'), blocksize=16):
    """Set 1 - Challenge 7"""
    data = pkcs7(data, blocksize)
    key = pkcs7(key, blocksize)
    decrypter = AES.new(key, AES.MODE_ECB)
    return np.frombuffer(decrypter.decrypt(data), dtype=np.uint8)


def detect_aes_ecb(data):
    """Set 1 - Challenge 8

    Returns index of AES ECB encoded row.
    """
    row_scores = []
    for i, row in enumerate(data):
        blocks = row.view(dtype=np.dtype([('data', (np.uint8, 16))]))
        counts = np.unique(blocks, return_counts=True)[1]
        most_repetition = counts.max()
        row_scores.append((i, most_repetition))
    return max(row_scores, key=lambda index_count: index_count[1])


def pkcs7(data, blocksize=16, return_len=False):
    """Set 1 - Challenge 9

    Pad an array to `blocksize` with a constant value: the number of bytes
    needed to complete the last block.

    `return_len`, if set to True, will also return the pad value used.
    """
    pad_remainder = data.size % blocksize
    pad_len = (blocksize - pad_remainder) % blocksize
    padded_data = np.pad(data, (0, pad_len), mode='constant',
                         constant_values=pad_len)
    if return_len:
        return pad_len, padded_data
    else:
        return padded_data


def encrypt_aes_ecb(data, key):
    """Set 2 - Challenge 10"""
    data = pkcs7(data, 16)
    key = pkcs7(key, 16)
    encrypter = AES.new(key, AES.MODE_ECB)
    return np.frombuffer(encrypter.encrypt(data), dtype=np.uint8)


def encrypt_aes_cbc(plaintext, key, iv, blocksize=16):
    plain = pkcs7(plaintext, blocksize=blocksize)
    plain.shape = (-1, blocksize)
    cipher = np.empty_like(plain)
    for i, _ in enumerate(cipher):
        if i == 0:
            cipher[i] = encrypt_aes_ecb(plain[i] ^ iv, key=key)
        else:
            cipher[i] = encrypt_aes_ecb(plain[i] ^ cipher[i-1], key=key)
    cipher.shape = (-1,)
    return cipher


def decrypt_aes_cbc_serial(ciphertext, key, iv, blocksize=16):
    """Set 2 - Challenge 10"""
    cipher = pkcs7(ciphertext, blocksize=blocksize)
    cipher.shape = (-1, blocksize)
    plain = np.empty_like(cipher)
    for i, _ in enumerate(cipher):
        if i == 0:
            plain[i] = decrypt_aes_ecb(cipher[i], key=key) ^ iv
        else:
            plain[i] = decrypt_aes_ecb(cipher[i], key=key) ^ cipher[i-1]
    plain.shape = (-1,)
    return plain


def decrypt_aes_cbc(ciphertext, key, iv, blocksize=16):
    """Set 2 - Challenge 10

    Vectorized.
    """
    # decrypt
    cipher = pkcs7(ciphertext, blocksize=blocksize)
    plain = afb(decrypt_aes_ecb(cipher, key=key, blocksize=blocksize))

    # XOR plaintext blocks with previous ciphertext blocks
    # (iv for 0th block)
    cipher.shape = (-1, blocksize)
    plain.shape = (-1, blocksize)
    plain = plain ^ np.vstack((iv, cipher[:-1]))

    plain.shape = (-1,)
    return plain


def random_aes_key(blocksize=16):
    """Set 2 - Challenge 11"""
    return afb(np.random.bytes(blocksize))


def encryption_oracle(plaintext, blocksize=16):
    """Set 2 - Challenge 11

    Encrypt data using a random key, with random padding, in ECB or CBC mode
    (randomly).
    """
    left_pad, right_pad = np.random.randint(0, 5, 2)
    padded = np.pad(plaintext, (left_pad, right_pad), mode='constant')

    key = random_aes_key(blocksize=blocksize)

    encryption_modes = ['ECB', 'CBC']
    mode = np.random.choice(encryption_modes)

    if mode == 'ECB':
        cipher = encrypt_aes_ecb(pkcs7(padded, blocksize=blocksize),
                                 key=key,
                                 blocksize=blocksize)
    elif mode == 'CBC':
        cipher = encrypt_aes_cbc(pkcs7(padded, blocksize=blocksize),
                                 key=key,
                                 iv=random_aes_key(blocksize=blocksize),
                                 blocksize=blocksize)
    else:
        assert False, 'Unreachable state'

    return cipher


# # # Tests for Crypto # # #


def test_xor():
    """Test - Set 1 - Challenge 2

    XOR is a builtin for numpy arrays.
    """
    data = afh(b"1c0111001f010100061a024b53535009181c")
    key = afh(b"686974207468652062756c6c277320657965")
    expected = afh(b"746865206b696420646f6e277420706c6179")
    result = data ^ key
    assert np.all(expected == result)


def test_decrypt_single_byte_xor():
    hex_data = b'1b37373331363f78151b7f2b783431333d78397828372d363c78373e783a393b3736'
    plaintext = bfa(decrypt_single_byte_xor(afh(hex_data)))
    assert plaintext == b"Cooking MC's like a pound of bacon"


def test_detect_single_byte_xor():
    ciphertext = line_array_from_hex_file('./4.txt')
    message = detect_single_byte_xor(ciphertext)
    assert bfa(message) == b'Now that the party is jumping\n'


def test_encrypt_repeating_key_xor():
    test_data = afb(b"Burning 'em, if you ain't quick and nimble\nI go crazy when I hear a cymbal")
    key = afb(b"ICE")
    expected = afh(b"0b3637272a2b2e63622c2e69692a23693a2a3c6324202d623d63343c2a26226324272765272a282b2f20430a652e2c652a3124333a653e2b2027630c692b20283165286326302e27282f")
    assert np.all(encrypt_repeating_key_xor(test_data, key) == expected)


def test_decrypt_aes_ecb():
    key = afb(b'YELLOW SUBMARINE')
    with open('7.txt') as fh:
        data = base64_decode(fh.read())
        cipher_data = np.frombuffer(data, np.uint8)
        plain_data = decrypt_aes_ecb(cipher_data, key=key)
        print(plain_data)


def test_pkcs7():
    data = afb(b"YELLOW SUBMARINE")
    assert bfa(pkcs7(data, 20)) == b"YELLOW SUBMARINE\x04\x04\x04\x04"

    data = afb(b"YELLOW SUBMARINE")
    assert bfa(pkcs7(data, 16)) == b"YELLOW SUBMARINE"

    data = afb(b"BLUE SUBMARINE")
    assert bfa(pkcs7(data, 15)) == b"BLUE SUBMARINE\x01"


def test_encrypt_aes_ecb():
    data = afb(b'MORE PYTHONS')
    key = afb(b'YELLOW SUBMARINE')
    encrypted = encrypt_aes_ecb(data, key=key)
    decrypted = decrypt_aes_ecb(encrypted, key=key)
    assert np.all(pkcs7(data, 16) == decrypted)


def test_decrypt_aes_cbc():
    """Test - Set 2 - Challenge 10"""
    key = afb(b"YELLOW SUBMARINE")
    iv = np.zeros(16, dtype=np.uint8)
    with open('10.txt') as fh:
        ciphertext = afb64(fh.read())
    plaintext = decrypt_aes_cbc(ciphertext, key, iv)
    print(bfa(plaintext))


def test_aes_cbc_round_trip():
    key = afb(b"YELLOW SUBMARINE")
    iv = np.zeros(16, dtype=np.uint8)
    plaintext = afb(b"I was raised by a cup of coffee!")
    ciphertext = encrypt_aes_cbc(plaintext, key, iv)
    result = decrypt_aes_cbc(ciphertext, key, iv)
    assert np.all(plaintext == result)


def test_random_aes_key():
    key = random_aes_key()
    assert key.size == 16
    assert key.dtype == np.uint8


def test_encryption_oracle():
    plaintext = afb(b"I was raised by a cup of coffee")
    ciphertext = encryption_oracle(plaintext)
    assert plaintext.size + 10 <= ciphertext.size <= plaintext.size + 20
