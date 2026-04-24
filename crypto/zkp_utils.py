import hashlib
import secrets
import time


# Large prime group parameters
P = 208351617316091241234326746312124448251235562226470491514186331217050270460481
G = 2


def _hash_to_int(*values) -> int:
    h = hashlib.sha256()
    for v in values:
        if isinstance(v, int):
            h.update(str(v).encode())
        elif isinstance(v, bytes):
            h.update(v)
        else:
            h.update(str(v).encode())
    return int.from_bytes(h.digest(), "big") % (P - 1)


def keygen():
    start = time.time()
    secret_key = secrets.randbelow(P - 2) + 1
    public_key = pow(G, secret_key, P)
    keygen_ms = (time.time() - start) * 1000
    return public_key, secret_key, keygen_ms


def generate_proof(secret_key: int, update_bytes: bytes, client_id: str):
    start = time.time()

    update_hash = hashlib.sha256(update_bytes).hexdigest()

    r = secrets.randbelow(P - 2) + 1
    commitment = pow(G, r, P)

    challenge = _hash_to_int(client_id, update_hash, commitment)

    response = (r + challenge * secret_key) % (P - 1)

    proof_ms = (time.time() - start) * 1000

    return {
        "update_hash": update_hash,
        "commitment": commitment,
        "challenge": challenge,
        "response": response,
        "proof_ms": proof_ms,
    }


def verify_proof(public_key: int, update_bytes: bytes, client_id: str, proof: dict):
    start = time.time()

    expected_hash = hashlib.sha256(update_bytes).hexdigest()

    if proof["update_hash"] != expected_hash:
        return False, 0.0

    expected_challenge = _hash_to_int(
        client_id,
        proof["update_hash"],
        proof["commitment"],
    )

    if proof["challenge"] != expected_challenge:
        return False, 0.0

    left = pow(G, proof["response"], P)
    right = (proof["commitment"] * pow(public_key, proof["challenge"], P)) % P

    verify_ms = (time.time() - start) * 1000

    return left == right, verify_ms