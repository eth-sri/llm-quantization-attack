import cryptography

# Generating an RSA private key with custom keysize
def custom_private_key(size):
    private_key = cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key(
        public_exponent=65537,
        key_size=size,
        backend=cryptography.hazmat.backends.default_backend()
    )

    return private_key

