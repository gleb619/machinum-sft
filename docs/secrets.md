# Secrets in `.env`

`.env` is the only place a token lives, and it is git-ignored. Variable **names** are
plain; each secret **value** is wrapped by `msft secrets encrypt` as `enc:v1:<token>`,
so `cat .env`, a log line or a screenshot shows a name and ciphertext, not the token.

```
docs/secrets.json          # plaintext source, git-ignored, never committed
        │  msft secrets encrypt
        ▼
.env                       # NAME=enc:v1:..., mode 0600, git-ignored
        │  copied by hand to the box
        ▼
remote .env                # decrypted at CLI start-up by the code's default key
```

## Workflow

```bash
# 1. Put the plaintext object here (git-ignored):
#    docs/secrets.json  ->  {"hf_token": "hf_..."}

# 2. Encrypt every value and merge into .env. Prints names only.
msft secrets encrypt
msft secrets list                  # NAME<TAB>encrypted|plain, never a value

# 3. Copy the encrypted file to the other machine; the same code decrypts it.
scp .env box:~/WORKSPACE/machinum-sft/.env
```

Keys are upper-cased to become environment names: `hf_token` becomes `HF_TOKEN`.
`msft` loads `.env` at start-up and `os.environ.setdefault` means a variable already
exported in the shell always wins over the file.

`msft secrets encrypt` refuses to write a path git tracks or would not ignore, and
writes the file atomically with mode `0600`.

## Threat model

The default key is checked into `src/msft/secrets.py`. This is **obfuscation, not
secrecy**: anyone who can read the repo can decrypt `.env`. It prevents accidental
leakage through git, logs, chat pastes and screenshots, and makes a lost `.env` a
rotatable incident rather than a forever leak. It does not protect the value from
someone with the code.

For actual secrecy, export a private Fernet key before every command on every machine:

```bash
export MSFT_SECRET_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
```

Encrypt and load with that same `MSFT_SECRET_KEY`. Do not commit it.

## Rotation

1. Replace the value in `docs/secrets.json` (or edit `.env` by hand only if it is not a secret).
2. Re-run `msft secrets encrypt`; existing names are updated in place.
3. Copy `.env` to every machine, then revoke the old token at the provider.

`MSFT_ENV_FILE` points the loader at a different file when needed.
