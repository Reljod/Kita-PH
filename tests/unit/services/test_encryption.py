"""Tests for app.services.encryption — Fernet wrapper for client API keys."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet, InvalidToken

from app.services.encryption import decrypt_key, encrypt_key, get_fernet


class TestGetFernet:
    def test_builds_a_fernet_from_the_env_key(self, monkeypatch):
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("API_KEY_ENCRYPTION_KEY", key)
        assert isinstance(get_fernet(), Fernet)

    def test_raises_when_the_key_is_unset(self, monkeypatch):
        monkeypatch.delenv("API_KEY_ENCRYPTION_KEY", raising=False)
        with pytest.raises(ValueError, match="API_KEY_ENCRYPTION_KEY"):
            get_fernet()

    def test_raises_when_the_key_is_empty(self, monkeypatch):
        """An exported-but-blank key is the common misconfiguration, and it
        must fail loudly rather than build a broken cipher."""
        monkeypatch.setenv("API_KEY_ENCRYPTION_KEY", "")
        with pytest.raises(ValueError, match="API_KEY_ENCRYPTION_KEY"):
            get_fernet()

    def test_raises_when_the_key_is_malformed(self, monkeypatch):
        monkeypatch.setenv("API_KEY_ENCRYPTION_KEY", "not-a-valid-fernet-key")
        with pytest.raises(ValueError, match="32 url-safe base64-encoded bytes"):
            get_fernet()


class TestRoundTrip:
    @pytest.fixture(autouse=True)
    def _key(self, monkeypatch):
        monkeypatch.setenv("API_KEY_ENCRYPTION_KEY", Fernet.generate_key().decode())

    @pytest.mark.parametrize(
        "plaintext",
        [
            "6665c829e1c8b32d20123456",
            "a",
            "x" * 4096,
            "unicode-∂ƒ©-key",
            "with spaces and\nnewlines",
            "",
        ],
        ids=["objectid", "single-char", "long", "unicode", "whitespace", "empty"],
    )
    def test_decrypt_reverses_encrypt(self, plaintext):
        assert decrypt_key(encrypt_key(plaintext)) == plaintext

    def test_ciphertext_is_not_the_plaintext(self):
        assert encrypt_key("secret-value") != "secret-value"

    def test_encrypting_twice_gives_different_ciphertexts(self):
        """Fernet embeds a random IV, so identical plaintexts must not produce
        identical ciphertexts — otherwise stored keys would be comparable."""
        assert encrypt_key("same") != encrypt_key("same")

    def test_both_ciphertexts_still_decrypt_to_the_same_plaintext(self):
        assert decrypt_key(encrypt_key("same")) == decrypt_key(encrypt_key("same"))

    def test_encrypt_returns_str_not_bytes(self):
        assert isinstance(encrypt_key("x"), str)

    def test_decrypt_returns_str_not_bytes(self):
        assert isinstance(decrypt_key(encrypt_key("x")), str)


class TestDecryptFailures:
    @pytest.fixture(autouse=True)
    def _key(self, monkeypatch):
        monkeypatch.setenv("API_KEY_ENCRYPTION_KEY", Fernet.generate_key().decode())

    def test_garbage_ciphertext_raises(self):
        with pytest.raises(InvalidToken):
            decrypt_key("not-a-real-token")

    def test_empty_ciphertext_raises(self):
        with pytest.raises(InvalidToken):
            decrypt_key("")

    def test_a_key_encrypted_under_a_different_secret_cannot_be_read(self, monkeypatch):
        """Rotating API_KEY_ENCRYPTION_KEY must invalidate old ciphertexts
        rather than silently decrypt to garbage."""
        ciphertext = encrypt_key("client-api-key")
        monkeypatch.setenv("API_KEY_ENCRYPTION_KEY", Fernet.generate_key().decode())
        with pytest.raises(InvalidToken):
            decrypt_key(ciphertext)

    def test_tampered_ciphertext_is_rejected(self):
        ciphertext = encrypt_key("client-api-key")
        tampered = ciphertext[:-4] + ("AAAA" if ciphertext[-4:] != "AAAA" else "BBBB")
        with pytest.raises(InvalidToken):
            decrypt_key(tampered)
