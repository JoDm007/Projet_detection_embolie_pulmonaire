# backend/tests/test_security.py — bloc core/security
from backend.core.security import (hash_password, password_problem, token_digest,
                                   verify_password)


def test_hash_and_verify():
    h = hash_password("motdepasse-42")
    assert h != "motdepasse-42"
    assert verify_password(h, "motdepasse-42")
    assert not verify_password(h, "mauvais-42")


def test_unknown_user_never_verifies():
    assert not verify_password(None, "nimporte-quoi-1")


def test_password_rules():
    assert password_problem("court1") is not None
    assert password_problem("seulementdeslettres") is not None
    assert password_problem("lettres-et-42") is None


def test_token_digest_is_stable_and_not_the_token():
    assert token_digest("abc") == token_digest("abc")
    assert token_digest("abc") != "abc"
