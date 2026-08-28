"""Password hashing.

Argon2id, the current OWASP first choice: memory-hard, so a leaked hash is
expensive to attack with GPUs in a way bcrypt no longer is.

``argon2-cffi`` is used directly rather than through passlib, which has not
had a release in years and has known incompatibilities with modern bcrypt.
"""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from ..core.logging import get_logger

logger = get_logger(__name__)

#: Library defaults follow the RFC 9106 recommended profile.
_hasher = PasswordHasher()

#: Enforced on registration and password change, not on login -- rejecting a
#: short password at login would leak that the stored one is short.
MIN_PASSWORD_LENGTH = 12


def hash_password(password: str) -> str:
    """Hash a plaintext password. The salt is generated and embedded."""
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Check a password against its hash.

    Returns False rather than raising for any failure, so callers cannot
    accidentally distinguish "wrong password" from "corrupt hash" in a way that
    reaches the user.
    """
    try:
        _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except InvalidHashError:
        logger.error("password_hash_unreadable")
        return False
    # Never let a hashing-library internal error surface as a 500 on login.
    except Exception:
        logger.exception("password_verification_failed")
        return False
    return True


def needs_rehash(password_hash: str) -> bool:
    """Whether a stored hash uses outdated parameters.

    Lets the cost factor be raised over time: on a successful login with an
    outdated hash, re-hash and store the new one.
    """
    try:
        return _hasher.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True
