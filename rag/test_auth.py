"""The invite-code registration gate — the one piece of custom auth logic. The rest
(JWT login, current_active_user gating) is stock fastapi-users. No DB needed: a wrong
code is rejected before create() ever touches the database."""

import asyncio

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from rag.auth import SIGNUP_INVITE_CODE, UserCreate, UserManager


def test_invite_code_required():
    with pytest.raises(ValidationError):
        UserCreate(email="a@b.com", password="pw")


def test_wrong_invite_code_rejected():
    manager = UserManager(None)  # DB never reached: the code check raises first
    with pytest.raises(HTTPException) as exc:
        asyncio.run(manager.create(UserCreate(email="a@b.com", password="pw", invite_code="wrong")))
    assert exc.value.status_code == 400


def test_right_invite_code_passes_the_gate():
    # Correct code gets past the invite check; it then fails reaching the (absent) DB,
    # which is exactly the boundary this test asserts — not an AttributeError on the guard.
    manager = UserManager(None)
    with pytest.raises(Exception) as exc:
        asyncio.run(
            manager.create(
                UserCreate(email="a@b.com", password="pw", invite_code=SIGNUP_INVITE_CODE)
            )
        )
    assert not isinstance(exc.value, HTTPException)
