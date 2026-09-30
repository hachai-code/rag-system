"""User identity: fastapi-users over async SQLAlchemy, sharing the app's Postgres.

Registration is gated by a shared invite code (SIGNUP_INVITE_CODE); login issues a JWT
bearer token the frontend sends as `Authorization: Bearer` on every request. The `user`
table and the user_id owner columns live in db/migrations/0007 — this module maps the
existing table, it does not create schema.

The rest of the app talks to Postgres through sync psycopg; only this module uses async
SQLAlchemy (asyncpg), because that is what fastapi-users is built on. Separate pools, one
database. A sync endpoint can still depend on the async `current_active_user`.
"""

import os
import uuid
from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, schemas
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy
from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .db import DB_URL

# ponytail: dev defaults so `fastapi dev` and CI import run out of the box. In production
# both MUST be set in the environment — an unset SECRET means forgeable tokens.
SECRET = os.environ.get("SECRET", "dev-insecure-secret-change-me")
SIGNUP_INVITE_CODE = os.environ.get("SIGNUP_INVITE_CODE", "innerdance")
TOKEN_LIFETIME = 60 * 60 * 24 * 7  # one week, then re-login

_ASYNC_DB_URL = DB_URL.replace("postgresql://", "postgresql+asyncpg://", 1).replace(
    "postgres://", "postgresql+asyncpg://", 1
)
engine = create_async_engine(_ASYNC_DB_URL)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    pass


async def get_async_session() -> AsyncGenerator[AsyncSession]:
    async with async_session_maker() as session:
        yield session


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)


class UserRead(schemas.BaseUser[uuid.UUID]):
    pass


class UserCreate(schemas.BaseUserCreate):
    invite_code: str


class UserUpdate(schemas.BaseUserUpdate):
    pass


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def create(
        self, user_create: UserCreate, safe: bool = False, request: Request | None = None
    ) -> User:
        if user_create.invite_code != SIGNUP_INVITE_CODE:
            raise HTTPException(status_code=400, detail="INVALID_INVITE_CODE")
        clean = schemas.BaseUserCreate(email=user_create.email, password=user_create.password)
        return await super().create(clean, safe=safe, request=request)


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=TOKEN_LIFETIME)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=BearerTransport(tokenUrl="auth/jwt/login"),
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

current_active_user = fastapi_users.current_user(active=True)
