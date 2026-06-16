from __future__ import annotations

import re
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web.models.database import User, get_db
from web.models.schemas import UserLogin, Token, UserResponse
from web.services.auth import verify_password, create_access_token, get_current_user, get_password_hash

router = APIRouter(prefix="/api/auth", tags=["auth"])

EMAIL_REGEX = re.compile(r"^[^@]+@[^@]+\.[^@]+$")


@router.post("/login", response_model=Token)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """Log in a user and return a JWT access token."""
    email_clean = credentials.email.strip().lower()
    
    if not EMAIL_REGEX.match(email_clean):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format",
        )

    result = await db.execute(select(User).where(User.email == email_clean))
    user = result.scalar_one_or_none()

    if not user:
        # Auto-registration for new users
        hashed = get_password_hash(credentials.password)
        new_user = User(
            email=email_clean,
            hashed_password=hashed,
            is_active=True,
        )
        db.add(new_user)
        await db.commit()
        user = new_user

    else:
        if not verify_password(credentials.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user account",
            )

    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: User = Depends(get_current_user),
):
    """Return the profile details of the currently logged-in user."""
    return current_user

