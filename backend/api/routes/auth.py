"""
Authentication API Routes
Handles user authentication and authorization
"""

import asyncio
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import structlog

from ...utils.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()
security = HTTPBearer()

class UserLogin(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=100)

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = Field(None, max_length=100)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

class UserProfile(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    preferences: dict

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)

# Mock user database (in production, use real database)
MOCK_USERS = {
    "admin": {
        "user_id": "admin_001",
        "username": "admin",
        "email": "admin@astrafind.com",
        "password_hash": "hashed_password_here",  # In production, use proper hashing
        "full_name": "Administrator",
        "created_at": datetime.now(),
        "last_login": None,
        "preferences": {
            "language": "en",
            "theme": "light",
            "safe_search": True
        }
    }
}

# Mock tokens (in production, use proper JWT tokens)
MOCK_TOKENS = {}

def create_access_token(user_id: str) -> str:
    """Create access token (simplified implementation)"""
    # In production, use proper JWT tokens
    token = f"mock_token_{user_id}_{datetime.now().timestamp()}"
    MOCK_TOKENS[token] = {
        "user_id": user_id,
        "expires_at": datetime.now() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    return token

def verify_token(token: str) -> Optional[str]:
    """Verify access token (simplified implementation)"""
    if token in MOCK_TOKENS:
        token_data = MOCK_TOKENS[token]
        if token_data["expires_at"] > datetime.now():
            return token_data["user_id"]
    return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get current user from token"""
    user_id = verify_token(credentials.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id

@router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    """User login endpoint"""
    try:
        # Check if user exists (simplified)
        if user_data.username not in MOCK_USERS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        user = MOCK_USERS[user_data.username]
        
        # In production, verify password hash
        if user_data.password != "password":  # Simplified password check
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Update last login
        user["last_login"] = datetime.now()
        
        # Create access token
        access_token = create_access_token(user["user_id"])
        
        return TokenResponse(
            access_token=access_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=None  # In production, implement refresh tokens
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/register", response_model=TokenResponse)
async def register(user_data: UserRegister):
    """User registration endpoint"""
    try:
        # Check if username already exists
        if user_data.username in MOCK_USERS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Create new user (simplified)
        user_id = f"user_{len(MOCK_USERS) + 1:03d}"
        new_user = {
            "user_id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "password_hash": "hashed_password_here",  # In production, hash password
            "full_name": user_data.full_name,
            "created_at": datetime.now(),
            "last_login": None,
            "preferences": {
                "language": "en",
                "theme": "light",
                "safe_search": True
            }
        }
        
        MOCK_USERS[user_data.username] = new_user
        
        # Create access token
        access_token = create_access_token(user_id)
        
        return TokenResponse(
            access_token=access_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/logout")
async def logout(current_user: str = Depends(get_current_user)):
    """User logout endpoint"""
    try:
        # In production, invalidate token
        # For now, just return success
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/profile", response_model=UserProfile)
async def get_user_profile(current_user: str = Depends(get_current_user)):
    """Get user profile"""
    try:
        # Find user by ID
        user = None
        for u in MOCK_USERS.values():
            if u["user_id"] == current_user:
                user = u
                break
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserProfile(
            user_id=user["user_id"],
            username=user["username"],
            email=user["email"],
            full_name=user["full_name"],
            created_at=user["created_at"],
            last_login=user["last_login"],
            preferences=user["preferences"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/profile")
async def update_user_profile(
    profile_update: dict,
    current_user: str = Depends(get_current_user)
):
    """Update user profile"""
    try:
        # Find user by ID
        user = None
        for u in MOCK_USERS.values():
            if u["user_id"] == current_user:
                user = u
                break
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update allowed fields
        allowed_fields = ["full_name", "preferences"]
        for field, value in profile_update.items():
            if field in allowed_fields:
                user[field] = value
        
        return {"message": "Profile updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user profile: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: str = Depends(get_current_user)
):
    """Change user password"""
    try:
        # Find user by ID
        user = None
        for u in MOCK_USERS.values():
            if u["user_id"] == current_user:
                user = u
                break
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Verify current password (simplified)
        if password_data.current_password != "password":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password (in production, hash the new password)
        user["password_hash"] = "new_hashed_password_here"
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to change password: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/account")
async def delete_account(
    password: str,
    current_user: str = Depends(get_current_user)
):
    """Delete user account"""
    try:
        # Find user by ID
        user = None
        username = None
        for uname, u in MOCK_USERS.items():
            if u["user_id"] == current_user:
                user = u
                username = uname
                break
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Verify password (simplified)
        if password != "password":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password is incorrect"
            )
        
        # Delete user
        del MOCK_USERS[username]
        
        return {"message": "Account deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete account: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/verify")
async def verify_token_endpoint(current_user: str = Depends(get_current_user)):
    """Verify token validity"""
    try:
        return {"valid": True, "user_id": current_user}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")