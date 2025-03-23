from typing import Optional
from pydantic import BaseModel

class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None

class TokenPayload(BaseModel):
    """Token payload model."""
    sub: str
    exp: int
    iat: int 