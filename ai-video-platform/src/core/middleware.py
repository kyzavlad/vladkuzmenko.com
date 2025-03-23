import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .monitoring import record_request_metrics
from .logging import setup_logging

logger = setup_logging()

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logger

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        self.logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_host": request.client.host if request.client else None
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            record_request_metrics(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
                duration=duration
            )
            
            # Log response
            self.logger.info(
                f"Request completed: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration": duration
                }
            )
            
            return response
            
        except Exception as e:
            # Log error
            self.logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e)
                },
                exc_info=True
            )
            raise

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logger

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else None
        
        # Check rate limit
        if await self.is_rate_limited(client_ip):
            return Response(
                status_code=429,
                content={"error": "Too many requests"}
            )
        
        return await call_next(request)
    
    async def is_rate_limited(self, client_ip: str) -> bool:
        # Implement rate limiting logic here
        # This is a placeholder implementation
        return False

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logger

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response 