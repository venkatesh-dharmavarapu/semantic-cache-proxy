from fastapi import HTTPException, status


class UpstreamProviderError(HTTPException):
    def __init__(self, detail: str = "Upstream LLM provider failed to respond."):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": {
                    "message": detail,
                    "type": "upstream_provider_error",
                    "code": 502,
                }
            },
        )


class UpstreamTimeoutError(HTTPException):
    def __init__(self, detail: str = "Upstream LLM provider request timed out."):
        super().__init__(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error": {
                    "message": detail,
                    "type": "upstream_timeout_error",
                    "code": 504,
                }
            },
        )


class RateLimitExceededError(HTTPException):
    def __init__(self, retry_after: int, limit: int):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "message": f"Rate limit exceeded. Maximum {limit} requests per minute allowed.",
                    "type": "rate_limit_exceeded",
                    "code": 429,
                }
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
            },
        )


class TenantBudgetExceededError(HTTPException):
    def __init__(self, tenant_id: str, current_used: int, max_tokens: int):
        super().__init__(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": {
                    "message": f"Token budget exceeded for tenant '{tenant_id}'. Used {current_used}/{max_tokens} tokens.",
                    "type": "budget_exceeded_error",
                    "code": 402,
                }
            },
        )