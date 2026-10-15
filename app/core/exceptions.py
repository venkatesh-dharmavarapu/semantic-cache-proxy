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