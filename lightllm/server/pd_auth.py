import hmac

from fastapi import WebSocket

from lightllm.utils.envs_utils import get_pd_auth_token
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def get_pd_auth_headers():
    token = get_pd_auth_token()
    if token is None:
        return None
    return {"Authorization": f"Bearer {token}"}


async def check_pd_auth(websocket: WebSocket) -> bool:
    token = get_pd_auth_token()
    if token is None:
        return True
    received = websocket.headers.get("authorization", "")
    if hmac.compare_digest(received.encode(), f"Bearer {token}".encode()):
        return True
    logger.warning(f"rejected unauthenticated PD connection from {websocket.client}")
    await websocket.close(code=1008)
    return False
