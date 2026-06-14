import asyncio
import io
import http
import logging
import time
import traceback
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from PIL import Image
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            # WAN clients send multi-MB image observations through an SSH reverse
            # tunnel. Disable server-side websocket keepalive pings so a slow or
            # temporarily busy client isn't closed while uploading / rendering.
            ping_interval=None,
            close_timeout=10,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))
        logger.info("Metadata sent to %s: keys=%s", websocket.remote_address, sorted(self._metadata.keys()))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                raw_obs = await websocket.recv()
                recv_time = time.monotonic() - start_time
                logger.info(
                    "Observation received from %s: frame_type=%s bytes=%s recv_ms=%.1f",
                    websocket.remote_address,
                    type(raw_obs).__name__,
                    len(raw_obs) if hasattr(raw_obs, "__len__") else None,
                    recv_time * 1000,
                )
                obs = _decode_compressed_images(msgpack_numpy.unpackb(raw_obs))

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
                logger.info(
                    "Inference succeeded for %s: action_shape=%s infer_ms=%.1f total_ms=%.1f",
                    websocket.remote_address,
                    _get_action_shape(action),
                    infer_time * 1000,
                    prev_total_time * 1000,
                )

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def _decode_compressed_images(obs: dict[str, Any]) -> dict[str, Any]:
    """Accept JPEG/PNG bytes in obs['images'] and convert them to CHW uint8.

    The original OpenPI websocket client sends raw numpy arrays, which are still
    supported. WAN clients can instead send the JPEG bytes already present in
    policy_deployment sim bundles; this reduces each 3-camera observation from
    ~2.77 MB to ~0.28 MB before it enters the SSH reverse tunnel.
    """

    images = obs.get("images")
    if not isinstance(images, dict):
        return obs

    decoded_any = False
    decoded_images: dict[str, Any] = {}
    encoded_bytes = 0
    decoded_shapes: dict[str, tuple[int, ...]] = {}

    for key, value in images.items():
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
            encoded_bytes += len(raw)
            with Image.open(io.BytesIO(raw)) as image:
                arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
            decoded = np.ascontiguousarray(arr.transpose(2, 0, 1))
            decoded_images[key] = decoded
            decoded_shapes[key] = tuple(decoded.shape)
            decoded_any = True
        else:
            decoded_images[key] = value

    if decoded_any:
        obs = dict(obs)
        obs["images"] = decoded_images
        logger.info(
            "Decoded compressed observation images: encoded_bytes=%s decoded_shapes=%s",
            encoded_bytes,
            decoded_shapes,
        )
    return obs


def _get_action_shape(action: dict[str, Any]) -> tuple[int, ...] | None:
    actions = action.get("actions")
    return tuple(actions.shape) if hasattr(actions, "shape") else None
