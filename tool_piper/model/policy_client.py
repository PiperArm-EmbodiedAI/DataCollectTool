from __future__ import annotations

import time
from typing import Any

import numpy as np
import websockets.sync.client

from tool_piper.model import msgpack_numpy


class PolicyClient:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, api_key: str | None = None, wait: bool = False):
        if host.startswith("ws"):
            self.uri = host
        else:
            self.uri = f"ws://{host}"
        if port is not None:
            self.uri += f":{port}"
        self.api_key = api_key
        self.ws = self._connect(wait=wait)
        self.metadata = msgpack_numpy.unpackb(self.ws.recv())
        self.packer = msgpack_numpy.Packer()

    def _connect(self, wait: bool):
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
                return websockets.sync.client.connect(self.uri, compression=None, max_size=None, additional_headers=headers)
            except ConnectionRefusedError:
                if not wait:
                    raise
                time.sleep(5)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.ws.send(self.packer.pack(observation))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Inference server returned an error:\n{response}")
        return msgpack_numpy.unpackb(response)


def policy_dry_run(host: str, port: int, observation: dict, api_key: str | None = None) -> np.ndarray:
    client = PolicyClient(host=host, port=port, api_key=api_key)
    result = client.infer(observation)
    if "actions" not in result:
        raise KeyError(f"Policy response missing 'actions': {result.keys()}")
    return np.asarray(result["actions"], dtype=np.float32)[:, :7]
