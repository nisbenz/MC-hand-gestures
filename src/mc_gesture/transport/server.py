"""WebSocket server (D10): games connect and receive JSON messages.

    ws://127.0.0.1:8765

On connect the server sends {"type": "hello", "layers": [...]}. A client picks
its layers by sending {"subscribe": ["events", "cursor", "landmarks", "health"]};
the default is events + health (D7).

The engine loop runs on the main thread; the server has its own thread and
asyncio loop, and publish() hands messages over thread-safely.
"""

import asyncio
import json
import threading

from websockets.asyncio.server import ServerConnection, broadcast, serve

LAYERS = ("events", "cursor", "landmarks", "health")
DEFAULT_LAYERS = {"events", "health"}


class Server:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host, self.port = host, port
        self._clients: dict[ServerConnection, set[str]] = {}
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop: asyncio.Future | None = None
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="mc-gesture-ws", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error:
            raise self._error

    @property
    def clients(self) -> int:
        return len(self._clients)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        self._stop = self._loop.create_future()
        try:
            async with serve(self._handle, self.host, self.port):
                self._ready.set()
                await self._stop
        except OSError as error:  # e.g. port already in use
            self._error = error
            self._ready.set()

    async def _handle(self, ws: ServerConnection) -> None:
        self._clients[ws] = set(DEFAULT_LAYERS)
        await ws.send(json.dumps({"type": "hello", "layers": sorted(DEFAULT_LAYERS), "available": list(LAYERS)}))
        try:
            async for raw in ws:
                try:
                    message = json.loads(raw)
                    layers = {layer for layer in message.get("subscribe", []) if layer in LAYERS}
                except (ValueError, AttributeError):
                    await ws.send(json.dumps({"type": "error", "message": "expected {\"subscribe\": [...]}"}))
                    continue
                self._clients[ws] = layers
                await ws.send(json.dumps({"type": "subscribed", "layers": sorted(layers)}))
        finally:
            self._clients.pop(ws, None)

    def publish(self, layer: str, message: dict) -> None:
        if self._clients:
            self._loop.call_soon_threadsafe(self._send, layer, json.dumps(message))

    def _send(self, layer: str, text: str) -> None:
        broadcast([ws for ws, layers in self._clients.items() if layer in layers], text)

    def close(self) -> None:
        if self._stop and not self._stop.done():
            self._loop.call_soon_threadsafe(self._stop.set_result, None)
        self._thread.join(timeout=2)
