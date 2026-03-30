import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# Global Worker Pool injected from main
worker_pool = None

def init_websockets(wp):
    global worker_pool
    worker_pool = wp

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Client sends Base64 encoded Frames & Audio chunks
    Server streams back fused emotion JSON.
    """
    await websocket.accept()
    
    # Background task to send fused updates to client
    async def result_sender():
        async for result in worker_pool.get_fused_result_stream():
            try:
                await websocket.send_json(result)
            except WebSocketDisconnect:
                break
                
    task = asyncio.create_task(result_sender())

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            # Note: In production, payload would decode base64 bytes to NumPy arrays
            if payload.get("type") == "video_frame":
                # Mock decode
                # frame = decode_b64(payload['data'])
                # await worker_pool.face_queue.put(frame)
                pass
            elif payload.get("type") == "audio_chunk":
                # Mock decode
                # chunk = decode_b64(payload['data'])
                # await worker_pool.audio_queue.put(chunk)
                pass

    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        task.cancel()
