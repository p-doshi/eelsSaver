import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from app.schemas import InferenceRequest, InferenceStartResponse
from app.services.jobs import create_job
from app.state import jobs, job_subscribers

router = APIRouter(prefix="/api/inference", tags=["inference"])

@router.post("/start", response_model=InferenceStartResponse)
async def start_inference(req: InferenceRequest):
    job_id = create_job(req)
    return {"job_id": job_id, "status": "running"}

@router.get("/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.websocket("/ws/{job_id}")
async def inference_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    queue = asyncio.Queue()
    job_subscribers[job_id].append(queue)
    try:
        if job_id in jobs:
            await websocket.send_json(jobs[job_id])
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
            if payload.get("stage") in {"completed", "failed"}:
                break
    except WebSocketDisconnect:
        pass
    finally:
        if queue in job_subscribers[job_id]:
            job_subscribers[job_id].remove(queue)