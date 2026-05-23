import asyncio
import uuid
from app.state import jobs, job_subscribers
from app.services.fun_facts import get_fact
from app.services.stac_client import search_sentinel2
from app.services.feature_builder import build_features_from_stac
from app.services.model_runner import model_service

async def broadcast(job_id: str, payload: dict):
    queues = job_subscribers.get(job_id, [])
    for q in queues:
        await q.put(payload)

def update_job(job_id: str, stage: str, progress: int, message: str):
    jobs[job_id].update({
        "stage": stage,
        "progress": progress,
        "message": message,
        "fact": get_fact(progress)
    })

async def run_inference_job(job_id: str, req):
    try:
        update_job(job_id, "queued", 5, "Job accepted. Preparing area request.")
        await broadcast(job_id, jobs[job_id].copy())
        await asyncio.sleep(0.4)

        update_job(job_id, "searching", 20, "Searching Sentinel-2 scenes.")
        await broadcast(job_id, jobs[job_id].copy())
        stac = await search_sentinel2(req.bbox, req.datetime_start, req.datetime_end, req.max_cloud_cover)
        await asyncio.sleep(0.4)

        update_job(job_id, "features", 50, "Building remote-sensing feature vector.")
        await broadcast(job_id, jobs[job_id].copy())
        feats = build_features_from_stac(stac, req.bbox)
        await asyncio.sleep(0.4)

        update_job(job_id, "inference", 75, "Running GBM inference and confidence checks.")
        await broadcast(job_id, jobs[job_id].copy())
        pred = model_service.predict(feats)
        await asyncio.sleep(0.4)

        result = {
            "job_id": job_id,
            "area_name": req.name,
            "bbox": req.bbox,
            "scene_count": int(feats["scene_count"]),
            "source": "sentinel-2-l2a-stac",
            "status": "completed",
            **pred,
        }
        jobs[job_id].update({
            "stage": "completed",
            "progress": 100,
            "message": "Inference complete.",
            "result": result
        })
        await broadcast(job_id, jobs[job_id].copy())

    except Exception as e:
        jobs[job_id].update({
            "stage": "failed",
            "progress": 100,
            "message": str(e),
            "status": "failed"
        })
        await broadcast(job_id, jobs[job_id].copy())

def create_job(req):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "stage": "created",
        "progress": 0,
        "message": "Created."
    }
    asyncio.create_task(run_inference_job(job_id, req))
    return job_id