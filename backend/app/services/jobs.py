import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

from app.services.feature_builder import build_feature_matrix
from app.services.fun_facts import get_fact
from app.services.model_runner import model_service
from app.state import job_subscribers, jobs

log = logging.getLogger(__name__)

# Pixel reads + XGBoost inference are blocking & CPU/IO-bound — keep them off
# the event loop so the WebSocket progress stream stays responsive.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference")


async def broadcast(job_id: str, payload: dict):
    queues = job_subscribers.get(job_id, [])
    for q in queues:
        await q.put(payload)


def update_job(job_id: str, stage: str, progress: int, message: str):
    jobs[job_id].update(
        {
            "stage": stage,
            "progress": progress,
            "message": message,
            "fact": get_fact(progress),
        }
    )


async def _push(job_id: str, stage: str, progress: int, message: str):
    update_job(job_id, stage, progress, message)
    await broadcast(job_id, jobs[job_id].copy())


async def run_inference_job(job_id: str, req):
    try:
        await _push(job_id, "queued", 5, "Job accepted. Preparing area request.")

        await _push(
            job_id,
            "searching",
            15,
            f"Searching Sentinel-2 scenes ({req.datetime_start} → {req.datetime_end}).",
        )

        await _push(
            job_id,
            "features",
            35,
            "Asking Earth Engine to compute spectral indices and sample pixels.",
        )

        loop = asyncio.get_running_loop()
        features, meta = await loop.run_in_executor(
            _executor,
            build_feature_matrix,
            req.bbox,
            req.datetime_start,
            req.datetime_end,
            req.max_cloud_cover,
            req.name or "bbox_query",
        )

        await _push(
            job_id,
            "features_ready",
            65,
            f"{meta['scene_count']} scenes · {meta['pixel_count']} pixels · "
            f"{meta['window_count']} rolling windows.",
        )

        await _push(job_id, "inference", 82, "Running XGBoost students and OOD checks.")
        pred = await loop.run_in_executor(
            _executor, model_service.predict_from_features, features, meta
        )

        result = {
            "job_id": job_id,
            "area_name": req.name,
            "bbox": req.bbox,
            "scene_count": int(meta["scene_count"]),
            "source": meta.get("data_source", "earthengine"),
            "date_range": [meta["date_min"], meta["date_max"]],
            "status": "completed",
            **pred,
        }
        jobs[job_id].update(
            {
                "stage": "completed",
                "progress": 100,
                "message": "Inference complete.",
                "result": result,
            }
        )
        await broadcast(job_id, jobs[job_id].copy())

    except Exception as e:
        log.exception("Inference job %s failed", job_id)
        jobs[job_id].update(
            {
                "stage": "failed",
                "progress": 100,
                "message": str(e),
                "status": "failed",
            }
        )
        await broadcast(job_id, jobs[job_id].copy())


def create_job(req):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "stage": "created",
        "progress": 0,
        "message": "Created.",
    }
    asyncio.create_task(run_inference_job(job_id, req))
    return job_id
