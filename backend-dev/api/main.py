"""
api.main
--------
FastAPI entry point.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import BASE_DIR
from api.routes import router

app = FastAPI(
    title='eelsSaver API',
    description='Pan-Atlantic eelgrass early-warning system.',
    version='0.1.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],          # tighten for production
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)

app.include_router(router, prefix='/api')

# Serve the Leaflet web map at /
WEB_DIR = BASE_DIR / 'web'
if WEB_DIR.exists():
    app.mount('/', StaticFiles(directory=str(WEB_DIR), html=True), name='web')


@app.get('/healthz', include_in_schema=False)
def healthz():
    return {'ok': True}
