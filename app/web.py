"""Backend API & Dashboard: serves the web page and forwards its requests to the Speed Limit Service."""
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

SPEEDLIMIT_URL = os.getenv("SPEEDLIMIT_URL", "http://localhost:8001")
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Speed Limit Dashboard")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.api_route("/api/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    async with httpx.AsyncClient(timeout=600) as client:
        r = await client.request(request.method, f"{SPEEDLIMIT_URL}/{path}",
                                 params=dict(request.query_params), content=await request.body(),
                                 headers={"content-type": request.headers.get("content-type", "application/json")})
    return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/health")
def health():
    return {"status": "ok"}
