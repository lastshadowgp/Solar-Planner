import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import hmac
import traceback

from fastapi import FastAPI, BackgroundTasks, Header, HTTPException, Body
from fastapi.responses import Response
from pydantic import BaseModel
from supabase import create_client

# ป้องกันแอปพังถ้ายังไม่มีไฟล์จำลองหรือระบบ PDF
try:
    from simulate import run_simulation
except Exception as e:
    print("Warning: Could not import simulate:", e)

try:
    from report import render_pdf
except Exception as e:
    print("Warning: Could not import render_pdf:", e)
    render_pdf = None

app = FastAPI(title="Solar Engine")

ENGINE_KEY = os.environ.get("ENGINE_KEY", "")
supabase_url = os.environ.get("SUPABASE_URL", "").strip()
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()

sb = None
try:
    if supabase_url and supabase_key:
        sb = create_client(supabase_url, supabase_key)
        print("Supabase client initialized successfully.")
    else:
        print("Warning: Supabase credentials missing.")
except Exception as e:
    print("Error initializing Supabase client:")
    traceback.print_exc()

def check_key(key: str):
    if not hmac.compare_digest(key or "", ENGINE_KEY):
        raise HTTPException(401, "bad key")

class Job(BaseModel):
    project_id: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/report", status_code=202)
def report(job: Job, bg: BackgroundTasks, x_engine_key: str = Header("")):
    check_key(x_engine_key)
    if sb:
        claimed = (sb.table("projects").update({"status": "processing", "error": None})
                  .eq("id", job.project_id).eq("status", "paid").execute())
        if not claimed.data:
            raise HTTPException(400, "Project not found or status is not paid")
    return {"status": "accepted"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)