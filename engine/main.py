import os, hmac, traceback
from fastapi import FastAPI, BackgroundTasks, Header, HTTPException, Body
from fastapi.responses import Response
from pydantic import BaseModel
from supabase import create_client
from simulate import run_simulation
from report import render_pdf

ENGINE_KEY = os.environ["ENGINE_KEY"]
sb = (create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
      if os.getenv("SUPABASE_URL") else None)
app = FastAPI(title="Solar Engine")

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
    claimed = (sb.table("projects").update({"status": "processing", "error": None})
               .eq("id", job.project_id).eq("status", "paid").execute())
    if not claimed.data:
        return {"skipped": True}
    bg.add_task(process, job.project_id)
    return {"queued": True}

def process(pid: str):
    try:
        p = sb.table("projects").select("user_id,data").eq("id", pid).single().execute().data
        result = run_simulation(p["data"])
        pdf = render_pdf(p["data"], result, pid)
        path = f"{p['user_id']}/{pid}.pdf"
        sb.storage.from_("reports").upload(path, pdf, {"content-type": "application/pdf", "upsert": "true"})
        sb.table("projects").update({
            "status": "done", "pdf_path": path,
            "result": {"summary": result["summary"], "monthly": result["monthly"]},
        }).eq("id", pid).execute()
    except Exception as e:
        traceback.print_exc()
        sb.table("projects").update({"status": "error", "error": str(e)[:500]}).eq("id", pid).execute()

@app.post("/preview")
def preview(project: dict = Body(...), x_engine_key: str = Header("")):
    check_key(x_engine_key)
    result = run_simulation(project)
    return Response(render_pdf(project, result, "PREVIEW"), media_type="application/pdf")
