import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import hmac
import traceback
from fastapi import FastAPI, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client

app = FastAPI(title="Solar Engine")

ENGINE_KEY = os.environ.get("ENGINE_KEY", "")
supabase_url = os.environ.get("SUPABASE_URL", "").strip()
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()

sb = None
try:
    if supabase_url and supabase_key:
        sb = create_client(supabase_url, supabase_key)
        print("Supabase client initialized successfully.")
except Exception as e:
    print("Error initializing Supabase client:")
    traceback.print_exc()

def check_key(key: str):
    if not hmac.compare_digest(key or "", ENGINE_KEY):
        raise HTTPException(401, "bad key")

class Job(BaseModel):
    project_id: str

def process_job_background(project_id: str):
    print(f"Starting background job for project: {project_id}")
    try:
        # โหลดไลบรารีหนักๆ เฉพาะตอนที่ต้องประมวลผลจริงเท่านั้น เพื่อประหยัด RAM ตอนเริ่มต้น
        try:
            from simulate import run_simulation
        except Exception as e:
            print("Error importing simulate:", e)
            run_simulation = None

        try:
            from report import render_pdf
        except Exception as e:
            print("Error importing report:", e)
            render_pdf = None

        if not sb:
            print("Supabase client not available.")
            return
        
        # 1. ดึงข้อมูลโปรเจกต์จาก Supabase
        res = sb.table("projects").select("*").eq("id", project_id).execute()
        if not res.data:
            print(f"Project {project_id} not found.")
            return
        project = res.data[0]

        # 2. รันจำลองระบบโซลาเซลล์
        sim_result = {}
        if run_simulation:
            sim_result = run_simulation(project)
        
        # 3. สร้างรายงาน PDF
        pdf_path = f"/tmp/{project_id}.pdf"
        if render_pdf:
            render_pdf(project, sim_result, pdf_path)
        
        # 4. อัปโหลด PDF ขึ้น Supabase Storage
        public_url = None
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                file_bytes = f.read()
            
            storage_path = f"reports/{project_id}.pdf"
            sb.storage.from_("reports").upload(
                storage_path, 
                file_bytes, 
                {"fileOptions": {"upsert": "true"}}
            )
            public_url = sb.storage.from_("reports").get_public_url(storage_path)

        # 5. อัปเดตสถานะเป็น completed
        sb.table("projects").update({
            "status": "completed",
            "pdf_url": public_url,
            "error": None
        }).eq("id", project_id).execute()
        
        print(f"Project {project_id} completed successfully.")

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"Error processing project {project_id}: {err_msg}")
        if sb:
            sb.table("projects").update({
                "status": "error",
                "error": str(e)
            }).eq("id", project_id).execute()

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
    
    bg.add_task(process_job_background, job.project_id)
    return {"status": "accepted", "project_id": job.project_id}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)