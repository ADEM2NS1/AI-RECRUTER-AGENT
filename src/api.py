# api.py
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any, Dict
import os
import PyPDF2
import logging
import time
import json
from pathlib import Path
import tempfile

from .cv_storage import CVStorage
from qdrant_client import QdrantClient

try:
    from .agentGraph import run_agent, get_bert_embeddings, run_agent_email, run_agent_schedule, _load_last_state
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from agentGraph import run_agent, run_agent_email, run_agent_schedule, _load_last_state

from fastapi import UploadFile, File, Form, Request
from io import BytesIO

try:
    from .preprocessing import preprocess_text
except Exception:
    def preprocess_text(t: str) -> str:
        return " ".join((t or "").split())

app = FastAPI(title="AI Recruiter Agent API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Timing middleware
@app.middleware("http")
async def timing_middleware(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dur_ms = (time.perf_counter() - t0) * 1000
    try:
        response.headers["X-Process-Time-ms"] = f"{dur_ms:.1f}"
    except Exception:
        pass
    print(f"[TIMING] {request.method} {request.url.path} -> {dur_ms:.1f} ms")
    return response

class JobRequest(BaseModel):
    job_description: str
    top_k: int = 5
    with_explain: bool = True

class CandidateResult(BaseModel):
    candidate_id: Any
    score: float
    skills: List[str] | None = None
    experience: Any | None = None
    explanation: str | None = None

class AgentResponse(BaseModel):
    results: List[CandidateResult]

class EmailRequest(BaseModel):
    job_description: str
    top_k: int = 5
    subject: str | None = None
    role: str | None = None
    reply_to: str | None = None

# Chat models and agent wiring
try:
    from .llm_agent import RecruiterAgent, parse_requested_top_k
except ImportError:
    from llm_agent import RecruiterAgent, parse_requested_top_k

class ChatRequest(BaseModel):
    message: str
    max_candidates: int | None = None

class ChatResponse(BaseModel):
    reply: str

agent = RecruiterAgent()

# Email sender wiring
try:
    from .mailer import send_email, render_invite_subject, render_invite_body
except ImportError:
    from mailer import send_email, render_invite_subject, render_invite_body

@app.get("/health")
def health():
    return {"status": "ok"}

# PDF text extractor
def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing error: {e}")

@app.post("/query_from_pdf", response_model=AgentResponse)
def query_from_pdf(file: UploadFile = File(...), top_k: int = 5, with_explain: bool = True):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        raw = file.file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file.")
        text = _extract_text_from_pdf_bytes(raw)
        if not text:
            raise HTTPException(status_code=400, detail="No text found in PDF.")
        job_desc = preprocess_text(text)

        try:
            results: List[Dict[str, Any]] = run_agent(job_desc=job_desc, top_k=top_k, with_explain=with_explain)
        except TypeError:
            try:
                results = run_agent(job_desc, top_k=top_k)
            except TypeError:
                results = run_agent(job_desc)

        for r in results:
            r["score"] = float(r.get("score", 0.0))
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("PDF query error")
        raise HTTPException(status_code=500, detail=str(e))

# Get calendar endpoint
@app.get("/get_calendar")
def get_calendar():
    """Get scheduled interviews from calendar state file."""
    try:
        CALENDAR_STATE_FILE = Path(tempfile.gettempdir()) / "ai_recruiter_calendar_state.json"
        
        if CALENDAR_STATE_FILE.exists():
            with open(CALENDAR_STATE_FILE, 'r') as f:
                state = json.load(f)
            
            appointments = state.get("appointments", {})
            results = state.get("results", [])
            
            # Format interviews
            interviews = []
            for result in results:
                cid = result.get("candidate_id")
                if str(cid) in appointments or cid in appointments:
                    appt = appointments.get(str(cid)) or appointments.get(cid)
                    if appt:
                        interviews.append({
                            "candidate_id": str(cid),
                            "email": result.get("email", "unknown"),
                            "date": appt.get("date"),
                            "time": appt.get("time"),
                            "meeting_link": appt.get("meeting_link")
                        })
            
            print(f"[✓ API] /get_calendar returning {len(interviews)} interviews")
            return {"interviews": interviews, "count": len(interviews)}
        else:
            print("[⚠️ API] /get_calendar - no calendar file found")
            return {"interviews": [], "count": 0}
    except Exception as e:
        logging.exception("Get calendar error")
        return {"interviews": [], "count": 0, "error": str(e)}

# UPDATED: Schedule with config - NOW SUPPORTS APPEND MODE
@app.post("/schedule_with_config")
def schedule_with_config(
    start_date: str = Form(...),
    start_time: str = Form("10:00"),
    duration: int = Form(30),
    append: str = Form("false")  # NEW: append parameter
):
    """Schedule interviews with custom config - can append to existing interviews."""
    try:
        CALENDAR_STATE_FILE = Path(tempfile.gettempdir()) / "ai_recruiter_calendar_state.json"
        
        # Load last state
        state = _load_last_state()
        if not state:
            return {"success": False, "message": "No previous candidate list found. Please run a search first."}
        
        # Add scheduling configuration to state
        state["schedule_start_date"] = start_date
        state["schedule_start_time"] = start_time
        state["schedule_duration"] = duration
        
        # Load existing appointments if appending
        existing_appointments = {}
        existing_results = []
        
        if append == "true" and CALENDAR_STATE_FILE.exists():
            try:
                with open(CALENDAR_STATE_FILE, 'r') as f:
                    existing_state = json.load(f)
                existing_appointments = existing_state.get("appointments", {})
                existing_results = existing_state.get("results", [])
                print(f"[📂 API] Loaded {len(existing_appointments)} existing appointments")
            except Exception as e:
                print(f"[⚠️ API] Error loading existing appointments: {e}")
        
        # Run calendar node
        from .agentGraph import calendar_node, email_sender_node, _save_named_state
        
        with_calendar = calendar_node(state)
        new_appointments = with_calendar.get("appointments", {})
        
        # Merge appointments if appending
        if append == "true":
            # Check for conflicts
            conflicts = []
            for cid, appt in new_appointments.items():
                new_date = appt.get("date")
                new_time = appt.get("time")
                
                # Check if any existing appointment has same date/time
                for existing_cid, existing_appt in existing_appointments.items():
                    if (existing_appt.get("date") == new_date and 
                        existing_appt.get("time") == new_time):
                        conflicts.append({
                            "date": new_date,
                            "time": new_time,
                            "existing_candidate": existing_cid
                        })
            
            # Merge appointments (new ones override conflicts)
            merged_appointments = existing_appointments.copy()
            merged_appointments.update(new_appointments)
            with_calendar["appointments"] = merged_appointments
            
            # Merge results (avoid duplicates)
            merged_results = existing_results.copy()
            new_results = state.get("results", [])
            for new_result in new_results:
                new_cid = new_result.get("candidate_id")
                if not any(r.get("candidate_id") == new_cid for r in merged_results):
                    merged_results.append(new_result)
            with_calendar["results"] = merged_results
            
            print(f"[✓ API] Merged {len(new_appointments)} new with {len(existing_appointments)} existing = {len(merged_appointments)} total")
        
        # Send emails
        email_out = email_sender_node(with_calendar)
        merged = dict(with_calendar)
        merged.update(email_out)
        
        # Save calendar state
        _save_named_state("last_run_with_calendar", merged)
        
        # Save to calendar state file
        try:
            with open(CALENDAR_STATE_FILE, 'w') as f:
                json.dump(merged, f, indent=2)
            print(f"[✓ API] Saved calendar state to {CALENDAR_STATE_FILE}")
        except Exception as e:
            print(f"[⚠️ API] Error saving calendar state: {e}")
        
        entries = email_out.get("emailed_entries", [])
        count = sum(1 for e in entries if e.get("status") == "sent") if entries else 0
        
        total_appointments = len(merged.get("appointments", {}))
        mode = "added" if append == "true" else "scheduled"
        
        return {
            "success": True,
            "message": f"✓ {count} interview(s) {mode}. Total: {total_appointments} appointments.",
            "count": count,
            "total_appointments": total_appointments,
            "interviews": [
                {
                    "candidate_id": str(cid),
                    "email": next((r.get("email") for r in merged.get("results", []) if r.get("candidate_id") == cid), "unknown"),
                    "date": appt.get("date"),
                    "time": appt.get("time"),
                    "meeting_link": appt.get("meeting_link")
                }
                for cid, appt in merged.get("appointments", {}).items()
            ]
        }
    except Exception as e:
        logging.exception("Schedule with config error")
        return {"success": False, "message": str(e)}

# Updated chat endpoint with interview scheduling support
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    file: UploadFile | None = File(None),
    message: str | None = Form(None),
    max_candidates: str | None = Form(None),
):
    try:
        ctype = request.headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                data = await request.json()
            except Exception:
                data = {}
            if isinstance(data, dict):
                message = data.get("message", message)
                if max_candidates is None:
                    max_candidates = data.get("max_candidates")

        user_msg = (message or "").strip()

        requested_max = None
        if max_candidates is not None:
            try:
                requested_max = int(max_candidates)
            except (ValueError, TypeError):
                pass

        requested_k = None
        try:
            requested_k = parse_requested_top_k(user_msg)
        except Exception:
            requested_k = None

        final_k = requested_max if requested_max is not None else requested_k

        # Handle PDF file upload
        if file is not None:
            raw = await file.read()
            if not raw:
                raise HTTPException(status_code=400, detail="Empty file.")
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files are supported.")
            text = _extract_text_from_pdf_bytes(raw)
            if not text:
                raise HTTPException(status_code=400, detail="No text found in PDF.")
            job_desc = preprocess_text(text)
            if user_msg:
                job_desc = f"{job_desc}\n\nAdditional notes: {user_msg}"
            k = final_k or agent.infer_top_k_from_text(user_msg)
            reply = agent.handle_job_description(job_desc, top_k=k)
            return {"reply": reply}

        # Check for schedule interview command
        if "schedule" in user_msg.lower() and "interview" in user_msg.lower():
            print("[🗓️ API] Schedule interviews command detected")
            
            schedule_result = run_agent_schedule()
            
            try:
                CALENDAR_STATE_FILE = Path(tempfile.gettempdir()) / "ai_recruiter_calendar_state.json"
                
                if CALENDAR_STATE_FILE.exists():
                    with open(CALENDAR_STATE_FILE, 'r') as f:
                        state = json.load(f)
                    print(f"[📂 API] Loaded calendar state from file")
                else:
                    state = _load_last_state()
                    print(f"[📂 API] Using regular state as fallback")
            except Exception as e:
                print(f"[⚠️ API] Error loading calendar state: {e}")
                state = _load_last_state()
            
            appointments = state.get("appointments", {}) if state else {}
            results = state.get("results", []) if state else []
            
            print(f"[DEBUG API] Found {len(appointments)} appointments, {len(results)} results")
            
            interview_lines = [schedule_result]
            
            if appointments:
                interview_lines.append("\n📅 Interview Schedule:")
                for result in results:
                    cid = result.get("candidate_id")
                    if str(cid) in appointments or cid in appointments:
                        appt = appointments.get(str(cid)) or appointments.get(cid)
                        if appt:
                            date = appt.get("date")
                            time = appt.get("time")
                            email = result.get("email", "unknown")
                            interview_lines.append(f"Candidate {cid} <{email}> - {date} at {time}")
            else:
                interview_lines.append("\n⚠️ No appointments were created.")
            
            reply = "\n".join(interview_lines)
            print(f"[✓ API] Returning schedule with {len(appointments)} appointments")
            return {"reply": reply}

        # Check for send email command
        if ("send" in user_msg.lower() and "email" in user_msg.lower()) or \
           ("email" in user_msg.lower() and "candidate" in user_msg.lower()):
            print("[📧 API] Send email command detected")
            email_result = run_agent_email()
            return {"reply": email_result}

        # Check if it's a search query
        search_keywords = ["find", "search", "look for", "get", "show me", "candidates", "developers", "engineers"]
        is_search = any(keyword in user_msg.lower() for keyword in search_keywords)
        
        if is_search:
            print(f"[🔍 API] Candidate search detected, using run_agent with top_k={final_k or 5}")
            results = run_agent(job_desc=user_msg, top_k=final_k or 5)
            
            if results:
                reply_lines = ["Here are the top recommendations:\n"]
                for i, r in enumerate(results, 1):
                    cid = r.get("candidate_id")
                    explanation = r.get("explanation", "")
                    skills = r.get("skills", [])
                    
                    reply_lines.append(f"{i}. {cid}: {explanation}")
                    if skills:
                        reply_lines.append(f"   Skills: {', '.join(skills[:5])}")
                    reply_lines.append("")
                
                reply = "\n".join(reply_lines)
            else:
                reply = "No candidates found matching your criteria."
            
            return {"reply": reply}
        else:
            reply = agent.handle_message(user_msg, forced_top_k=final_k)
            return {"reply": reply}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=AgentResponse)
def query_agent(request: JobRequest):
    try:
        try:
            results: List[Dict[str, Any]] = run_agent(
                request.job_description,
                top_k=request.top_k,
                with_explain=request.with_explain
            )
        except TypeError:
            results = run_agent(
                request.job_description,
                top_k=request.top_k
            )

        for r in results:
            r["score"] = float(r.get("score", 0.0))

        return {"results": results}
    except Exception as e:
        logging.exception("Agent error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/email")
def email():
    summary = run_agent_email()
    return {"summary": summary}

@app.post("/send_emails")
def send_emails(req: EmailRequest):
    """Run candidate search then send (or dry-run) emails to top matches."""
    try:
        try:
            results: List[Dict[str, Any]] = run_agent(
                req.job_description, top_k=req.top_k, with_explain=True
            )
        except TypeError:
            results = run_agent(req.job_description, top_k=req.top_k)

        from .cv_storage import CVStorage
        storage = CVStorage()
        subject = req.subject or render_invite_subject(req.role)
        sent = 0
        attempted = 0
        failures: list[dict] = []

        for r in results[: req.top_k]:
            cid = r.get("candidate_id")
            if cid is None:
                continue
            email_addr = r.get("email") or storage.get_email_by_candidate_id(cid)
            if not email_addr:
                failures.append({"candidate_id": cid, "reason": "no email"})
                continue
            body = render_invite_body(r.get("name"), req.job_description, req.reply_to)
            attempted += 1
            try:
                if send_email(email_addr, subject, body, reply_to=req.reply_to):
                    sent += 1
                else:
                    failures.append({"candidate_id": cid, "reason": "send failed"})
            except Exception as e:
                failures.append({"candidate_id": cid, "reason": str(e)})

        return {
            "attempted": attempted,
            "sent": sent,
            "failures": failures,
            "from_email": os.getenv("FROM_EMAIL", os.getenv("SMTP_USER", "")),
            "dry_run": os.getenv("EMAIL_DRY_RUN", "1") == "1",
        }
    except Exception as e:
        logging.exception("Send emails error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/schedule_interviews")
def schedule_interviews():
    """Generate calendar appointments for last results and email candidates with details."""
    try:
        summary = run_agent_schedule()
        return {"summary": summary}
    except Exception as e:
        logging.exception("Schedule interviews error")
        raise HTTPException(status_code=500, detail=str(e))
# Add these imports at the top of api.py
from cv_processor import process_cv_file, score_cv_against_job

# Add these two endpoints anywhere in your api.py after line 100 or so:

@app.post("/upload_cv")
async def upload_cv(
    file: UploadFile = File(...),
    candidate_name: str = Form(None),
    candidate_email: str = Form(None),
    candidate_phone: str = Form(None)
):
    """Upload CV (PDF or image) and add to Qdrant database with BERT embeddings."""
    try:
        # Read file
        file_bytes = await file.read()
        
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Process and store using cv_processor
        result = process_cv_file(
            file_bytes=file_bytes,
            filename=file.filename,
            name=candidate_name,
            email=candidate_email,
            phone=candidate_phone
        )
        
        if result.get("success"):
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get("message", "Processing failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Upload CV error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score_cv")
async def score_cv_endpoint(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    """Score a CV against job description using BERT embeddings."""
    try:
        file_bytes = await file.read()
        
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        
        if not job_description or not job_description.strip():
            raise HTTPException(status_code=400, detail="Job description is required")
        
        # Score CV
        result = score_cv_against_job(
            file_bytes=file_bytes,
            filename=file.filename,
            job_description=job_description
        )
        
        if result.get("success"):
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Scoring failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Score CV error")
        raise HTTPException(status_code=500, detail=str(e))
