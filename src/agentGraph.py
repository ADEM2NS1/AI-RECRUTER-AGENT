from typing import Dict, Any, List
import numpy as np
import re
import os
import json  # ✅ ADD THIS
import tempfile  # ✅ ADD THIS
from pathlib import Path  # ✅ ADD THIS

# ... rest of your imports ...

# ✅ ADD THESE LINES after all imports and before cv_storage initialization
STATE_FILE = Path(tempfile.gettempdir()) / "ai_recruiter_state.json"
CALENDAR_STATE_FILE = Path(tempfile.gettempdir()) / "ai_recruiter_calendar_state.json"
from datetime import datetime, timedelta

# Robust import across langgraph versions
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    try:
        from langgraph.graph.graph import Graph as StateGraph
        from langgraph.graph import END
    except Exception:
        StateGraph = None
        END = None

from cv_storage import CVStorage

# Attempt to import real email sender utilities
try:
    from mailer import send_email, render_invite_subject, render_invite_body
except Exception:
    try:
        from .mailer import send_email, render_invite_subject, render_invite_body  # type: ignore
    except Exception:
        send_email = None
        def render_invite_subject(role: str | None = None) -> str:
            return f"Exciting Opportunity: Interview Invitation{f' - {role} Position' if role else ''}"
        
        def render_invite_body(candidate_name: str | None, job_desc_snippet: str, reply_to: str | None = None) -> str:
            name = candidate_name or "Candidate"
            return (
                f"Dear {name},\n\n"
                f"We were impressed by your background and would like to invite you to interview for an exciting opportunity.\n\n"
                f"Position Overview:\n{job_desc_snippet[:500]}\n\n"
                f"We believe your experience aligns well with this role and would love to discuss how you could contribute to our team.\n\n"
                f"Please let us know your availability for an interview.\n\n"
                f"Best regards,\n"
                f"Talent Acquisition Team"
            )

# Try imports from agentNodes; provide safe fallbacks
try:
    from agentNodes import analyze_cv, rank_candidates, explain_rankingLLM, _sanitize_vec
except Exception:
    def _sanitize_vec(vec):
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
        return arr

    def analyze_cv(t: str):
        return {"skills": [], "experience": None, "full_text": t or ""}

    def explain_rankingLLM(info: Dict[str, Any], job: str) -> str:
        return "Strong candidate profile matching the role requirements."

    def rank_candidates(cv_embeddings, job_embedding, candidate_ids):
        from sklearn.metrics.pairwise import cosine_similarity
        scores = []
        for cid, emb in zip(candidate_ids, cv_embeddings):
            if emb is None or job_embedding is None:
                scores.append((cid, 0.0))
                continue
            a = _sanitize_vec(emb).reshape(1, -1)
            b = _sanitize_vec(job_embedding).reshape(1, -1)
            if a.shape[1] != b.shape[1]:
                scores.append((cid, 0.0))
                continue
            s = float(cosine_similarity(a, b)[0][0])
            scores.append((cid, s))
        return sorted(scores, key=lambda x: x[1], reverse=True)

# Fallback preprocess_text
try:
    from preprocessing import preprocess_text
except Exception:
    def preprocess_text(t: str) -> str:
        return " ".join((t or "").lower().split())

# Embedder import with fallback
try:
    from embeddings import get_bert_embeddings
except ImportError:
    from embeddings import generate_embeddings
    import pandas as pd

    def get_bert_embeddings(texts: List[str]):
        df_tmp = pd.DataFrame({"_x": texts})
        embs = generate_embeddings(df_tmp, ["_x"])
        return embs["_x"]

# Qdrant storage
cv_storage = CVStorage(host="localhost", port=6333, collection_name="cvs")

def _extract_vector_from_point(p):
    if getattr(p, "vector", None) is not None:
        return p.vector
    vs = getattr(p, "vectors", None)
    if vs is None:
        return None
    if isinstance(vs, dict):
        return vs.get("default") or (next(iter(vs.values())) if vs else None)
    data = getattr(vs, "data", None)
    if isinstance(data, dict):
        return data.get("default") or (next(iter(data.values())) if data else None)
    return None

# Extended skill keywords
SKILL_KEYWORDS = {
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust", "ruby", "php", "swift", "kotlin",
    "sql", "nosql", "mongodb", "postgres", "postgresql", "mysql", "redis", "cassandra", "elasticsearch",
    "docker", "kubernetes", "k8s", "aws", "azure", "gcp", "terraform", "jenkins", "ci/cd", "ansible",
    "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy", "scipy", "nlp", "machine learning", "ml",
    "deep learning", "data science", "data analysis", "computer vision", "transformers",
    "react", "angular", "vue", "node", "nodejs", "django", "flask", "fastapi", "spring", "express",
    "git", "github", "gitlab", "jira", "linux", "unix", "bash", "spark", "hadoop", "kafka",
    "tableau", "powerbi", "matplotlib", "seaborn", "plotly", "d3.js",
    "pytest", "jest", "selenium", "junit", "unit testing", "integration testing",
    "agile", "scrum", "leadership", "team collaboration", "communication", "problem solving"
}

def _extract_skills_from_text(text: str) -> List[str]:
    if not text:
        return []
    text_lower = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        if re.search(rf'\b{re.escape(skill)}\b', text_lower):
            found.append(skill)
    return sorted(set(found))[:20]

def _embed_job_desc(job_desc: str):
    job_desc_clean = preprocess_text(job_desc or "")
    job_vec = _sanitize_vec(get_bert_embeddings([job_desc_clean])[0])
    return job_desc_clean, job_vec

def _vector_search(job_vec, top_k: int):
    print(f"[🔍 Search] Looking for top {top_k} candidates...")
    try:
        hits = cv_storage.client.search(
            collection_name=cv_storage.collection_name,
            query_vector=job_vec.tolist(),
            limit=top_k,
            with_vectors=True,
            with_payload=True,
        )
    except Exception as e:
        print(f"[⚠️ Search] Primary search error: {e}")
        hits = []
    if not hits:
        hits, _ = cv_storage.client.scroll(
            collection_name=cv_storage.collection_name,
            limit=top_k,
            with_vectors=True,
            with_payload=True,
        )
        print(f"[📋 Search] Fallback scroll returned {len(hits)} candidates")
    else:
        print(f"[✓ Search] Found {len(hits)} matching candidates")
    return hits

def _prepare_cvs(hits, job_vec):
    cvs = []
    for h in hits:
        payload = h.payload or {}
        text = payload.get("resume_text") or ""
        vec_raw = _extract_vector_from_point(h)
        if vec_raw is None and text:
            try:
                vec_raw = get_bert_embeddings([text])[0]
            except Exception as e:
                print(f"[⚠️ Prepare] Embedding error for candidate {h.id}: {e}")
                vec_raw = None
        vec = _sanitize_vec(vec_raw) if vec_raw is not None else None
        if vec is None or vec.shape[0] != job_vec.shape[0]:
            print(f"[⚠️ Prepare] Skipping candidate {h.id} - vector mismatch")
            continue
        email = _email_from_payload(payload) or _extract_email(text)
        cvs.append({
            "candidate_id": payload.get("candidate_id", h.id),
            "resume_text": text,
            "vector": vec,
            "email": email,
        })
    print(f"[✓ Prepare] Processed {len(cvs)} valid candidates")
    return cvs

def _normalize_whitespace(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

EMAIL_FALLBACK_DOMAIN = os.getenv("EMAIL_FALLBACK_DOMAIN", "example.com")
EMAIL_FIELD_NAMES = ("email", "Email", "candidate_email", "contact_email")

def _extract_email(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0).strip() if m else None

def _email_from_payload(payload: Dict[str, Any]) -> str | None:
    if not payload:
        return None
    for k in EMAIL_FIELD_NAMES:
        val = payload.get(k)
        if isinstance(val, str) and "@" in val:
            return val.strip()
    return None

def _candidate_name_from_text(text: str) -> str | None:
    if not text:
        return None
    lines = text.strip().splitlines()
    for line in lines[:5]:
        tokens = [t for t in re.split(r"[\s,]+", line) if t]
        caps = [t for t in tokens if re.match(r"^[A-Z][a-zA-Z\-']+$", t)]
        if 1 <= len(caps) <= 3:
            return " ".join(caps[:2])
    return None

def _next_business_day(start: datetime) -> datetime:
    """Get next business day (skip weekends)."""
    d = start
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d

# --- Graph Node Functions (must be defined before build_agent_graph) ---

def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    job_desc = state.get("job_desc", "")
    top_k = int(state.get("top_k", 5))
    
    job_desc_clean, job_vec = _embed_job_desc(job_desc)
    hits = _vector_search(job_vec, top_k)
    cvs = _prepare_cvs(hits, job_vec)
    
    new_state = dict(state)
    new_state["job_desc_clean"] = job_desc_clean
    new_state["job_embedding"] = job_vec
    new_state["cvs"] = cvs
    return new_state

def analyze_node(state: Dict[str, Any]) -> Dict[str, Any]:
    cvs = state.get("cvs", [])
    analyzed = {}
    print(f"[📊 Analyze] Processing {len(cvs)} candidate resumes...")
    for cv in cvs:
        analyzed[cv["candidate_id"]] = analyze_cv(cv["resume_text"])
    new_state = dict(state)
    new_state["analyzed"] = analyzed
    return new_state

def skill_extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    analyzed = state.get("analyzed", {})
    cvs = state.get("cvs", [])
    skill_map = {}
    print(f"[🔧 Skills] Extracting skills from {len(cvs)} candidates...")
    for cv in cvs:
        cid = cv["candidate_id"]
        info = analyzed.get(cid, {})
        skills = info.get("skills") or []
        if not skills:
            skills = _extract_skills_from_text(cv.get("resume_text", ""))
            if cid in analyzed:
                analyzed[cid]["skills"] = skills
        skill_map[cid] = skills
    new_state = dict(state)
    new_state["skills_extracted"] = skill_map
    new_state["analyzed"] = analyzed
    return new_state

def rank_node(state: Dict[str, Any]) -> Dict[str, Any]:
    cvs = state.get("cvs", [])
    job_emb = state.get("job_embedding")
    print(f"[📈 Rank] Ranking {len(cvs)} candidates...")
    ids = [c["candidate_id"] for c in cvs]
    embs = [c["vector"] for c in cvs]
    ranking = rank_candidates(embs, job_emb, ids)
    score_map = {cid: s for cid, s in ranking}
    ranked = [{**c, "score": float(score_map.get(c["candidate_id"], 0.0))} for c in cvs]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    new_state = dict(state)
    new_state["ranked"] = ranked
    return new_state

def explain_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ranked = state.get("ranked", [])
    analyzed = state.get("analyzed", {})
    skill_map = state.get("skills_extracted", {})
    job_desc = state.get("job_desc_clean", "")
    top_k = int(state.get("top_k", 5))
    print(f"[💡 Explain] Generating explanations for top {top_k} candidates...")
    results = []
    for cv in ranked[:top_k]:
        cid = cv["candidate_id"]
        info = analyzed.get(cid, {"full_text": cv.get("resume_text", "")})
        if "full_text" not in info:
            info["full_text"] = cv.get("resume_text", "")
        skills = info.get("skills") or skill_map.get(cid, [])
        try:
            explanation = explain_rankingLLM(info, job_desc)
        except Exception as e:
            explanation = f"Candidate shows relevant experience and skills for this role."
            print(f"[⚠️ Explain] LLM error for candidate {cid}: {e}")
        email = cv.get("email") or _extract_email(info.get("full_text", ""))
        results.append({
            "candidate_id": cid,
            "score": cv["score"],
            "skills": skills,
            "experience": info.get("experience"),
            "explanation": explanation,
            "email": email,
        })
    new_state = dict(state)
    new_state["results"] = results
    return new_state

def calendar_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Create unique interview appointments for each top candidate."""
    results = state.get("results") or []
    if not results:
        ranked = state.get("ranked", [])
        results = [{"candidate_id": c.get("candidate_id"), "resume_text": c.get("resume_text", "")} for c in ranked]

    if not results:
        print("[📅 Calendar] No candidates to schedule")
        new_state = dict(state)
        new_state["appointments"] = {}
        return new_state

    print(f"[📅 Calendar] Scheduling interviews for {len(results)} candidates...")
    
    # Get custom start date from state, or use default
    custom_start_date = state.get("schedule_start_date")
    custom_start_time = state.get("schedule_start_time", "10:00")
    slot_duration = state.get("schedule_duration", 30)  # minutes
    
    if custom_start_date:
        try:
            start = datetime.strptime(custom_start_date, "%Y-%m-%d")
            print(f"[📅 Calendar] Using custom start date: {custom_start_date}")
        except:
            start = _next_business_day(datetime.now() + timedelta(days=1))
    else:
        start = _next_business_day(datetime.now() + timedelta(days=1))
    
    # Parse start time
    try:
        hour, minute = map(int, custom_start_time.split(':'))
        slot_time = start.replace(hour=hour, minute=minute, second=0, microsecond=0)
    except:
        slot_time = start.replace(hour=10, minute=0, second=0, microsecond=0)
    
    slot_delta = timedelta(minutes=slot_duration)

    appointments: Dict[Any, Dict[str, str]] = {}
    for i, r in enumerate(results):
        cid = r.get("candidate_id")
        if cid is None:
            continue
        t = slot_time + i * slot_delta
        date_str = t.strftime("%Y-%m-%d")
        time_str = t.strftime("%H:%M")
        link = f"https://meet.company.com/{cid}"
        appointments[cid] = {"date": date_str, "time": time_str, "meeting_link": link}
        print(f"[📅 Calendar] Scheduled candidate {cid} → {date_str} at {time_str}")

    new_state = dict(state)
    new_state["appointments"] = appointments
    _save_named_state("last_run_with_calendar", new_state)
    return new_state

def email_sender_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Send interview invitation emails to top candidates."""
    results = state.get("results") or []
    ranked = state.get("ranked") or []
    top_k = int(state.get("top_k", 5))
    job_desc_clean = state.get("job_desc_clean", state.get("job_desc", ""))

    if not results and ranked:
        results = []
        for cv in ranked[:top_k]:
            email_guess = cv.get("email") or _extract_email(cv.get("resume_text", ""))
            results.append({
                "candidate_id": cv.get("candidate_id"),
                "score": cv.get("score", 0.0),
                "explanation": cv.get("explanation", ""),
                "email": email_guess,
                "resume_text": cv.get("resume_text", ""),
            })

    if not results:
        summary = "❌ No candidate list available. Please run a candidate search first."
        print(f"[📧 Email] {summary}")
        return {"email_summary": summary, "emailed": [], "emailed_entries": []}

    print(f"[📧 Email] Preparing to send invitations to {len(results[:top_k])} candidates...")
    emailed_ids: List[Any] = []
    emailed_entries: List[Dict[str, Any]] = []
    by_id = {cv.get("candidate_id"): cv for cv in ranked} if ranked else {}
    subject = render_invite_subject(None)
    reply_to = os.getenv("REPLY_TO") or os.getenv("FROM_EMAIL") or os.getenv("SMTP_USER")
    appointments = state.get("appointments", {})

    for r in results[:top_k]:
        cid = r.get("candidate_id")
        if cid is None:
            continue
        email = r.get("email")
        if not email and cid in by_id:
            email = by_id[cid].get("email")
        if not email:
            analyzed = state.get("analyzed", {})
            info = analyzed.get(cid, {})
            email = _extract_email(info.get("full_text", "")) or _extract_email(r.get("explanation", ""))
        if not email:
            try:
                email = cv_storage.get_email_by_candidate_id(cid)
            except Exception:
                email = None

        if email and send_email is not None:
            name_source = r.get("resume_text") or by_id.get(cid, {}).get("resume_text", "")
            name = _candidate_name_from_text(name_source)
            body = render_invite_body(name, job_desc_clean, reply_to)
            appt = appointments.get(cid)
            if isinstance(appt, dict):
                date = appt.get("date")
                time = appt.get("time")
                link = appt.get("meeting_link")
                if date and time and link:
                    body += f"\n\n📅 Interview Details:\nDate: {date}\nTime: {time}\nMeeting Link: {link}"
            try:
                ok = send_email(email, subject, body, reply_to=reply_to)
            except Exception as e:
                print(f"[❌ Email] Failed to send to candidate {cid} <{email}>: {e}")
                ok = False
            status = "sent" if ok else "failed"
        else:
            status = "skipped" if email else "no_email"

        if not email:
            email = "unknown"
        status_icon = "✓" if status == "sent" else "⚠️" if status == "failed" else "⊘"
        print(f"[{status_icon} Email] {status.upper()} → Candidate {cid} <{email}>")
        emailed_ids.append(cid)
        emailed_entries.append({"candidate_id": cid, "email": email, "status": status})

    sent_count = sum(1 for e in emailed_entries if e.get("status") == "sent")
    summary = f"✓ Successfully sent {sent_count} out of {len(emailed_ids)} interview invitations."
    return {"email_summary": summary, "emailed": emailed_ids, "emailed_entries": emailed_entries}

# --- State Management ---
_last_state: Dict[str, Any] | None = None
_last_state_with_calendar: Dict[str, Any] | None = None

def _save_last_state(state: Dict[str, Any]):
    global _last_state
    _last_state = state
    results_count = len(state.get("results", []))
    
    # Save to file for persistence across reloads
    try:
        # Convert to JSON-serializable format (exclude numpy arrays)
        serializable_state = {}
        for key, value in state.items():
            if key == "job_embedding":
                continue  # Skip numpy arrays
            elif isinstance(value, list):
                serializable_state[key] = []
                for item in value:
                    if isinstance(item, dict):
                        clean_item = {k: v for k, v in item.items() if k != "vector"}
                        serializable_state[key].append(clean_item)
                    else:
                        serializable_state[key].append(item)
            else:
                serializable_state[key] = value
        
        with open(STATE_FILE, 'w') as f:
            json.dump(serializable_state, f)
        print(f"[💾 State] Saved {results_count} candidates to memory AND file")
    except Exception as e:
        print(f"[⚠️ State] File save failed: {e}, using memory only")


def _load_last_state() -> Dict[str, Any] | None:
    global _last_state
    
    # Try memory first
    if _last_state is not None:
        results_count = len(_last_state.get("results", []))
        print(f"[📂 State] Loaded from memory: {results_count} candidates")
        return _last_state
    
    # Try file persistence
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            results_count = len(state.get("results", []))
            print(f"[📂 State] Loaded from file: {results_count} candidates")
            _last_state = state
            return state
    except Exception as e:
        print(f"[⚠️ State] File load failed: {e}")
    
    print("[⚠️ State] No saved state found!")
    return None


def _save_named_state(name: str, state: Dict[str, Any]):
    global _last_state_with_calendar
    
    if name == "last_run_with_calendar":
        _last_state_with_calendar = state
        
        # Save to file
        try:
            serializable_state = {}
            for key, value in state.items():
                if key == "job_embedding":
                    continue
                elif isinstance(value, list):
                    serializable_state[key] = []
                    for item in value:
                        if isinstance(item, dict):
                            clean_item = {k: v for k, v in item.items() if k != "vector"}
                            serializable_state[key].append(clean_item)
                        else:
                            serializable_state[key].append(item)
                else:
                    serializable_state[key] = value
            
            with open(CALENDAR_STATE_FILE, 'w') as f:
                json.dump(serializable_state, f)
            print(f"[💾 State] Saved '{name}' to memory AND file")
        except Exception as e:
            print(f"[⚠️ State] File save failed for '{name}': {e}")
    else:
        print(f"[💾 State] Saved '{name}' in memory only")


# --- Graph Building ---
def build_agent_graph():
    if StateGraph is None or END is None:
        raise RuntimeError("LangGraph not available")
    graph = StateGraph(dict)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("skillExtractor", skill_extractor_node)
    graph.add_node("rank", rank_node)
    graph.add_node("explain", explain_node)
    graph.add_node("emailSender", email_sender_node)
    graph.add_node("calendar", calendar_node)
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "skillExtractor")
    graph.add_edge("skillExtractor", "rank")
    graph.add_edge("rank", "explain")
    graph.add_edge("explain", END)
    graph.set_entry_point("retrieve")
    return graph.compile()

try:
    _compiled_graph = build_agent_graph()
    print("[✓ Graph] Agent graph compiled successfully")
except Exception as e:
    _compiled_graph = None
    print(f"[⚠️ Graph] Failed to compile: {e}")

# --- Main Run Functions ---
def run_agent(job_desc: str, top_k: int = 5, with_explain: bool = False) -> List[Dict[str, Any]]:
    print(f"[🚀 Agent] Starting search for top {top_k} candidates...")
    
    if _compiled_graph is None:
        return run_agent_simple(job_desc, top_k=top_k, with_explain=with_explain)

    out = _compiled_graph.invoke({"job_desc": job_desc, "top_k": top_k})
    if not isinstance(out, dict):
        print(f"[⚠️ Agent] Unexpected output type: {type(out)}")
        return run_agent_simple(job_desc, top_k=top_k, with_explain=with_explain)

    res = out.get("results")
    if res:
        _save_last_state(out)
        print(f"[✓ Agent] Successfully retrieved {len(res)} candidates")
        return res
    
    ranked = out.get("ranked", [])
    analyzed = out.get("analyzed", {})
    job_desc_clean = out.get("job_desc_clean", "")
    if ranked:
        results = []
        for cv in ranked[:top_k]:
            info = analyzed.get(cv["candidate_id"], {"full_text": cv.get("resume_text", "")})
            if "full_text" not in info:
                info["full_text"] = cv.get("resume_text", "")
            try:
                explanation = explain_rankingLLM(info, job_desc_clean)
            except Exception:
                explanation = "Strong candidate profile matching the role requirements."
            email = cv.get("email") or _extract_email(info.get("full_text", ""))
            results.append({
                "candidate_id": cv["candidate_id"],
                "score": cv["score"],
                "skills": info.get("skills", []),
                "experience": info.get("experience"),
                "explanation": explanation,
                "email": email,
            })
        new_state = dict(out)
        new_state["results"] = results
        _save_last_state(new_state)
        print(f"[✓ Agent] Built {len(results)} results from ranked candidates")
        return results

    return run_agent_simple(job_desc, top_k=top_k, with_explain=False)

def run_agent_simple(job_desc: str, top_k: int = 5, with_explain: bool = False) -> List[Dict[str, Any]]:
    print(f"[🔄 Simple] Running simple agent mode with top_k={top_k}")
    job_desc_clean, job_vec = _embed_job_desc(job_desc)
    top_k = min(max(top_k, 1), 100)
    hits = _vector_search(job_vec, top_k)
    cvs = _prepare_cvs(hits, job_vec)
    results = []
    for cv in cvs:
        info = analyze_cv(cv["resume_text"])
        try:
            explanation = explain_rankingLLM(info, job_desc_clean)
        except Exception:
            explanation = "Candidate profile matches role requirements."
        email = cv.get("email") or _extract_email(cv.get("resume_text", ""))
        results.append({
            "candidate_id": cv["candidate_id"],
            "score": cv.get("score", 0.0),
            "skills": info.get("skills", []),
            "experience": info.get("experience"),
            "explanation": explanation,
            "email": email,
        })
    _save_last_state({"results": results, "top_k": top_k, "job_desc_clean": job_desc_clean})
    return results

def run_agent_email() -> str:
    print("[📧 Email] Loading previous candidate search...")
    state = _load_last_state()
    if not state:
        return "❌ No previous candidate list found. Please run a search first."
    email_out = email_sender_node(state)
    merged_state = dict(state)
    merged_state.update(email_out)
    _save_last_state(merged_state)
    entries = email_out.get("emailed_entries", [])
    if not entries:
        return email_out.get("email_summary", "✓ Email process completed.")
    lines = [email_out.get("email_summary", "✓ Email process completed.")]
    for e in entries:
        status_icon = "✓" if e.get("status") == "sent" else "⚠️"
        lines.append(f"{status_icon} Candidate {e.get('candidate_id')} <{e.get('email')}>")
    return "\n".join(lines)

def run_agent_schedule() -> str:
    print("[📅 Schedule] Generating interview schedule...")
    state = _load_last_state()
    
    if not state:
        return "❌ No previous candidate list found. Please run a search first."
    
    if not state.get("results") and not state.get("ranked"):
        return "❌ No candidates found in previous search. Please search for candidates first."
    
    with_calendar = calendar_node(state)
    email_out = email_sender_node(with_calendar)
    merged = dict(with_calendar)
    merged.update(email_out)
    _save_named_state("last_run_with_calendar", merged)
    
    entries = email_out.get("emailed_entries", [])
    count = sum(1 for e in entries if e.get("status") == "sent") if entries else 0
    
    return f"✓ Interview invitations with calendar links sent to {count} candidates."
