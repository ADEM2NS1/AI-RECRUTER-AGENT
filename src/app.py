import streamlit as st
import requests
import os
import re
from io import BytesIO
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import time
import calendar

# API URL config
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")
INTERVIEWS_FILE = Path(tempfile.gettempdir()) / "ai_recruiter_interviews.json"

# --------------- Helper Functions ---------------
def load_interviews_from_file():
    try:
        if INTERVIEWS_FILE.exists():
            with open(INTERVIEWS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"[⚠️] Failed to load interviews: {e}")
    return []

def save_interviews_to_file(interviews):
    try:
        with open(INTERVIEWS_FILE, 'w') as f:
            json.dump(interviews, f, indent=2)
        return True
    except Exception as e:
        print(f"[⚠️] Failed to save interviews: {e}")
        return False

def parse_candidates(text: str) -> list:
    """Parse candidates - handles both numeric IDs and hex IDs"""
    candidates = []
    
    # Strategy: Split by numbered list items "1. ", "2. ", etc.
    parts = re.split(r'\n(\d+)\.\s+', text)
    
    # parts[0] is intro text, then alternates: [number, content, number, content, ...]
    i = 1
    while i < len(parts) - 1:
        rank = parts[i]  # The rank number (1, 2, 3, etc.)
        content = parts[i + 1]  # The content after the number
        
        # Try to extract candidate ID and explanation
        # Format 1: "212: explanation..." (numeric ID)
        # Format 2: "Candidate 212: explanation..."
        # Format 3: "8e0abbc7: explanation..." (hex ID)
        match = re.match(r'(?:[Cc]andidate\s+)?([a-zA-Z0-9]+):\s*(.*)', content, re.DOTALL)
        
        if match:
            cid, explanation = match.groups()
            
            # Skip candidates with "Insufficient data" - these are invalid
            if "Insufficient data" in explanation or len(explanation.strip()) < 20:
                print(f"[DEBUG] Skipping candidate {rank} ({cid}) - insufficient data")
                i += 2
                continue
            
            # Extract skills
            skills = []
            skill_keywords = [
                'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#',
                'ML', 'Machine Learning', 'NLP', 'Natural Language Processing',
                'TensorFlow', 'PyTorch', 'Keras', 'scikit-learn',
                'Deep Learning', 'Data Science', 'Data Analysis',
                'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL',
                'React', 'Angular', 'Vue', 'Node', 'Django', 'Flask',
                'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes'
            ]
            
            for skill in skill_keywords:
                if re.search(r'\b' + re.escape(skill) + r'\b', explanation, re.IGNORECASE):
                    if skill not in skills:
                        skills.append(skill)
            
            # Clean explanation
            clean_exp = explanation.strip()
            clean_exp = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_exp)  # Remove markdown bold
            clean_exp = re.sub(r'\s+', ' ', clean_exp)  # Normalize whitespace
            clean_exp = clean_exp.strip()
            
            candidates.append({
                'rank': int(rank),
                'id': str(cid),
                'explanation': clean_exp,
                'preview': clean_exp[:200] + "..." if len(clean_exp) > 200 else clean_exp,
                'skills': skills[:5]
            })
            
            print(f"[DEBUG] Added candidate {rank} - ID: {cid}")
        else:
            print(f"[DEBUG] Could not parse rank {rank}, content: {content[:100]}")
        
        i += 2  # Move to next number
    
    print(f"[DEBUG] Total candidates found: {len(candidates)} - Ranks: {[c['rank'] for c in candidates]}")
    
    return candidates

def sanitize_html(text: str) -> str:
    if not text:
        return ""
    return (text.replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;'))

def format_message_content(content: str, role: str, message_index: int) -> str:
    if role == "agent" and "Here are the top recommendations:" in content:
        candidates = parse_candidates(content)
        if candidates:
            st.session_state[f'candidates_{message_index}'] = candidates
            return "CANDIDATES_DISPLAY"
    
    if role == "agent" and ("Interview Schedule:" in content or "Interview invitations" in content or "calendar links sent" in content):
        try:
            response = requests.get(f"{API_URL.replace('/chat', '')}/get_calendar", timeout=5)
            if response.status_code == 200:
                data = response.json()
                interviews = data.get("interviews", [])
                if interviews:
                    st.session_state.scheduled_interviews = interviews
                    save_interviews_to_file(interviews)
        except Exception as e:
            print(f"[⚠️] Failed to get calendar: {e}")
    
    sanitized = sanitize_html(content)
    return sanitized.replace('\n', '<br>')

# --------------- Page Config ---------------
st.set_page_config(page_title="AI Recruiting Assistant", page_icon="🎯", layout="wide")

# --------------- ORIGINAL CSS WITH PURPLE-BLUE GRADIENT ---------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ORIGINAL PURPLE-BLUE GRADIENT BACKGROUND */
.main { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    padding: 1rem 0; 
}

.block-container { 
    max-width: 1400px; 
    padding: 1rem 2rem; 
}

/* Header with gradient text */
.header-container {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    text-align: center;
}

/* Candidate Cards */
.candidate-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.candidate-card:hover { 
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}

.candidate-score {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.skills-tag {
    display: inline-block;
    background: #ede9fe;
    color: #7c3aed;
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    font-size: 0.85rem;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

/* Buttons with purple gradient */
.stButton button {
    border-radius: 12px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    transition: all 0.2s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* Calendar day buttons with gradient */
.interview-day-btn button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0 !important;
    margin: 0 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    height: 45px !important;
    width: 100% !important;
}

.interview-day-btn button:hover {
    background: linear-gradient(135deg, #7c8ef0 0%, #8a5bb8 100%) !important;
    transform: scale(1.05) !important;
}

/* Input styling */
.stTextInput input, .stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #e2e8f0;
    padding: 0.75rem;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* Navigation tabs */
.nav-logo {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------- Session State ---------------
if "page" not in st.session_state:
    st.session_state.page = "chat"

if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "agent", "content": "Hi! I'm your AI Recruiting Assistant. How can I help you today?"}]

if "max_candidates" not in st.session_state:
    st.session_state.max_candidates = 5

if "scheduled_interviews" not in st.session_state:
    st.session_state.scheduled_interviews = load_interviews_from_file()

if "expanded_candidates" not in st.session_state:
    st.session_state.expanded_candidates = set()

if "calendar_date" not in st.session_state:
    st.session_state.calendar_date = datetime.now()

if "selected_date" not in st.session_state:
    st.session_state.selected_date = None

# --------------- Header ---------------
st.markdown('''
<div class="header-container">
    <h1 class="header-title">🎯 AI Recruiting Assistant</h1>
    <p style="text-align:center;color:#64748b;font-size:1.1rem;margin-top:0.5rem;">Find the perfect candidates with AI-powered search</p>
</div>
''', unsafe_allow_html=True)

# --------------- Navigation ---------------
nav_cols = st.columns(5)

with nav_cols[0]:
    if st.button("💬 Chat & Schedule", key="nav_chat", use_container_width=True):
        st.session_state.page = "chat"
        st.rerun()

with nav_cols[1]:
    if st.button("📤 Upload CV", key="nav_upload", use_container_width=True):
        st.session_state.page = "upload"
        st.rerun()

with nav_cols[2]:
    if st.button("🎯 Score CV", key="nav_score", use_container_width=True):
        st.session_state.page = "score"
        st.rerun()

with nav_cols[3]:
    if st.button("📅 Calendar", key="nav_calendar", use_container_width=True):
        st.session_state.page = "calendar"
        st.rerun()

with nav_cols[4]:
    if st.button("⚙️ Settings", key="nav_settings", use_container_width=True):
        st.session_state.page = "settings"
        st.rerun()

st.markdown("---")

# =============== CHAT PAGE WITH SIDEBAR ===============
if st.session_state.page == "chat":
    col_main, col_sidebar = st.columns([2, 1])
    
    with col_main:
        st.title("💬 AI Chat Assistant")
        
        # Chat messages
        for idx, m in enumerate(st.session_state.chat):
            role = m.get("role")
            content = m.get("content", "")
            formatted_content = format_message_content(content, role, idx)
            
            if formatted_content == "CANDIDATES_DISPLAY":
                st.markdown('<div style="background:#f0fdf4;padding:1rem;border-radius:10px;margin:1rem 0;border-left:3px solid #10b981;"><strong>🤖 AI Assistant</strong></div>', unsafe_allow_html=True)
                candidates = st.session_state.get(f'candidates_{idx}', [])
                st.markdown('<h3 style="color:#1e293b;">🎯 Top Candidates</h3>', unsafe_allow_html=True)
                
                for candidate in candidates:
                    candidate_key = f"{idx}_{candidate['id']}"
                    is_expanded = candidate_key in st.session_state.expanded_candidates
                    
                    html = f'''
                    <div class="candidate-card">
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                            <div style="font-size:1.1rem;font-weight:600;">#{candidate['rank']} - Candidate {candidate['id']}</div>
                            <div class="candidate-score">Match: {100 - candidate['rank'] * 5}%</div>
                        </div>
                        <div style="color:#475569;line-height:1.6;margin-bottom:0.75rem;">
                            {sanitize_html(candidate['explanation'] if is_expanded else candidate['preview'])}
                        </div>
                    '''
                    
                    if candidate['skills']:
                        html += '<div>'
                        for skill in candidate['skills']:
                            html += f'<span class="skills-tag">{sanitize_html(skill)}</span>'
                        html += '</div>'
                    
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
                    
                    if st.button("Hide ▲" if is_expanded else "View Details ▼", key=f"btn_{candidate_key}"):
                        if is_expanded:
                            st.session_state.expanded_candidates.remove(candidate_key)
                        else:
                            st.session_state.expanded_candidates.add(candidate_key)
                        st.rerun()
            
            elif role == "user":
                st.markdown(f'<div style="background:#f0f9ff;padding:1rem;border-radius:10px;margin:1rem 0;border-left:3px solid #0ea5e9;"><strong>👤 You:</strong><br>{sanitize_html(content)}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background:#f0fdf4;padding:1rem;border-radius:10px;margin:1rem 0;border-left:3px solid #10b981;"><strong>🤖 AI:</strong><br>{formatted_content}</div>', unsafe_allow_html=True)
        
        # Input area with better alignment
        st.markdown("---")
        st.markdown("### 💬 Send Message")

        uploaded_pdf = st.file_uploader("📄 Upload Job Description PDF (optional)", type=["pdf"], key="chat_pdf")

        # Custom HTML/CSS for perfect alignment
        st.markdown("""
        <style>
        .input-row {
            display: flex;
            gap: 0.5rem;
            align-items: stretch;
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([6, 1], gap="small")
        with col1:
            st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True) 
            user_input = st.text_input(
                "Type your message", 
                placeholder="e.g., Find data scientists with Python and ML experience...", 
                key="chat_input",
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True)  # Small spacer
            send_btn = st.button("Send 📤", key="send_chat", use_container_width=True, type="primary")

        
        if send_btn:
            msg = (user_input or "").strip()
            if msg or uploaded_pdf:
                display_msg = msg if msg else "📄 PDF uploaded"
                st.session_state.chat.append({"role": "user", "content": display_msg})
                
                file_bytes = uploaded_pdf.read() if uploaded_pdf else None
                filename = uploaded_pdf.name if uploaded_pdf else None
                
                with st.spinner("Processing..."):
                    try:
                        if file_bytes:
                            files = {"file": (filename, BytesIO(file_bytes), "application/pdf")}
                            data = {"message": msg, "max_candidates": str(st.session_state.max_candidates)}
                            response = requests.post(API_URL, files=files, data=data, timeout=240)
                        else:
                            response = requests.post(API_URL, json={"message": msg, "max_candidates": st.session_state.max_candidates}, timeout=180)
                        
                        if response.status_code == 200:
                            reply = response.json().get("reply", response.text)
                            st.session_state.chat.append({"role": "agent", "content": reply})
                        else:
                            st.session_state.chat.append({"role": "agent", "content": f"Error: {response.status_code}"})
                    except Exception as e:
                        st.session_state.chat.append({"role": "agent", "content": f"Error: {e}"})
                
                st.rerun()
            else:
                st.warning("Please type a message or upload a PDF")
        
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat = [{"role": "agent", "content": "Chat cleared! How can I help?"}]
            st.session_state.expanded_candidates = set()
            st.rerun()
    
    with col_sidebar:
        st.markdown("### 📆 Interview Scheduling")
        
        min_date = datetime.now().date() + timedelta(days=1)
        interview_date = st.date_input("📅 Date", value=min_date, min_value=min_date, key="sched_date")
        
        col1, col2 = st.columns(2)
        with col1:
            start_hour = st.selectbox("🕐 Hour", options=list(range(8, 19)), index=2, key="sched_hour")
        with col2:
            start_minute = st.selectbox("⏱️ Min", options=[0, 15, 30, 45], index=0, key="sched_min")
        
        duration = st.selectbox("⏳ Duration (min)", options=[15, 30, 45, 60, 90], index=1, key="sched_duration")
        
        st.info(f"📍 {interview_date.strftime('%b %d')} at {start_hour:02d}:{start_minute:02d} • {duration} min")
        
        if st.button("🗓️ Schedule Interviews", use_container_width=True, type="primary", key="do_schedule"):
            with st.spinner("Scheduling..."):
                try:
                    response = requests.post(
                        f"{API_URL.replace('/chat', '')}/schedule_with_config",
                        data={
                            "start_date": interview_date.strftime("%Y-%m-%d"),
                            "start_time": f"{start_hour:02d}:{start_minute:02d}",
                            "duration": duration,
                            "append": "true"
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            st.success(result.get("message"))
                            st.session_state.scheduled_interviews = result.get("interviews", [])
                            save_interviews_to_file(st.session_state.scheduled_interviews)
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(result.get("message"))
                    else:
                        st.error("Failed to schedule")
                except Exception as e:
                    st.error(f"Error: {e}")

# =============== UPLOAD CV PAGE ===============
elif st.session_state.page == "upload":
    st.title("📤 Upload Candidate CV")
    
    
    uploaded_cv = st.file_uploader("Choose CV (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"], key="upload_cv")
    
    with st.expander("➕ Add Candidate Info (Optional)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            cv_name = st.text_input("Name", placeholder="John Doe", key="cv_name")
        with col2:
            cv_email = st.text_input("Email", placeholder="john@example.com", key="cv_email")
        with col3:
            cv_phone = st.text_input("Phone", placeholder="+1234567890", key="cv_phone")
    
    if st.button("📤 Upload to Database", type="primary", use_container_width=True, key="do_upload"):
        if uploaded_cv:
            with st.spinner("Processing CV..."):
                try:
                    files = {"file": (uploaded_cv.name, uploaded_cv.getvalue(), uploaded_cv.type)}
                    data = {"candidate_name": cv_name, "candidate_email": cv_email, "candidate_phone": cv_phone}
                    
                    response = requests.post(f"{API_URL.replace('/chat', '')}/upload_cv", files=files, data=data, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            st.success(result.get("message"))
                            
                            # ORIGINAL GRADIENT CARD
                            st.markdown(f"""
                            <div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:1.5rem;border-radius:15px;margin:1rem 0;'>
                                <h3 style='margin:0 0 1rem 0;'>✓ Candidate Added to Database</h3>
                                <div style='display:grid;grid-template-columns:1fr 1fr;gap:1rem;'>
                                    <div>
                                        <p style='margin:0.5rem 0;'><strong>ID:</strong> {result.get('candidate_id')}</p>
                                        <p style='margin:0.5rem 0;'><strong>Name:</strong> {result.get('name')}</p>
                                        <p style='margin:0.5rem 0;'><strong>Email:</strong> {result.get('email')}</p>
                                    </div>
                                    <div>
                                        <p style='margin:0.5rem 0;'><strong>Phone:</strong> {result.get('phone')}</p>
                                        <p style='margin:0.5rem 0;'><strong>Experience:</strong> {result.get('experience',0)} years</p>
                                        <p style='margin:0.5rem 0;'><strong>Embedding:</strong> {result.get('embedding_dimension')}D vector</p>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if result.get('skills'):
                                st.markdown("**🎯 Detected Skills:**")
                                skills_html = " ".join([f"<span class='skills-tag'>{skill}</span>" for skill in result.get('skills', [])[:15]])
                                st.markdown(f"<div>{skills_html}</div>", unsafe_allow_html=True)
                        else:
                            st.error(result.get("message"))
                    else:
                        st.error("Upload failed")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please select a file")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============== SCORE CV PAGE ===============
elif st.session_state.page == "score":
    st.title("🎯 Score CV Against Job")
    
    
    score_cv = st.file_uploader("Choose CV", type=["pdf", "png", "jpg", "jpeg"], key="score_cv")
    job_desc = st.text_area("Job Description", height=150, placeholder="Enter job requirements...", key="score_job")
    
    if st.button("🎯 Calculate Score", type="primary", use_container_width=True, key="do_score"):
        if score_cv and job_desc:
            with st.spinner("Analyzing..."):
                try:
                    files = {"file": (score_cv.name, score_cv.getvalue(), score_cv.type)}
                    data = {"job_description": job_desc}
                    
                    response = requests.post(f"{API_URL.replace('/chat', '')}/score_cv", files=files, data=data, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            score = result.get("score", 0)
                            color = "#10b981" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
                            emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                            
                            st.markdown(f"""
                            <div style='background:{color};color:white;padding:2rem;border-radius:15px;text-align:center;margin:1rem 0;'>
                                <h1 style='margin:0;font-size:3.5rem;'>{emoji} {score}%</h1>
                                <h2 style='margin:0.5rem 0 0 0;'>{result.get('match_level')}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.info(f"**💡 Recommendation:** {result.get('recommendation')}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**✅ Matched Skills**")
                                for skill in result.get('matched_skills', []):
                                    st.markdown(f"<div style='background:#d1fae5;color:#065f46;padding:0.5rem;border-radius:8px;margin:0.25rem 0;'>✓ {skill}</div>", unsafe_allow_html=True)
                            with col2:
                                st.markdown("**❌ Missing Skills**")
                                for skill in result.get('missing_skills', []):
                                    st.markdown(f"<div style='background:#fee2e2;color:#991b1b;padding:0.5rem;border-radius:8px;margin:0.25rem 0;'>✗ {skill}</div>", unsafe_allow_html=True)
                            
                            st.markdown("---")
                            details = result.get('details', {})
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Experience", f"{result.get('experience_years', 0)} yrs")
                            with col2:
                                st.metric("Skill Match", f"{details.get('skill_match_percentage', 0)}%")
                            with col3:
                                st.metric("Skills", f"{details.get('matched_skills_count', 0)}/{details.get('total_required_skills', 0)}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Provide both CV and job description")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============== CALENDAR PAGE WITH IMPROVED LAYOUT ===============
elif st.session_state.page == "calendar":
    st.title("📅 Interview Calendar")
    
    # Top controls
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh Calendar", use_container_width=True):
            try:
                response = requests.get(f"{API_URL.replace('/chat', '')}/get_calendar", timeout=5)
                if response.status_code == 200:
                    st.session_state.scheduled_interviews = response.json().get("interviews", [])
                    save_interviews_to_file(st.session_state.scheduled_interviews)
                    st.rerun()
            except:
                pass
    
    with col2:
        if st.session_state.scheduled_interviews:
            st.markdown(f"**{len(st.session_state.scheduled_interviews)} interviews scheduled**")
        else:
            st.info("📅 No interviews scheduled yet")
    
    if st.session_state.scheduled_interviews:
        # Main layout: 70% calendar, 30% interview list
        col_calendar, col_interviews = st.columns([6, 4])
        
        # LEFT COLUMN - CALENDAR (70%)
        with col_calendar:
            
            interviews_by_date = {}
            for interview in st.session_state.scheduled_interviews:
                date = interview.get('date')
                if date not in interviews_by_date:
                    interviews_by_date[date] = []
                interviews_by_date[date].append(interview)
            
            current_date = st.session_state.calendar_date
            
            # Navigation
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("◀", key="cal_prev", use_container_width=True):
                    if current_date.month == 1:
                        st.session_state.calendar_date = current_date.replace(year=current_date.year - 1, month=12, day=1)
                    else:
                        st.session_state.calendar_date = current_date.replace(month=current_date.month - 1, day=1)
                    st.rerun()
            with col2:
                st.markdown(f"<div style='text-align:center;font-weight:600;color:#667eea;font-size:1.5rem;'>{current_date.strftime('%B %Y')}</div>", unsafe_allow_html=True)
            with col3:
                if st.button("▶", key="cal_next", use_container_width=True):
                    if current_date.month == 12:
                        st.session_state.calendar_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
                    else:
                        st.session_state.calendar_date = current_date.replace(month=current_date.month + 1, day=1)
                    st.rerun()
            
            cal = calendar.monthcalendar(current_date.year, current_date.month)
            
            # Day names
            day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            cols = st.columns(7)
            for i, day_name in enumerate(day_names):
                cols[i].markdown(f"<div style='text-align:center;font-weight:600;color:#667eea;'>{day_name}</div>", unsafe_allow_html=True)
            
            # Calendar grid
            for week in cal:
                cols = st.columns(7)
                for i, day in enumerate(week):
                    with cols[i]:
                        if day == 0:
                            st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)
                        else:
                            date_str = f"{current_date.year:04d}-{current_date.month:02d}-{day:02d}"
                            has_interview = date_str in interviews_by_date
                            
                            if has_interview:
                                st.markdown('<div class="interview-day-btn">', unsafe_allow_html=True)
                                if st.button(f"{day}", key=f"cal_day_{date_str}", use_container_width=True):
                                    st.session_state.selected_date = date_str if st.session_state.get('selected_date') != date_str else None
                                    st.rerun()
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.markdown(f"<div style='text-align:center;margin-top:-8px;'><span style='background:#10b981;color:white;padding:2px 8px;border-radius:8px;font-size:0.7rem;font-weight:600;'>{len(interviews_by_date[date_str])}</span></div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='text-align:center;padding:15px;color:#64748b;font-size:0.9rem;'>{day}</div>", unsafe_allow_html=True)
            
            # Export and Clear buttons at bottom of calendar
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All", use_container_width=True, key="clear_all_cal"):
                    st.session_state.scheduled_interviews = []
                    save_interviews_to_file([])
                    st.session_state.selected_date = None
                    st.rerun()
            with col2:
                ics_content = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//AI Recruiter//EN\n"
                for interview in st.session_state.scheduled_interviews:
                    date_str = interview['date'].replace('-', '')
                    time_str = interview['time'].replace(':', '')
                    ics_content += f"""BEGIN:VEVENT
SUMMARY:Interview - Candidate {interview['candidate_id']}
DTSTART:{date_str}T{time_str}00
DURATION:PT30M
DESCRIPTION:Interview with {interview.get('email', 'candidate')}
END:VEVENT
"""
                ics_content += "END:VCALENDAR"
                
                st.download_button("📥 Export", data=ics_content, file_name="interviews.ics", mime="text/calendar", use_container_width=True, key="export_cal")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # RIGHT COLUMN - INTERVIEW LIST (30%)
        with col_interviews:
            
            # Show selected date interviews OR all upcoming interviews
            if st.session_state.selected_date and st.session_state.selected_date in interviews_by_date:
                formatted_date = datetime.strptime(st.session_state.selected_date, '%Y-%m-%d').strftime('%B %d, %Y')
                st.markdown(f"### 📍 {formatted_date}")
                st.markdown(f"**{len(interviews_by_date[st.session_state.selected_date])} interview(s)**")
                st.markdown("---")
                
                for interview in interviews_by_date[st.session_state.selected_date]:
                    st.markdown(f"""
                    <div style='background:linear-gradient(135deg,#f0f9ff 0%,#e0f2fe 100%);
                    padding:1rem;border-radius:12px;margin-bottom:1rem;border-left:3px solid #0ea5e9;'>
                        <div style='font-weight:600;color:#0369a1;font-size:1.1rem;margin-bottom:0.5rem;'>🕒 {interview['time']}</div>
                        <div style='color:#0c4a6e;margin-bottom:0.25rem;'>👤 <strong>Candidate {interview['candidate_id']}</strong></div>
                        <div style='color:#64748b;font-size:0.85rem;word-break:break-all;'>📧 {interview.get('email','N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("### 📋 All Interviews")
                st.markdown(f"**{len(st.session_state.scheduled_interviews)} total**")
                st.markdown("---")
                
                # Group by date
                for date in sorted(interviews_by_date.keys()):
                    formatted_date = datetime.strptime(date, '%Y-%m-%d').strftime('%b %d, %Y')
                    st.markdown(f"**📅 {formatted_date}**")
                    
                    for interview in interviews_by_date[date]:
                        st.markdown(f"""
                        <div style='background:#f8fafc;padding:0.75rem;border-radius:8px;margin:0.5rem 0;border-left:2px solid #667eea;'>
                            <div style='font-size:0.85rem;color:#475569;'><strong>{interview['time']}</strong> - Candidate {interview['candidate_id']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:white;border-radius:20px;padding:3rem;text-align:center;box-shadow:0 10px 40px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
        st.markdown("### 📅 No Interviews Scheduled")
        st.markdown("Start by searching for candidates in the Chat page, then schedule interviews!")
        st.markdown('</div>', unsafe_allow_html=True)


# =============== SETTINGS PAGE ===============
elif st.session_state.page == "settings":
    st.title("⚙️ Settings")
    
    
    st.markdown("### API Configuration")
    api_url = st.text_input("API URL", value=API_URL, key="settings_api_url")
    
    st.markdown("### Search Preferences")
    max_cand = st.slider("Default Max Candidates", min_value=1, max_value=20, value=st.session_state.max_candidates, key="settings_max_slider")
    
    st.info(f"📊 Currently set to return **{max_cand}** candidates per search")
    
    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        st.session_state.max_candidates = max_cand
        st.success(f"✓ Settings saved! Will now return {max_cand} candidates per search")
        time.sleep(1)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

from io import BytesIO

# Resolve API URL without relying on Streamlit secrets
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

# --------------- Page config & Styles ---------------
st.set_page_config(page_title="AI Recruiting Assistant", page_icon="🎯", layout="centered")

CUSTOM_CSS = """
<style>
/* Center title */
.center-title { text-align: center; margin-top: -1rem; }

/* Chat container */
.chat-container {
  max-width: 820px;
  margin: 0 auto;
  border: 1px solid #e6e6e6;
  border-radius: 12px;
  padding: 12px 12px 0 12px;
  height: 60vh;
  overflow-y: auto;
  background: #ffffff;
}
.msg { display: flex; margin: 8px 0; }
.msg.user { justify-content: flex-end; }
.msg.agent { justify-content: flex-start; }
.bubble {
  padding: 10px 14px;
  border-radius: 14px;
  max-width: 80%;
  line-height: 1.4;
  white-space: pre-wrap;
}
.bubble.user {
  background: #f0f0f0;
  color: #111;
}
.bubble.agent {
  background: #e8f0fe; /* light blue */
  color: #0b3d91;
}
.input-row {
  max-width: 820px;
  margin: 10px auto 0 auto;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------- Sidebar (optional settings) ---------------
with st.sidebar:
    st.markdown("### Settings")
    api_base = st.text_input("API URL", value=API_URL, help="FastAPI /chat endpoint URL")
    st.caption("Set API_URL env var to change default.")

# --------------- Session State ---------------
if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "agent", "content": "Hi, I’m your AI Recruiting Assistant. How can I help you today?"}
    ]

def add_message(role: str, content: str):
    st.session_state.chat.append({"role": role, "content": content})


# --------------- Title ---------------
st.markdown('<h1 class="center-title">🎯 AI Recruiting Assistant</h1>', unsafe_allow_html=True)

# --------------- Chat Container ---------------
chat_placeholder = st.container()
with chat_placeholder:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for m in st.session_state.chat:
        role = m.get("role")
        cls = "user" if role == "user" else "agent"
        content = m.get("content", "")
        st.markdown(f'<div class="msg {cls}"><div class="bubble {cls}">{content}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------- Input Area ---------------
st.markdown('<div class="input-row">', unsafe_allow_html=True)
col1, col2 = st.columns([4, 1])
with col1:
    user_text = st.text_input("Message", value="", placeholder="Type a message (e.g., Find top 3 data scientists with NLP) ...")
with col2:
    send_clicked = st.button("Send", type="primary", use_container_width=True)

uploaded_file = st.file_uploader("Upload job description PDF (optional)", type=["pdf"], accept_multiple_files=False)
st.markdown('</div>', unsafe_allow_html=True)


def call_chat_api(message: str, file_bytes: bytes | None, filename: str | None) -> tuple[bool, str]:
    """Return (ok, reply_text). Sends multipart if file present; else JSON."""
    headers = {}

    try:
        if file_bytes is not None and filename:
            files = {
                "file": (filename, BytesIO(file_bytes), "application/pdf"),
            }
            data = {"message": message}
            # Allow more time for PDF parsing + retrieval
            resp = requests.post(api_base, files=files, data=data, headers=headers, timeout=240)
        else:
            payload = {"message": message}
            # Allow more time for model/explanations on slower machines
            resp = requests.post(api_base, json=payload, headers=headers, timeout=180)

        if resp.status_code != 200:
            return False, f"API error {resp.status_code}: {resp.text}" 
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        reply = (data or {}).get("reply")
        if not reply:
            # Some endpoints might return plain text
            reply = resp.text or "(no reply)"
        return True, reply
    except Exception as e:
        return False, f"Request failed: {e}"


# --------------- Handle submission ---------------
if send_clicked:
    msg = (user_text or "").strip()
    if not msg and not uploaded_file:
        st.warning("Please type a message or upload a PDF.")
    else:
        if msg:
            add_message("user", msg)
        else:
            add_message("user", "(PDF uploaded)")

        file_bytes = uploaded_file.read() if uploaded_file else None
        filename = uploaded_file.name if uploaded_file else None

        with st.spinner("Thinking..."):
            ok, reply = call_chat_api(msg, file_bytes, filename)

        if ok:
            add_message("agent", reply)
        else:
            add_message("agent", f"⚠️ {reply}")

        # Rerun to refresh the chat container (compat across Streamlit versions)
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()  # older versions
            except Exception:
                pass

