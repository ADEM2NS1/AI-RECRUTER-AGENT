"""
CV Processor - Handles CV uploads (PDF & images with OCR) and database storage
Integrates with existing BERT embeddings and Qdrant vector store
"""

import PyPDF2
import pytesseract
from PIL import Image
from io import BytesIO
import re
import uuid
import random
from datetime import datetime
from typing import Dict, List, Optional
import logging
import numpy as np

# Import your existing modules
from embeddings import get_bert_embeddings
from cv_storage import CVStorage
from preprocessing import preprocess_text
from qdrant_client.http.models import PointStruct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CVProcessor:
    """Process CV files (PDF or images) and add to Qdrant database."""
    
    # Comprehensive skill keywords
    SKILL_KEYWORDS = [
        # Programming Languages
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Ruby", "Go", 
        "Rust", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB",
        
        # ML/AI
        "Machine Learning", "ML", "Deep Learning", "AI", "Artificial Intelligence",
        "NLP", "Natural Language Processing", "Computer Vision", "Neural Networks",
        "CNN", "RNN", "Transformer", "BERT", "GPT",
        
        # Frameworks/Libraries
        "TensorFlow", "PyTorch", "Keras", "scikit-learn", "OpenCV", "NLTK", "spaCy",
        "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly",
        
        # Data & Databases
        "Data Science", "Data Analysis", "Data Engineering", "Statistics",
        "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "Redis", "Elasticsearch",
        
        # Cloud & DevOps
        "Docker", "Kubernetes", "AWS", "Azure", "GCP", "CI/CD", "Jenkins",
        "Git", "GitHub", "GitLab", "DevOps", "Terraform", "Ansible",
        
        # Web Development
        "React", "Angular", "Vue", "Node.js", "Express", "Django", "Flask",
        "FastAPI", "REST API", "GraphQL", "HTML", "CSS",
        
        # Other
        "Agile", "Scrum", "Testing", "Leadership", "Communication"
    ]
    
    def __init__(self):
        """Initialize with Qdrant storage."""
        self.storage = CVStorage()
        logger.info("CVProcessor initialized with Qdrant storage")
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF."""
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            pages = []
            for page in reader.pages:
                try:
                    text = page.extract_text() or ""
                    pages.append(text)
                except Exception as e:
                    logger.warning(f"Page extraction failed: {e}")
                    pages.append("")
            
            full_text = "\n".join(pages).strip()
            logger.info(f"✓ Extracted {len(full_text)} chars from PDF")
            return full_text
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise ValueError(f"PDF extraction error: {e}")
    
    def extract_text_from_image(self, image_bytes: bytes) -> str:
        """Extract text from image using Tesseract OCR."""
        try:
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Run OCR with Tesseract
            text = pytesseract.image_to_string(image, lang='eng')
            
            logger.info(f"✓ OCR extracted {len(text)} chars from image")
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            raise ValueError(f"OCR extraction error: {e}")
    
    def detect_file_type(self, filename: str, file_bytes: bytes) -> str:
        """Detect if file is PDF or image."""
        filename_lower = filename.lower()
        
        # Check extension
        if filename_lower.endswith('.pdf'):
            return 'pdf'
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            return 'image'
        
        # Check magic bytes
        if file_bytes.startswith(b'%PDF'):
            return 'pdf'
        elif file_bytes.startswith((b'\x89PNG', b'\xff\xd8\xff')):
            return 'image'
        
        raise ValueError("Unsupported file type. Upload PDF or image (PNG/JPG)")
    
    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        """Extract text from CV (auto-detect type)."""
        file_type = self.detect_file_type(filename, file_bytes)
        
        if file_type == 'pdf':
            return self.extract_text_from_pdf(file_bytes)
        elif file_type == 'image':
            return self.extract_text_from_image(file_bytes)
        else:
            raise ValueError("Unsupported file type")
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from CV text."""
        detected_skills = []
        text_lower = text.lower()
        
        for skill in self.SKILL_KEYWORDS:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower) and skill not in detected_skills:
                detected_skills.append(skill)
        
        return detected_skills
    
    def extract_experience(self, text: str) -> int:
        """Extract years of experience."""
        text_lower = text.lower()
        
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
            r'experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s+in',
        ]
        
        max_experience = 0
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                try:
                    years = int(match.group(1))
                    max_experience = max(max_experience, years)
                except (ValueError, IndexError):
                    continue
        
        return max_experience
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email address and clean common OCR errors."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        
        if matches:
            email = matches[0]
            
            # Clean common OCR errors
            # Remove common prefixes that are from icons/symbols
            email = re.sub(r'^[^\w]+', '', email)  # Remove leading non-word chars
            
            # Fix common OCR mistakes in email usernames
            # If email starts with 'pea' it might be 'a' (from icon)
            if email.startswith('pea') and '@' in email:
                email = email[2:]  # Remove 'pe' prefix
            
            return email
        
        return None

    
    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number."""
        phone_patterns = [
            r'\+?[\d\s\-\(\)]{10,}',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            if matches:
                phone = re.sub(r'[^\d+]', '', matches[0])
                if len(phone) >= 10:
                    return matches[0]
        
        return None
    
    def extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name (first few lines)."""
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and 2 <= len(line.split()) <= 4:
                if not re.search(r'\d', line) and line[0].isupper():
                    return line
        return None
    
    def process_and_store_cv(
        self, 
        file_bytes: bytes, 
        filename: str,
        provided_name: Optional[str] = None,
        provided_email: Optional[str] = None,
        provided_phone: Optional[str] = None
    ) -> Dict:
        """
        Process CV and store in Qdrant database.
        
        Returns:
            Dict with candidate information
        """
        try:
            # Step 1: Extract text
            logger.info(f"Processing CV: {filename}")
            raw_text = self.extract_text(file_bytes, filename)

            # Step 2: Extract email/name/phone from RAW text FIRST (before preprocessing removes them)
            extracted_name = self.extract_name(raw_text)
            extracted_email = self.extract_email(raw_text)
            extracted_phone = self.extract_phone(raw_text)

            # Step 3: NOW preprocess for embedding (this removes emails, so do it after extraction)
            cv_text = preprocess_text(raw_text)

            if not cv_text or len(cv_text) < 50:
                raise ValueError("CV is empty or too short")

            
            # Use provided or extracted info
            candidate_name = provided_name or extracted_name or "Unknown Candidate"
            candidate_email = provided_email or extracted_email or f"candidate_{uuid.uuid4().hex[:8]}@example.com"
            candidate_phone = provided_phone or extracted_phone or "N/A"
            
            # Step 4: Extract skills and experience
            skills = self.extract_skills(cv_text)
            experience = self.extract_experience(cv_text)
            
            # Step 5: Generate candidate IDs
            candidate_id = str(uuid.uuid4())[:8]  # Display ID
            numeric_id = random.randint(100000, 999999)  # Qdrant ID
            
            # Step 6: Generate BERT embedding
            logger.info("Generating BERT embedding...")
            embeddings = get_bert_embeddings([cv_text], batch_size=1)
            
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("Failed to generate embedding")
            
            embedding_vector = embeddings[0]
            
            # Step 7: Store in Qdrant directly
            logger.info(f"Storing CV in Qdrant: {candidate_name}")
            
            # Create payload
            payload = {
                "candidate_id": candidate_id,
                "name": candidate_name,
                "email": candidate_email,
                "phone": candidate_phone,
                "skills": skills,
                "experience": experience,
                "resume": cv_text,
                "raw_text": raw_text,
                "uploaded_at": datetime.now().isoformat(),
                "filename": filename
            }
            
            # Create point and store
            point = PointStruct(
                id=numeric_id,
                vector=embedding_vector.tolist() if hasattr(embedding_vector, 'tolist') else list(embedding_vector),
                payload=payload
            )
            
            self.storage.client.upsert(
                collection_name=self.storage.collection_name,
                points=[point]
            )
            
            logger.info(f"✓ CV stored successfully: {candidate_name} ({candidate_id})")
            
            return {
                "success": True,
                "candidate_id": candidate_id,
                "name": candidate_name,
                "email": candidate_email,
                "phone": candidate_phone,
                "skills": skills[:10],
                "experience": experience,
                "embedding_dimension": len(embedding_vector),
                "message": f"✓ CV successfully added to database for {candidate_name}"
            }
            
        except Exception as e:
            logger.error(f"CV processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to process CV: {e}"
            }


# Standalone functions for API use
def process_cv_file(file_bytes: bytes, filename: str, name: str = None, email: str = None, phone: str = None) -> Dict:
    """Process and store a CV file."""
    processor = CVProcessor()
    return processor.process_and_store_cv(file_bytes, filename, name, email, phone)


def score_cv_against_job(file_bytes: bytes, filename: str, job_description: str) -> Dict:
    """Score a CV against a job description using BERT embeddings."""
    try:
        processor = CVProcessor()
        
        # Extract and process CV
        raw_text = processor.extract_text(file_bytes, filename)
        cv_text = preprocess_text(raw_text)
        job_desc = preprocess_text(job_description)
        
        # Generate embeddings
        cv_embedding = get_bert_embeddings([cv_text], batch_size=1)[0]
        job_embedding = get_bert_embeddings([job_desc], batch_size=1)[0]
        
        # Calculate cosine similarity
        cv_norm = np.linalg.norm(cv_embedding)
        job_norm = np.linalg.norm(job_embedding)
        
        if cv_norm > 0 and job_norm > 0:
            similarity = np.dot(cv_embedding, job_embedding) / (cv_norm * job_norm)
            score = float(similarity * 100)
        else:
            score = 0.0
        
        # Extract skills
        cv_skills = processor.extract_skills(cv_text)
        job_skills = processor.extract_skills(job_desc)
        
        matched_skills = [s for s in job_skills if s in cv_skills]
        missing_skills = [s for s in job_skills if s not in cv_skills]
        
        # Experience
        experience = processor.extract_experience(cv_text)
        
        # Match level
        if score >= 80:
            match_level = "Excellent Match"
            recommendation = "Highly recommended for interview"
        elif score >= 60:
            match_level = "Good Match"
            recommendation = "Recommended for consideration"
        elif score >= 40:
            match_level = "Partial Match"
            recommendation = "May be suitable with training"
        else:
            match_level = "Low Match"
            recommendation = "Not recommended"
        
        logger.info(f"✓ CV scored {score:.1f}% against job description")
        
        return {
            "success": True,
            "score": round(score, 2),
            "match_level": match_level,
            "recommendation": recommendation,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "experience_years": experience,
            "details": {
                "total_required_skills": len(job_skills),
                "matched_skills_count": len(matched_skills),
                "skill_match_percentage": round((len(matched_skills) / len(job_skills) * 100) if job_skills else 0, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"CV scoring failed: {e}")
        return {"success": False, "error": str(e)}
