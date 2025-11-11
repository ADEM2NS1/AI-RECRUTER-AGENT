# retriever.py

import os
import numpy as np
from typing import List, Dict
from together import Together
from embeddings import get_bert_embeddings
from storage import CVStorage  # your existing storage class

# -----------------------------
# Initialize Together AI client
# -----------------------------
together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
TOGETHER_MODEL = os.environ.get("TOGETHER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

def explain_rankingLLM(cv_info: dict, job_desc_clean: str) -> str:
    """
    Generate explanation why a candidate fits a job using Together AI.
    """
    prompt = f"""You are an HR assistant. Explain why this candidate is suitable for the job.

Candidate Resume Text:
{cv_info['full_text']}

Job Description:
{job_desc_clean}

Highlight skills, experience, and relevant points. Be concise."""
    
    try:
        response = together_client.chat.completions.create(
            model=TOGETHER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating explanation: {e}"


# -----------------------------
# Retriever Class
# -----------------------------
class CVRetriever:
    def __init__(self, storage: CVStorage, top_k: int = 5):
        """
        RAG Retriever using Qdrant storage.

        Args:
            storage: CVStorage instance
            top_k: number of top candidates to return
        """
        self.storage = storage
        self.top_k = top_k

    def retrieve_top_candidates(self, job_description: str) -> List[Dict]:
        """
        Given a job description, retrieve top CVs and generate explanations.
        """
        # Generate embedding for job description
        job_embedding = get_bert_embeddings([job_description])[0]  # shape (768,)

        # Retrieve top candidates from Qdrant
        results = self.storage.client.search(
            collection_name=self.storage.collection_name,
            query_vector=job_embedding.tolist(),
            limit=self.top_k
        )

        # Prepare output with explanations
        candidates = []
        for r in results:
            payload = r.payload or {}
            cv_info = {
                "full_text": payload.get("resume_text"),
                "candidate_id": payload.get("candidate_id"),
            }
            explanation = explain_rankingLLM(cv_info, job_description)
            candidates.append(
                {
                    "id": r.id,
                    "score": r.score,
                    "candidate_id": cv_info["candidate_id"],
                    "cv_text": cv_info["full_text"],
                    "explanation": explanation
                }
            )

        return candidates