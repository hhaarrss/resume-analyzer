from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

# PDF and DOCX processing
try:
    import fitz  # PyMuPDF for PDFs
except ImportError:
    fitz = None

try:
    import docx2txt  # for DOCX
except ImportError:
    docx2txt = None

# NLP and ML
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    nlp = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Analyzer API", version="1.0.0")

# Configuration
UPLOAD_DIR = "uploads/resumes"
JD_DIR = "uploads/jd"
JD_PATH = os.path.join(JD_DIR, "job_description.txt")

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(JD_DIR, exist_ok=True)

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "API is running", "message": "Resume Analyzer API"}

@app.get("/health")
def detailed_health():
    """Check if all dependencies are available"""
    status = {
        "api": "running",
        "dependencies": {
            "fitz": fitz is not None,
            "docx2txt": docx2txt is not None,
            "spacy": nlp is not None
        }
    }
    return status

def extract_text(file_path: str) -> str:
    """Extract text from various file formats"""
    try:
        file_path = str(file_path)
        
        if file_path.lower().endswith(".pdf"):
            if fitz is None:
                raise HTTPException(status_code=500, detail="PDF processing not available. Install PyMuPDF: pip install PyMuPDF")
            
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
            
        elif file_path.lower().endswith(".docx"):
            if docx2txt is None:
                raise HTTPException(status_code=500, detail="DOCX processing not available. Install docx2txt: pip install docx2txt")
            return docx2txt.process(file_path)
            
        elif file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                return f.read()
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {Path(file_path).suffix}")
            
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def extract_info(text: str) -> Dict[str, str]:
    """Extract personal information from text"""
    try:
        info = {
            "name": "N/A",
            "email": "N/A", 
            "phone": "N/A"
        }
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            info["email"] = emails[0]
        
        # Extract phone
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            # Join tuple elements if pattern returns tuples
            phone = ''.join(phones[0]) if isinstance(phones[0], tuple) else phones[0]
            info["phone"] = phone.strip()
        
        # Extract name using spaCy if available
        if nlp is not None:
            doc = nlp(text[:1000])  # Process first 1000 chars for efficiency
            persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
            if persons:
                # Get the first person name that looks like a full name
                for person in persons:
                    if len(person.split()) >= 2:  # At least first and last name
                        info["name"] = person
                        break
                else:
                    info["name"] = persons[0]  # Fallback to first person found
        else:
            # Fallback: try to extract name from first few lines
            lines = text.split('\n')[:10]
            for line in lines:
                line = line.strip()
                # Look for lines that might be names (2-4 words, mostly alphabetic)
                words = line.split()
                if (len(words) >= 2 and len(words) <= 4 and 
                    all(word.replace('.', '').isalpha() for word in words)):
                    info["name"] = line
                    break
        
        return info
        
    except Exception as e:
        logger.error(f"Error extracting info: {str(e)}")
        return {"name": "N/A", "email": "N/A", "phone": "N/A"}

def get_similarity_score(resume_text: str, jd_text: str) -> float:
    """Calculate similarity score between resume and job description"""
    try:
        if not resume_text.strip() or not jd_text.strip():
            return 0.0
            
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=True
        )
        
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return round(similarity_score * 100, 2)
        
    except Exception as e:
        logger.error(f"Error calculating similarity score: {str(e)}")
        return 0.0

def generate_feedback(info: Dict[str, str], score: float) -> List[str]:
    """Generate feedback based on extracted info and score"""
    feedback = []
    
    # Contact information feedback
    missing_contact = []
    if info["email"] == "N/A":
        missing_contact.append("email")
    if info["phone"] == "N/A":
        missing_contact.append("phone number")
    if info["name"] == "N/A":
        missing_contact.append("name")
    
    if missing_contact:
        feedback.append(f"Missing contact information: {', '.join(missing_contact)}")
    
    # Score-based feedback
    if score < 30:
        feedback.append("Very low match with job requirements. Consider adding more relevant skills and keywords.")
    elif score < 50:
        feedback.append("Skills do not adequately match the job description. Add more relevant experience and keywords.")
    elif score < 75:
        feedback.append("Partial match with job requirements. Consider adding more relevant keywords and experience.")
    else:
        feedback.append("Good match with job requirements. Resume aligns well with the job description.")
    
    return feedback

@app.post("/upload-jd/")
async def upload_job_description(jd: UploadFile = File(...)):
    """Upload job description file"""
    try:
        # Validate file type
        if not jd.filename.lower().endswith(('.txt', '.pdf', '.docx')):
            raise HTTPException(status_code=400, detail="Only .txt, .pdf, and .docx files are supported")
        
        # Save the uploaded file
        temp_path = os.path.join(JD_DIR, f"temp_{jd.filename}")
        with open(temp_path, "wb") as f:
            content = await jd.read()
            f.write(content)
        
        # Extract text and save as .txt
        jd_text = extract_text(temp_path)
        
        if not jd_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        
        with open(JD_PATH, "w", encoding="utf-8") as f:
            f.write(jd_text)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {"message": "Job description uploaded successfully", "filename": jd.filename}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading job description: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading job description: {str(e)}")

@app.post("/analyze-resumes/")
async def analyze_resumes(files: List[UploadFile] = File(...)):
    """Analyze multiple resume files against the job description"""
    try:
        # Check if job description exists
        if not os.path.exists(JD_PATH):
            raise HTTPException(status_code=400, detail="No job description found. Please upload a job description first.")
        
        # Load job description
        with open(JD_PATH, "r", encoding="utf-8") as f:
            jd_text = f.read()
        
        if not jd_text.strip():
            raise HTTPException(status_code=400, detail="Job description is empty. Please upload a valid job description.")
        
        results = []
        
        for file in files:
            try:
                # Validate file type
                if not file.filename.lower().endswith(('.txt', '.pdf', '.docx')):
                    results.append({
                        "filename": file.filename,
                        "error": "Unsupported file format. Only .txt, .pdf, and .docx files are supported",
                        "name": "N/A",
                        "email": "N/A",
                        "score": 0,
                        "status": "Error",
                        "feedback": ["File format not supported"]
                    })
                    continue
                
                # Save uploaded file
                file_path = os.path.join(UPLOAD_DIR, file.filename)
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                # Process resume
                resume_text = extract_text(file_path)
                
                if not resume_text.strip():
                    results.append({
                        "filename": file.filename,
                        "error": "No text could be extracted from file",
                        "name": "N/A",
                        "email": "N/A", 
                        "score": 0,
                        "status": "Error",
                        "feedback": ["Could not extract text from file"]
                    })
                    continue
                
                # Extract information and calculate score
                info = extract_info(resume_text)
                score = get_similarity_score(resume_text, jd_text)
                feedback = generate_feedback(info, score)
                
                # Determine status
                if score >= 75:
                    status = "Excellent"
                elif score >= 50:
                    status = "Good"
                elif score >= 30:
                    status = "Fair"
                else:
                    status = "Poor"
                
                results.append({
                    "filename": file.filename,
                    "name": info["name"],
                    "email": info["email"],
                    "phone": info["phone"],
                    "score": score,
                    "status": status,
                    "feedback": feedback
                })
                
                # Clean up uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "name": "N/A",
                    "email": "N/A",
                    "score": 0,
                    "status": "Error",
                    "feedback": [f"Processing error: {str(e)}"]
                })
        
        return {"results": results, "total_processed": len(results)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_resumes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing resumes: {str(e)}")

# Additional utility endpoints
@app.get("/jd-status/")
def check_jd_status():
    """Check if job description is uploaded"""
    if os.path.exists(JD_PATH):
        with open(JD_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "uploaded": True,
            "length": len(content),
            "preview": content[:200] + "..." if len(content) > 200 else content
        }
    return {"uploaded": False}

@app.delete("/clear-jd/")
def clear_job_description():
    """Clear the current job description"""
    if os.path.exists(JD_PATH):
        os.remove(JD_PATH)
        return {"message": "Job description cleared"}
    return {"message": "No job description to clear"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)