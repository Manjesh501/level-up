from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
from typing import List
import pdfplumber
import spacy
from transformers import pipeline
from collections import defaultdict
import re

# Initialize FastAPI app
app = FastAPI(title="NAAC SSR Summarizer API", version="1.0")

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize NLP and summarization models
nlp = spacy.load("en_core_web_lg")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


class SSRSummarizer:
    def __init__(self):
        self.key_metrics = {
            'institution': None,
            'establishment_year': None,
            'programs': [],
            'faculty_stats': defaultdict(int),
            'infrastructure': {},
            'research_stats': defaultdict(int),
            'naac_grade': None
        }

    def process_ssr(self, text: str) -> dict:
        """Main processing pipeline"""
        # Extract structured data
        self._extract_entities(text)
        self._extract_stats(text)
        
        # Generate summaries
        return {
            "executive_summary": self._generate_summary(text),
            "key_metrics": self.key_metrics,
            "critical_findings": self._identify_key_sections(text)
        }

    def _extract_entities(self, text: str):
        """Extract key entities using rule-based + ML approach"""
        doc = nlp(text)
        
        # Institution Name (First occurrence of ORG + College/University)
        for ent in doc.ents:
            if ent.label_ == "ORG" and any(x in ent.text.lower() for x in ["college", "university"]):
                self.key_metrics['institution'] = ent.text
                break
                
        # Establishment Year
        years = re.findall(r"\b(19|20)\d{2}\b", text)
        self.key_metrics['establishment_year'] = min(years) if years else None
        
        # Academic Programs
        programs = []
        for match in re.finditer(r"\b(B\.?[A-Z]+|M\.?[A-Z]+|PhD)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text):
            programs.append(f"{match.group(1)} in {match.group(2)}")
        self.key_metrics['programs'] = list(set(programs))[:10]  # Top 10 unique programs

    def _extract_stats(self, text: str):
        """Extract numerical statistics"""
        # Faculty Numbers
        faculty_matches = re.findall(
            r"(permanent faculty|teaching staff)\D*(\d+)", 
            text, 
            re.IGNORECASE
        )
        self.key_metrics['faculty_stats']['total'] = sum(int(m[1]) for m in faculty_matches)
        
        # Infrastructure
        infra_matches = re.findall(
            r"(classrooms|laboratories|computers)\D*(\d+)", 
            text, 
            re.IGNORECASE
        )
        for item, count in infra_matches:
            self.key_metrics['infrastructure'][item.lower()] = int(count)
            
        # Research Metrics
        research_matches = re.findall(
            r"(publications|patents|research projects)\D*(\d+)", 
            text, 
            re.IGNORECASE
        )
        for item, count in research_matches:
            self.key_metrics['research_stats'][item.lower()] = int(count)

    def _generate_summary(self, text: str) -> str:
        """Generate abstractive summary with key sections"""
        # Generate summary
        return summarizer(
            text[:1024],  # Truncate to avoid model input limits
            max_length=500,
            min_length=200,
            do_sample=False
        )[0]['summary_text']

    def _identify_key_sections(self, text: str) -> List[dict]:
        """Identify critical NAAC sections"""
        naac_keywords = {
            'strengths': ['strength', 'advantage', 'achievement'],
            'weaknesses': ['weakness', 'challenge', 'limitation'],
            'innovations': ['innovation', 'unique', 'patent'],
            'quality_indicators': ['naac', 'grade', 'accreditation']
        }
        
        findings = []
        for section, keywords in naac_keywords.items():
            matches = re.finditer("|".join(keywords), text, re.IGNORECASE)
            for match in matches:
                context = text[match.start()-200:match.end()+200].replace("\n", " ")
                findings.append({
                    'type': section,
                    'excerpt': f"...{context}..."
                })
                
        return findings[:5]  # Return top 5 critical findings


class TextInput(BaseModel):
    input_text: str


@app.get("/")
async def root():
    return {"message": "Welcome to the NAAC SSR Summarizer API. Use /docs for API documentation."}


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


@app.post("/process-text/")
async def process_text(text_input: TextInput):
    """Process plain text input"""
    try:
        summarizer = SSRSummarizer()
        result = summarizer.process_ssr(text_input.input_text)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-pdf/")
async def process_pdf(file: UploadFile):
    """Process PDF file input"""
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
        # Extract text from PDF
        with pdfplumber.open(file.file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        
        # Process the extracted text
        summarizer = SSRSummarizer()
        result = summarizer.process_ssr(text)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))