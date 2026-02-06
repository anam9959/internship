from flask import Flask, request, render_template, jsonify
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import pdfplumber
from markupsafe import Markup
import json
import re

# load_dotenv()

app = Flask(__name__)

groq = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model="llama-3.1-8b-instant")

def response_cleaner(response):
    response = Markup(response.replace("\n", "<br>"))
    return response

def extract_scores_from_response(response):
    """Extract match percentage and skills from AI response"""
    match_score = 0
    missing_skills = []
    matching_skills = []
    
    # Try to extract percentage
    percentage_match = re.search(r'(\d+)%', response)
    if percentage_match:
        match_score = int(percentage_match.group(1))
    else:
        # Estimate based on keywords
        if "excellent match" in response.lower() or "highly qualified" in response.lower():
            match_score = 85
        elif "good match" in response.lower() or "qualified" in response.lower():
            match_score = 70
        elif "moderate match" in response.lower():
            match_score = 55
        else:
            match_score = 40
    
    return {
        'match_score': match_score,
        'missing_skills': missing_skills,
        'matching_skills': matching_skills
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("upload_resume.html")

@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    job_description = request.files.get('job_description')
    resumes = request.files.getlist("resume")  # Changed to getlist for multiple files
    
    if not job_description or not resumes:
        return "Please upload both job description and at least one resume", 400
    
    # Extract job description
    extracted_description = ""
    with pdfplumber.open(job_description) as pdf:
        extracted_description = "\n".join(page.extract_text() for page in pdf.pages)

    print("Job Description Extracted:", extracted_description[:200])
    print("----"*50)

    # Extract all resumes
    resume_analyses = []
    for idx, resume_file in enumerate(resumes):
        with pdfplumber.open(resume_file) as pdf:
            resume_text = "\n".join(page.extract_text() for page in pdf.pages)
        
        print(f"Resume {idx+1} Extracted:", resume_text[:200])
        
        # Analyze each resume
        prompt_template = """You are an expert HR professional and ATS (Applicant Tracking System) analyzer. 
        Analyze the resume against the job description and provide a detailed assessment.

        Job Description:
        {job_description}

        Resume:
        {resume_text}

        Provide your analysis in the following format:

        MATCH SCORE: [Provide a percentage from 0-100]%

        STRENGTHS:
        - List key strengths and relevant qualifications
        - Highlight experiences that align well with the job

        SKILL GAPS:
        - List missing skills or qualifications
        - Identify areas for improvement

        RECOMMENDATIONS:
        - Specific suggestions to improve the resume
        - Keywords to add for better ATS compatibility
        - Formatting or content improvements

        OVERALL ASSESSMENT:
        Provide a summary of whether this candidate should be considered and why.
        """

        prompt = PromptTemplate(
            input_variables=["job_description", "resume_text"],
            template=prompt_template
        )

        model = LLMChain(llm=groq, prompt=prompt)
        response = model.run(job_description=extracted_description, resume_text=resume_text)
        
        # Extract metrics from response
        metrics = extract_scores_from_response(response)
        
        resume_analyses.append({
            'filename': resume_file.filename,
            'analysis': response,
            'match_score': metrics['match_score'],
            'resume_number': idx + 1
        })
    
    # Sort by match score
    resume_analyses.sort(key=lambda x: x['match_score'], reverse=True)
    
    return render_template("result.html", 
                         analyses=resume_analyses,
                         job_description=extracted_description)

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chatbot queries about the resume analysis"""
    data = request.json
    user_message = data.get('message', '')
    context = data.get('context', '')
    
    chat_prompt = """You are a helpful HR assistant. Based on the resume analysis context provided, 
    answer the user's question in a clear and concise manner.

    Context:
    {context}

    User Question: {question}

    Provide a helpful, professional response:"""
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=chat_prompt
    )
    
    model = LLMChain(llm=groq, prompt=prompt)
    response = model.run(context=context, question=user_message)
    
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True, port=3000)