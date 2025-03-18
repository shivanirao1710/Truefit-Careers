from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
import psycopg2
from psycopg2 import sql
import PyPDF2
import docx
import re
from transformers import pipeline
from psycopg2.extras import DictCursor
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure key

# Load the LLM model for generating insights and ranking jobs
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# Initialize SentenceTransformer model for semantic understanding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize GPT-2 for text generation (for rephrasing or fixing grammar)
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Initialize Hugging Face LLM pipeline for simple conversation
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")
# Connect to PostgreSQL database
def get_db_connection():
    connection = psycopg2.connect(
        dbname="major",  # Your PostgreSQL database name
        user="postgres",         # Your PostgreSQL username
        password="shivanirao1710",     # Your PostgreSQL password
        host="localhost",        # Your PostgreSQL host
        port="5432"              # Default PostgreSQL port
    )
    return connection

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Extract text from Word document
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Generate insights using LLM
def generate_insights(text):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# Extract specific sections
def extract_section(text, keywords):
    for keyword in keywords:
        if keyword.lower() in text.lower():
            start_index = text.lower().find(keyword.lower())
            end_index = find_next_section(text, start_index)
            return text[start_index:end_index].strip()
    return ""

# Find the next section header
def find_next_section(text, start_index):
    section_headers = ["experience", "education", "projects", "skills", "certifications", "achievements"]
    next_index = len(text)
    for header in section_headers:
        header_index = text.lower().find(header, start_index + 1)
        if header_index != -1 and header_index < next_index:
            next_index = header_index
    return next_index

# Dictionary to map skill variations to their canonical form
skill_variations = {
    "nodejs": "node.js",
    "node.js": "node.js",
    "react.js": "reactjs",
    "react.js": "react",
    "next.js": "nextjs",
    "javascript": "js",
    "typescript": "ts",
    "c++": "cpp",
    "c#": "csharp",
    "html5": "html",
    "css3": "css",
    "postgresql": "postgres",
    "mongodb": "mongo",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "azure": "microsoft azure",
    "kubernetes": "k8s",
    "python3": "python",
    "python2": "python",
    # Add more variations as needed
}

def normalize_skill(skill):
    """Normalize a skill to its canonical form."""
    return skill_variations.get(skill.lower(), skill.lower())

def normalize_text(text):
    """Normalize skill variations in the resume text."""
    for variation, canonical in skill_variations.items():
        text = re.sub(rf"\b{re.escape(variation)}\b", canonical, text, flags=re.IGNORECASE)
    return text

# Parse resume text
def parse_resume(text):
    data = {
        "name": "",
        "email": "",
        "phone": "",
        "skills": "",
        "experience": "",
        "education": "",
        "projects": "",
        "insights": ""
    }

    # Normalize the resume text
    text = normalize_text(text)

    # Extract email
    emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text, re.IGNORECASE)
    if emails:
        data["email"] = emails[0]

    # Extract phone number
    phones = re.findall(r"\+?\d[\d -]{8,12}\d", text)
    if phones:
        data["phone"] = phones[0]

    # Extract name (first line as a placeholder)
    data["name"] = text.split("\n")[0].strip()

    # Extract skills
    skills_list = [
        # Technical skills (Programming, Web frameworks, Databases, Cloud, etc.)
        "python", "java", "c++", "c#", "ruby", "sql", "html", "css", "javascript", "java swings", "typescript", "php",
        "flask", "django", "nodejs", "expressjs", "react", "reactjs", "nextjs", "angular", "vue.js", "jquery", "bootstrap", "spring", "asp.net",
        "swift", "kotlin", "objective-c", "go", "scala", "perl", "shell scripting", "bash", "powershell", "rust",
        "haskell", "sql server", "postgresql", "mongodb", "mysql", "oracle", "redis", "firebase", "sqlite",
        "hadoop", "spark", "kafka", "elasticsearch", "cassandra", "bigquery", "aws", "azure", "google cloud", "gcp",
        "terraform", "docker", "kubernetes", "ansible", "puppet", "chef", "jenkins", "git", "gitlab", "github", "bitbucket",
        "vagrant", "virtualbox", "jenkins", "ci/cd", "maven", "gradle", "npm", "yarn", "bower", "nginx", "apache",
        "webpack", "graphql", "rest api", "soap", "json", "xml", "protobuf", "swagger", "microservices", "devops",
        "cloudformation", "azure devops", "cloud storage", "cloud architecture", "containerization", "serverless",
        "elastic beanstalk", "lambda", "cloudwatch", "docker swarm", "nginx", "apache kafka", "fluentd", "prometheus",
        "grafana", "openstack", "vagrant", "selenium", "pytest", "junit", "mocha", "chai", "karma", "jasmine",
        "testng", "jupyter", "pandas", "matplotlib", "seaborn", "numpy", "scikit-learn", "tensorflow", "pytorch",
        "keras", "nltk", "spaCy", "openCV", "d3.js", "tableau", "power bi", "matlab", "sas", "r", "spss", "stata",
        "excel", "rds", "spark sql", "sas", "apache flink", "databricks", "etl", "business intelligence", "data mining",
        "data engineering", "data scientist", "etl pipelines", "data lakes", "deep learning", "machine learning",
        "computer vision", "natural language processing", "predictive analytics", "data visualization", "statistics",
        "blockchain", "cryptocurrency", "bitcoin", "ethereum", "iot", "iot protocols", "home automation", "arduino",
        "raspberry pi", "mqtt", "zigbee", "smart contracts", "solidity", "ethereum", "docker", "pytorch", "keras",
        "tensorflow", "scipy", "data wrangling", "jupyter notebooks", "tableau", "google analytics", "splunk",
        "elasticsearch", "salesforce", "service now", "aws lambda", "apache spark", "cloud computing", "cloud migration",
        "blockchain", "nfc", "qr codes", "tcp/ip", "vpn", "pentesting", "ethical hacking", "penetration testing",
        "security", "open security", "ssl", "tls", "http", "oauth", "network security", "firewall", "siem", "firewall",
        "authentication", "authorization", "ssh", "sftp", "ssl", "keycloak", "data encryption", "cybersecurity", "risk management",
        "communication", "teamwork", "leadership", "problem-solving", "creativity", "critical thinking", "time management",
        "adaptability", "collaboration", "conflict resolution", "empathy", "active listening", "negotiation", "presentation",
        "public speaking", "decision making", "attention to detail", "interpersonal skills", "self-motivation", "work ethic",
        "confidentiality", "organizational skills", "stress management", "self-learning", "positive attitude", "customer service",
        "accountability", "delegation", "mentorship", "project management", "resource management", "goal setting",
        "strategic thinking", "analytical thinking", "emotional intelligence", "networking", "team building", "influencing",
        "persuasion", "flexibility", "confidentiality", "coaching", "facilitation", "mindfulness", "decision-making",
        "adaptability", "learning agility", "self-awareness", "conflict management", "collaboration skills", "relationship-building"
    ]

    # Normalize skills in the skills list
    normalized_skills_list = [normalize_skill(skill) for skill in skills_list]

    # Find skills in the resume text
    found_skills = [skill for skill in normalized_skills_list if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)]
    data["skills"] = ", ".join(found_skills)

    # Extract experience
    experience_keywords = ["experience", "work history", "employment", "professional experience"]
    data["experience"] = extract_section(text, experience_keywords)

    # Extract education
    education_keywords = ["education", "academic background", "degrees", "qualifications"]
    data["education"] = extract_section(text, education_keywords)

    # Extract projects
    projects_keywords = ["projects", "personal projects", "project experience"]
    data["projects"] = extract_section(text, projects_keywords)

    # Generate insights using LLM
    data["insights"] = generate_insights(text)

    return data

def get_job_data_from_postgresql():
    """Fetch job data from PostgreSQL database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        cursor.execute("SELECT job_role, company_name, company_type, knowledge_cleaned, skills_cleaned, combined_features, embedding FROM job_data_cleaned")
        job_data = cursor.fetchall()
        cursor.close()
        conn.close()
        return job_data
    except psycopg2.Error as err:
        print(f"Error: {err}")
        return []


def prepare_faiss_index(job_data):
    """Prepare FAISS index for semantic search"""
    job_embeddings = []
    job_titles = []
    for job in job_data:
        embedding = job.get('embedding', '')
        if embedding:
            try:
                embedding = json.loads(embedding)
                job_embeddings.append(embedding)
                job_titles.append(job['job_role'])
            except json.JSONDecodeError:
                continue
    job_embeddings = np.array(job_embeddings).astype('float32')
    faiss_index = faiss.IndexFlatL2(job_embeddings.shape[1])  # L2 distance for similarity search
    faiss_index.add(job_embeddings)
    return faiss_index, job_titles


def find_job_roles_by_skills(skills, top_n=5):
    """Find job roles based on skills"""
    skills_query = skills.lower().split(",")  # Split skills by comma and strip
    skills_query = [skill.strip() for skill in skills_query]
    query = " ".join(skills_query)
    
    query_embedding = model.encode([query])[0]
    job_data = get_job_data_from_postgresql()

    faiss_index, job_titles = prepare_faiss_index(job_data)
    query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_n)

    recommended_jobs = []
    for index in indices[0]:
        job = next((job for i, job in enumerate(job_data) if job_titles[i] == job_titles[index]), None)
        if job:
            recommended_jobs.append(job)
    return recommended_jobs


def find_job_roles_by_job_role(job_role, top_n=5):
    """Find job roles by job role name"""
    job_role = job_role.lower().strip()
    query_embedding = model.encode([job_role])[0]
    job_data = get_job_data_from_postgresql()

    faiss_index, job_titles = prepare_faiss_index(job_data)
    query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_n)

    recommended_jobs = []
    for index in indices[0]:
        job = next((job for i, job in enumerate(job_data) if job_titles[i] == job_titles[index]), None)
        if job:
            recommended_jobs.append(job)
    return recommended_jobs


def find_job_roles_by_company(company_name, top_n=5):
    """Find job roles based on company name"""
    company_name = company_name.lower().strip()
    job_data = get_job_data_from_postgresql()
    
    # Filter job data based on the company name
    filtered_jobs = [job for job in job_data if job['company_name'].lower() == company_name]
    
    return filtered_jobs


def correct_grammar_and_generate_response(text):
    """Generate a well-formed response using GPT-2 for grammar correction"""
    inputs = gpt_tokenizer.encode(text, return_tensors='pt')
    
    # Use max_new_tokens to handle long input
    outputs = gpt_model.generate(
        inputs,
        max_new_tokens=150,  # Limit the generation to 150 new tokens
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.92,
        temperature=0.7
    )
    
    response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def extract_job_role_from_input(user_input):
    """Extract job role from the user's input."""
    job_role_keywords = ["tell me about", "what is", "role of", "job of", "describe", "give me information about"]
    for keyword in job_role_keywords:
        if user_input.lower().startswith(keyword):
            return user_input[len(keyword):].strip()
    return user_input.strip()


@app.route("/chatbot", methods=["GET", "POST"])
def chatbot_route():
    """Handle chatbot interaction"""
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        
        if not user_input:
            return render_template("chatbot.html", user_input=user_input, response="Please ask me something!")

        # Check if the user is asking about job roles
        job_role_keywords = ['job', 'role', 'position', 'career']
        skills_keywords = ['skills', 'technology', 'tools']

        response = ""

        # If the user input contains keywords related to job roles
        if any(keyword in user_input.lower() for keyword in job_role_keywords):
            job_role = extract_job_role_from_input(user_input)
            job_info = get_job_data_from_postgresql()
            job_details = next((job for job in job_info if job['job_role'].lower() == job_role), None)
            
            if job_details:
                response = f"Job Role: {job_details['job_role']} at {job_details['company_name']}\n"
                response += f"Skills Needed: {job_details['skills_cleaned']}\n"
                response += f"Knowledge Required: {job_details['knowledge_cleaned']}\n"
                response += f"Company Type: {job_details['company_type']}\n"
                response += f"Description: {job_details.get('combined_features', 'No description available')}"
            else:
                response = f"Sorry, I couldn't find details for the job role '{job_role}'. Can you try another one?"
        
        # If the user input contains keywords related to skills
        elif any(keyword in user_input.lower() for keyword in skills_keywords):
            skills_query = user_input.lower().strip()
            recommended_jobs = find_job_roles_by_skills(skills_query)
            
            if recommended_jobs:
                response = "I found some jobs related to the skills you're looking for:\n"
                for job in recommended_jobs:
                    response += f"- {job['job_role']} at {job['company_name']} (Skills Needed: {job['skills_cleaned']})\n"
            else:
                response = f"I couldn't find any jobs with the skills '{skills_query}'. Can you try different skills?"

        # General conversation: Use the chatbot for free-form conversations
        else:
            response = chatbot(user_input, max_length=50, num_return_sequences=1)[0]['generated_text']
        
        # Use GPT-2 to correct grammar and rephrase the response
        response = correct_grammar_and_generate_response(response)
        
        return render_template("chatbot.html", user_input=user_input, response=response)
    
    return render_template("chatbot.html", user_input="", response="Ask me about job roles, skills, or any other questions!")


# Home Route for search functionality
@app.route("/search", methods=["GET", "POST"])
def search():
    recommendations = []
    search_type = None
    if request.method == "POST":
        search_type = request.form.get("search_type", "skills")
        query = request.form.get("skills", "").strip()
        
        if not query:
            return render_template("search.html", recommendations=[], error="Please enter skills, job role, or company name.")
        
        if search_type == "skills":
            recommendations = find_job_roles_by_skills(query)
        elif search_type == "job_role":
            recommendations = find_job_roles_by_job_role(query)
        elif search_type == "company_name":
            recommendations = find_job_roles_by_company(query)
        
        session['search_type'] = search_type
        session['recommendations'] = recommendations
        session['query'] = query
    
    if 'recommendations' in session:
        recommendations = session['recommendations']
        search_type = session['search_type']
    
    return render_template("search.html", recommendations=recommendations, search_type=search_type)


# Job Details Route (for viewing specific job details)
@app.route("/job/<job_role>/<company_name>")
def job_details(job_role, company_name):
    job_data = get_job_data_from_postgresql()
    
    job_details = next((job for job in job_data if job['job_role'] == job_role and job['company_name'] == company_name), None)
    
    if job_details:
        return render_template("job_details.html", job=job_details)
    else:
        return render_template("job_details.html", error="Job not found.")


# Reset Route to clear session data
@app.route("/reset")
def reset():
    session.pop('recommendations', None)
    session.pop('search_type', None)
    session.pop('query', None)
    return redirect(url_for('search'))


@app.route('/')
def index():
    return redirect(url_for('login'))

# Home route
@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user exists
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if user:
            flash("Username already exists!")
            conn.close()
            return redirect(url_for('register'))
        
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        flash("Registration successful! You can log in now.")
        conn.close()
        return redirect(url_for('home'))
    
    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if user and user[2] == password:  # user[2] is the password column
            session['user_id'] = user[0]  # user[0] is the user ID
            conn.close()
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials. Please try again.")
            conn.close()
    
    return render_template('login.html')

# Upload and parse resume
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if "resume" not in request.files:
            flash("No file uploaded!")
            return redirect("/upload")
        
        file = request.files["resume"]
        if file.filename == "":
            flash("No file selected!")
            return redirect("/upload")

        # Extract text based on file type
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.filename.endswith(".docx") or file.filename.endswith(".doc"):
            text = extract_text_from_docx(file)
        else:
            flash("Unsupported file format! Please upload a PDF or DOCX file.")
            return redirect("/upload")

        # Parse resume text
        parsed_data = parse_resume(text)

        # Save to database
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO resumes (user_id, name, email, phone, skills, experience, education, projects, file_name, insights) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (session['user_id'], parsed_data["name"], parsed_data["email"], parsed_data["phone"],
                 parsed_data["skills"], parsed_data["experience"], parsed_data["education"],
                 parsed_data["projects"], file.filename, parsed_data["insights"])
            )
            conn.commit()
            cur.close()
            conn.close()

            flash("Resume uploaded and parsed successfully!")
            return redirect("/display")

        except Exception as e:
            print("Error inserting into database:", e)
            flash("Error uploading resume. Please try again.")
            return redirect("/upload")
    
    return render_template("upload.html")

# Display resume data
@app.route("/display", methods=["GET"])
def display():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch the latest resume for the logged-in user
    cur.execute(
        "SELECT * FROM resumes WHERE user_id = %s ORDER BY resume_id DESC LIMIT 1",
        (session['user_id'],)
    )
    resume_data = cur.fetchone()
    cur.close()
    conn.close()

    if resume_data:
        return render_template("display.html", resume=resume_data)
    else:
        flash("No resume data found!")
        return redirect("/upload")

# Get job recommendations
@app.route("/get_jobs", methods=["GET"])
def get_jobs():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Fetch the latest resume data for the logged-in user
        cur.execute(
            "SELECT skills, experience, projects FROM resumes WHERE user_id = %s ORDER BY resume_id DESC LIMIT 1",
            (session['user_id'],)
        )
        resume_data = cur.fetchone()

        if not resume_data:
            print("No resume data found!")
            return jsonify({"jobs": []})

        # Extract skills from the resume
        combined_skills = set()
        for skills_field in resume_data:
            if skills_field:  # Check if the field is not None
                combined_skills.update([skill.strip().lower() for skill in skills_field.split(",")])

        print("Resume Skills:", combined_skills)

        # Fetch all jobs from the job_roles table
        cur.execute("SELECT job_role, company_name, company_type, skills FROM job_roles")
        jobs = cur.fetchall()

        if not jobs:
            print("No jobs found in the job_roles table!")
            return jsonify({"jobs": []})

        recommended_jobs = []
        for job in jobs:
            job_role, company_name, company_type, job_skills = job
            job_skills_set = set(job_skills.lower().split(", "))
            matched_skills = combined_skills.intersection(job_skills_set)
            missing_skills = job_skills_set.difference(combined_skills)

            if matched_skills:
                # Calculate relevance score (percentage of matched skills)
                relevance_score = (len(matched_skills) / len(job_skills_set)) * 100
                relevance_score = round(relevance_score, 2)  # Round to 2 decimal places

                recommended_jobs.append({
                    "job_role": job_role,
                    "company_name": company_name,
                    "company_type": company_type,
                    "skills": job_skills,
                    "matched_skills": ", ".join(matched_skills),
                    "missing_skills": ", ".join(missing_skills),
                    "relevance_score": relevance_score
                })

        # Rank jobs by relevance score (highest first)
        ranked_jobs = sorted(recommended_jobs, key=lambda x: x["relevance_score"], reverse=True)[:5]

        print("Recommended Jobs:", ranked_jobs)
        return jsonify({"jobs": ranked_jobs})

    except Exception as e:
        print("Error fetching job recommendations:", e)
        return jsonify({"jobs": []})

    
# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)