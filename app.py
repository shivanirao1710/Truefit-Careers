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
import pickle

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

# Extract text from PDF #RESUME
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Extract text from Word document #RESUME
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Generate insights using LLM
def generate_insights(text): #RESUME
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# Extract specific sections  #RESUME
def extract_section(text, keywords):
    for keyword in keywords:
        if keyword.lower() in text.lower():
            start_index = text.lower().find(keyword.lower())
            end_index = find_next_section(text, start_index)
            return text[start_index:end_index].strip()
    return ""

# Find the next section header #RESUME
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
#RESUME
def normalize_skill(skill): 
    """Normalize a skill to its canonical form."""
    return skill_variations.get(skill.lower(), skill.lower())
#RESUME
def normalize_text(text):
    """Normalize skill variations in the resume text."""
    for variation, canonical in skill_variations.items():
        text = re.sub(rf"\b{re.escape(variation)}\b", canonical, text, flags=re.IGNORECASE)
    return text

# Parse resume text #RESUME
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
#JOB RECOMMENDATION #CHATBOT
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

#JOB RECOMMENDATION
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

#JOB RECOMMENDATION
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

#JOB RECOMMENDATION
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

#JOB RECOMMENDATION
def find_job_roles_by_company(company_name, top_n=5):
    """Find job roles based on company name"""
    company_name = company_name.lower().strip()
    job_data = get_job_data_from_postgresql()
    
    # Filter job data based on the company name
    filtered_jobs = [job for job in job_data if job['company_name'].lower() == company_name]
    
    return filtered_jobs

#CHATBOT
def get_user_name():
    user_id = session.get("user_id")  # Get logged-in user_id

    if not user_id:
        return "there"  # Default fallback if no user is logged in

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT name FROM resumes WHERE user_id = %s LIMIT 1;", (user_id,))
            result = cursor.fetchone()
            return result[0] if result else "there"  # Return name or fallback
    except Exception as e:
        print("Error fetching name:", e)
        return "there"
    finally:
        conn.close()

#CHATBOT
def correct_grammar_and_generate_response(text):
    """Fix grammar using GPT-2 and generate short, meaningful responses."""
    inputs = gpt_tokenizer.encode(text, return_tensors='pt')

    outputs = gpt_model.generate(
        inputs,
        max_new_tokens=30,  # Limit response length
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.85,
        temperature=0.6
    )

    generated_response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ✅ Extract the first proper sentence to avoid unnecessary text
    cleaned_response = re.split(r'[.!?]', generated_response)[0].strip() + "."

    return cleaned_response
#CHATBOT
# ✅ Chatbot Route
@app.route("/chatbot", methods=["GET", "POST"])
def chatbot_route():
    user_name = get_user_name()  # Fetch user name at the beginning
    greeting_message = f"Hello {user_name}! Ask me about job roles, skills, or anything else."
    
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()

        if not user_input:
            return render_template("chatbot.html", response=f"Hello {user_name}, please ask me something!")

        job_info = get_job_data_from_postgresql()  # Fetch job data
        response = ""

        # ✅ Check if user is asking about "skills needed for X"
        match = re.search(r"(?:skills needed for|skills for|technology required for)\s+(.+)", user_input, re.IGNORECASE)
        if match:
            job_role = match.group(1).strip()
            matching_jobs = [job for job in job_info if job_role.lower() in job["job_role"].lower()]

            if matching_jobs:
                response = f"Here are the key skills needed for **{job_role}**, {user_name}:\n\n"
                response += "\n".join(
                    f"- **{job['job_role']}** at {job['company_name']} (Skills: {job['skills_cleaned']})"
                    for job in matching_jobs[:3]
                )
            else:
                response = f"Sorry, {user_name}, I couldn't find skills for **'{job_role}'**. Try a different job title."

        # ✅ Check if user is asking about "job roles for X"
        elif re.search(r"(?:job roles for|roles for|positions for|careers in)\s+(.+)", user_input, re.IGNORECASE):
            job_role = re.search(r"(?:job roles for|roles for|positions for|careers in)\s+(.+)", user_input, re.IGNORECASE).group(1).strip()
            matching_jobs = [job for job in job_info if job_role.lower() in job["job_role"].lower()]

            if matching_jobs:
                response = f"Here are some job roles related to **{job_role}**, {user_name}:\n\n"
                response += "\n".join(
                    f"- **{job['job_role']}** at {job['company_name']} (Skills: {job['skills_cleaned']})"
                    for job in matching_jobs[:3]
                )
            else:
                response = f"Sorry, {user_name}, I couldn't find roles for **'{job_role}'**. Try a different job title."

        # ✅ Check if user is asking about "jobs at X"
        elif re.search(r"jobs at\s+(.+)", user_input, re.IGNORECASE):
            company_name = re.search(r"jobs at\s+(.+)", user_input, re.IGNORECASE).group(1).strip()
            matching_jobs = [job for job in job_info if company_name.lower() in job["company_name"].lower()]

            if matching_jobs:
                response = f"Here are some job roles available at **{company_name}**, {user_name}:\n\n"
                response += "\n".join(
                    f"- **{job['job_role']}** (Skills: {job['skills_cleaned']})"
                    for job in matching_jobs[:3]
                )
            else:
                response = f"Sorry, {user_name}, I couldn't find any jobs at **{company_name}**. Try checking the company's career page."

        # ✅ Handle General Queries with GPT-2 and DialoGPT
        else:
            corrected_input = correct_grammar_and_generate_response(user_input)  # Fix grammar first
            
            # ✅ Generate a conversational response using DialoGPT
            bot_response = chatbot(corrected_input, max_new_tokens=50, pad_token_id=gpt_tokenizer.eos_token_id)[0]['generated_text']
            
            # ✅ Extract the relevant response (removing repetitions)
            response = re.split(r'[\n]', bot_response)[0].strip()

        return render_template("chatbot.html", user_input=user_input, response=response, user_name=user_name)

    return render_template("chatbot.html", response=greeting_message, user_name=user_name)

#RESUME #PROFILE
def get_resume_data(user_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Query to get the most recent resume details using the user_id
        query = """
            SELECT resume_id, name, email, skills, education, insights, behavioral_tag
            FROM resumes 
            WHERE user_id = %s 
            ORDER BY resume_id DESC 
            LIMIT 1
        """
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()

        # Close the connection
        cursor.close()
        connection.close()

        # If data is found, return it as a dictionary
        if result:
            return {
                'resume_id': result[0],
                'name': result[1],
                'email': result[2],
                'skills': result[3],
                'education': result[4],
                'insights': result[5],
                'behavioral_tag': result[6]
            }
        else:
            return None

    except Exception as e:
        print("Error fetching data: ", e)
        return None
    

#BEHAVIOUR
# Load your trained Random Forest model
model = pickle.load(open("rf_model.pkl", "rb"))

# Function to fetch the questions from the database
def fetch_questions():
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT id, question_text FROM questions ORDER BY id")
        questions = cur.fetchall()
    conn.close()
    return questions

# Function to save responses and results in the database
def save_responses(user_id, answers, predicted_scores, behavioral_tag):
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Save responses (assuming 50 questions: answers is a list of 50 ints)
        placeholders = ','.join([f"q{i+1}" for i in range(len(answers))])
        values_placeholders = ','.join(['%s']*len(answers))
        insert_resp = f"INSERT INTO responses (user_id, {placeholders}) VALUES (%s, {values_placeholders})"
        cur.execute(insert_resp, [user_id] + answers)

        # Save results
        cur.execute(""" 
            INSERT INTO results (user_id, openness, conscientiousness, extraversion, agreeableness, neuroticism, behavioral_tag)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (user_id,
             predicted_scores['openness'],
             predicted_scores['conscientiousness'],
             predicted_scores['extraversion'],
             predicted_scores['agreeableness'],
             predicted_scores['neuroticism'],
             behavioral_tag)
        )
        conn.commit()
    conn.close()

# Function to generate behavioral tag based on the highest Big Five score
def get_behavioral_tag(predicted_scores):
    # Find the trait with the highest score
    max_trait = max(predicted_scores, key=predicted_scores.get)
    
    # Assign a behavioral tag based on the trait with the highest score
    if max_trait == 'openness':
        behavioral_tag = 'Creative'
    elif max_trait == 'conscientiousness':
        behavioral_tag = 'Organized'
    elif max_trait == 'extraversion':
        behavioral_tag = 'Extroverted'
    elif max_trait == 'agreeableness':
        behavioral_tag = 'Compassionate'
    elif max_trait == 'neuroticism':
        behavioral_tag = 'Neurotic'
    
    return behavioral_tag
def store_behavioral_tag(user_id, behavioral_tag):
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Update the behavioral_tag in the resumes table
        cur.execute("""
            UPDATE resumes
            SET behavioral_tag = %s
            WHERE user_id = %s
        """, (behavioral_tag, user_id))
        conn.commit()
    conn.close()

@app.route('/behaviour', methods=['GET'])
def behaviour():
    # Check if the user is logged in (using session or token)
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login page if not logged in

    # Fetch questions from the database
    questions = fetch_questions()
    
    # Render the behavioural analyzer page with the questions
    return render_template('behaviour.html', questions=questions)

@app.route('/submit', methods=['POST'])
def submit():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    # Step 1: Delete previous responses and results for this user (optional: keep history if needed)
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM responses WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM results WHERE user_id = %s", (user_id,))
        conn.commit()
    conn.close()

    # Step 2: Collect answers
    answers = []
    for i in range(1, 51):
        ans = request.form.get(f'q{i}')
        if ans is None or ans == '':
            return "Please answer all questions!", 400
        answers.append(int(ans))

    # Step 3: Predict scores
    prediction = model.predict([answers])[0]
    predicted_scores = {
        'openness': float(prediction[0]),
        'conscientiousness': float(prediction[1]),
        'extraversion': float(prediction[2]),
        'agreeableness': float(prediction[3]),
        'neuroticism': float(prediction[4])
    }

    # Step 4: Tag + Save
    behavioral_tag = get_behavioral_tag(predicted_scores)
    save_responses(user_id, answers, predicted_scores, behavioral_tag)
    store_behavioral_tag(user_id, behavioral_tag)

    return render_template(
        'result.html',
        scores=predicted_scores,
        behavioral_tag=behavioral_tag
    )


@app.route('/result')
def result():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    # Inline fetch_latest_result logic
    conn = get_db_connection()  # use your existing get_db_connection() function
    cur = conn.cursor()
    cur.execute("""
        SELECT openness, conscientiousness, extraversion, agreeableness, neuroticism, behavioral_tag
        FROM results
        WHERE user_id = %s
        LIMIT 1
    """, (user_id,))
    
    row = cur.fetchone()
    conn.close()

    if row:
        scores = {
            'openness': row[0],
            'conscientiousness': row[1],
            'extraversion': row[2],
            'agreeableness': row[3],
            'neuroticism': row[4]
        }
        behavioral_tag = row[5]
    else:
        return "No results found. Please take the test first."

    return render_template(
        'result.html',
        scores=scores,
        behavioral_tag=behavioral_tag
    )



# Route to display the profile page
#PROFILE
@app.route('/profile')
def profile():
    # Check if user_id exists in session
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    user_id = session['user_id']  # Get user_id from session
    resume_data = get_resume_data(user_id)
    
    if resume_data:
        return render_template('profile.html', data=resume_data)
    else:
        return "To update your profile, please upload your resume ", 404
    
# Home Route for search functionality
#JOB RECOMMENDATION
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
#JOB RECOMMENDATION
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

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_name = get_user_name()  # Fetch user's name from DB

    return render_template('home.html', user_name=user_name)

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
        flash("Registration successful!Log in")
        conn.close()
        return redirect(url_for('login'))  
    
    return render_template('register.html')


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
            return redirect(url_for('home'))  # Redirect to the home page after successful login
        else:
            flash("Invalid credentials. Please try again.")
            conn.close()

    return render_template('login.html')

# Upload and parse resume
#RESUME
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if "resume" not in request.files:
            flash("No file uploaded!")
            return redirect(url_for('upload'))
        
        file = request.files["resume"]
        if file.filename == "":
            flash("No file selected!")
            return redirect(url_for('upload'))

        # Extract text based on file type
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.filename.endswith(".docx") or file.filename.endswith(".doc"):
            text = extract_text_from_docx(file)
        else:
            flash("Unsupported file format! Please upload a PDF or DOCX file.")
            return redirect(url_for('upload'))

        # Parse resume text
        parsed_data = parse_resume(text)

        # Save to database
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            # Delete existing resumes for the user
            cur.execute(
                "DELETE FROM resumes WHERE user_id = %s",
                (session['user_id'],)
            )

            # Insert the new resume
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
            return redirect(url_for('display'))

        except Exception as e:
            print("Error inserting into database:", e)
            flash("Error uploading resume. Please try again.")
            return redirect(url_for('upload'))
    
    return render_template("upload.html")

# Display resume data
#RESUME
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
        return redirect(url_for('upload'))
    
# Get job recommendations
#RESUME
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