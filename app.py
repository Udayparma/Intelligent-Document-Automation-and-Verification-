from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_from_directory
)

import sqlite3
import os
import random
import re
import html

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

import pytesseract
import fitz

from PIL import Image

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename



# ============================================================
# SKILL SEARCH / MATCHING
# ============================================================

# Supported skills and common aliases. Matching is word-aware so,
# for example, Java does not incorrectly match JavaScript and C
# does not incorrectly match C++ or CSS.
SKILL_PATTERNS = {
    "Java": [r"(?<![A-Za-z0-9+#])java(?![A-Za-z0-9+#])"],
    "Python": [r"(?<![A-Za-z0-9+#])python(?![A-Za-z0-9+#])"],
    "C": [r"(?<![A-Za-z0-9+#])c(?![A-Za-z0-9+#])", r"\bc\s+(?:language|programming)\b"],
    "C++": [r"(?<![A-Za-z0-9])c\s*\+\s*\+(?![A-Za-z0-9])"],
    "C#": [r"(?<![A-Za-z0-9])c\s*#(?![A-Za-z0-9])", r"\bc sharp\b"],
    "JavaScript": [r"(?<![A-Za-z0-9])javascript(?![A-Za-z0-9])", r"\bjs\b"],
    "HTML": [r"\bhtml5?\b"],
    "CSS": [r"\bcss3?\b"],
    "SQL": [r"\bsql\b"],
    "MySQL": [r"\bmysql\b"],
    "PostgreSQL": [r"\bpostgres(?:ql)?\b"],
    "MongoDB": [r"\bmongodb\b", r"\bmongo db\b"],
    "Flask": [r"\bflask\b"],
    "Django": [r"\bdjango\b"],
    "React": [r"\breact(?:\.js|js)?\b"],
    "Node.js": [r"\bnode(?:\.js|js)?\b"],
    "Bootstrap": [r"\bbootstrap\b"],
    "Git": [r"\bgit\b"],
    "GitHub": [r"\bgithub\b"],
    "AWS": [r"\baws\b", r"\bamazon web services\b"],
    "Machine Learning": [r"\bmachine learning\b", r"\bml\b"],
    "Artificial Intelligence": [r"\bartificial intelligence\b", r"\bai\b"],
    "Data Science": [r"\bdata science\b", r"\bdata scientist\b"],
    "Data Analysis": [r"\bdata analysis\b", r"\bdata analytics\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "PyTorch": [r"\bpytorch\b"],
    "OpenCV": [r"\bopencv\b"],
    "NLP": [r"\bnatural language processing\b", r"\bnlp\b"],
    "Power BI": [r"\bpower bi\b"],
    "Excel": [r"\bms excel\b", r"\bmicrosoft excel\b", r"\bexcel\b"],
    "PHP": [r"\bphp\b"],
    "Laravel": [r"\blaravel\b"],
    "Spring Boot": [r"\bspring boot\b"],
    "Android": [r"\bandroid\b"],
    "Kotlin": [r"\bkotlin\b"],
    "Swift": [r"\bswift\b"],
    "Docker": [r"\bdocker\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "Linux": [r"\blinux\b"],
    "REST API": [r"\brest api\b", r"\brestful api\b"],
}
def canonicalize_skill(skill):
    """Return the official skill name for a user-entered skill."""
    if not skill:
        return None

    clean = skill.strip().lower()

    # Common user-entered aliases
    aliases = {
        "js": "JavaScript",
        "javascript": "JavaScript",
        "py": "Python",
        "cpp": "C++",
        "c plus plus": "C++",
        "csharp": "C#",
        "c sharp": "C#",
        "node": "Node.js",
        "nodejs": "Node.js",
        "ml": "Machine Learning",
        "ai": "Artificial Intelligence",
        "postgres": "PostgreSQL",
        "postgresql": "PostgreSQL",
        "mongo": "MongoDB",
        "mongodb": "MongoDB",
        "powerbi": "Power BI",
        "rest": "REST API",
        "restful api": "REST API",
    }

    if clean in aliases:
        return aliases[clean]

    for canonical_name in SKILL_PATTERNS:
        if canonical_name.lower() == clean:
            return canonical_name

    return None


def skill_found(text, skill):
    """Return True when the selected skill is found in extracted OCR text."""
    if not text or not skill:
        return False

    selected = canonicalize_skill(skill)

    if selected is None:
        return False

    normalized = text.lower()

    return any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in SKILL_PATTERNS[selected]
    )


def detect_skills(text):
    """Return all supported skills detected in a document."""
    if not text:
        return []

    return [
        skill
        for skill in SKILL_PATTERNS
        if skill_found(text, skill)
    ]

def calculate_skill_match(extracted_text, required_skills):
    """Return matched/missing skills and percentage for supported skills."""
    required = []

    for skill in required_skills:
        canonical = canonicalize_skill(skill)
        if canonical and canonical not in required:
            required.append(canonical)

    if not required:
        return [], [], 0

    matched = [skill for skill in required if skill_found(extracted_text or "", skill)]
    missing = [skill for skill in required if skill not in matched]
    score = round((len(matched) / len(required)) * 100)

    return matched, missing, score

# ============================================================
# RESUME SCORING / JOB MATCHING
# ============================================================

def calculate_resume_score(text, confidence=0):
    """Calculate a simple explainable resume score from OCR text.
    This is a rule-based score, not a claim of AI hiring accuracy.
    """
    text = text or ""
    normalized = text.lower()
    score = 0
    reasons = []

    # Contact information
    if re.search(r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b", text):
        score += 10
        reasons.append("Email found")
    if re.search(r"(?:\+91[-\s]?)?[6-9]\d{9}\b", text):
        score += 10
        reasons.append("Phone number found")

    # Education
    if any(term in normalized for term in ["b.tech", "btech", "bca", "mca", "b.sc", "m.tech", "b.e.", "b.e "]):
        score += 15
        reasons.append("Education found")

    # Sections commonly expected in resumes
    section_points = {
        "summary": ("summary", "career objective", "objective", "profile"),
        "skills": ("skills", "technical skills", "technical skill"),
        "projects": ("projects", "project"),
        "experience": ("experience", "work experience", "internship", "employment"),
        "certifications": ("certification", "certifications", "certificate"),
    }
    section_weights = {
        "summary": 10,
        "skills": 20,
        "projects": 15,
        "experience": 10,
        "certifications": 5,
    }
    for section, terms in section_points.items():
        if any(term in normalized for term in terms):
            score += section_weights[section]
            reasons.append(f"{section.title()} section found")

    detected = detect_skills(text)
    skill_bonus = min(20, len(detected) * 2)
    score += skill_bonus
    if detected:
        reasons.append(f"{len(detected)} supported skill(s) detected")

    # Confidence contributes a small amount without dominating the score.
    try:
        score += round(min(max(float(confidence or 0), 0), 100) * 0.10)
    except (TypeError, ValueError):
        pass

    score = min(100, int(round(score)))
    if score >= 80:
        label = "Excellent"
    elif score >= 65:
        label = "Good"
    elif score >= 50:
        label = "Average"
    else:
        label = "Needs Improvement"

    return score, label, reasons


def extract_skills_from_job_description(job_description):
    """Detect only supported skills from a job description."""
    if not job_description:
        return []
    return [skill for skill in SKILL_PATTERNS if skill_found(job_description, skill)]


def build_skill_recommendation(missing):
    """Return a short recruitment-oriented recommendation."""
    if not missing:
        return "Candidate has all requested skills."
    if len(missing) == 1:
        return f"Candidate should improve: {missing[0]}."
    return "Candidate should improve: " + ", ".join(missing) + "."


# ============================================================
# FILE UPLOAD SETTINGS
# ============================================================

ALLOWED_EXTENSIONS = {
    "pdf",
    "jpg",
    "jpeg",
    "png"
}

MAX_FILE_SIZE = 5 * 1024 * 1024


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# FLASK CONFIGURATION
# ============================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

app = Flask(__name__)

app.secret_key = "document_secret_key"

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ============================================================
# DATABASE / FOLDERS
# ============================================================

DATABASE = "database.db"

UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "reports"
PROFILE_FOLDER = os.path.join(UPLOAD_FOLDER, "profiles")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROFILE_FOLDER"] = PROFILE_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(PROFILE_FOLDER, exist_ok=True)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# DATABASE CREATION
# ============================================================

def create_database():

    conn = get_db()

    cursor = conn.cursor()

    # ---------------- USERS TABLE ---------------- #

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL,

            mobile TEXT DEFAULT '',
            address TEXT DEFAULT '',
            profile_photo TEXT DEFAULT ''

        )
    """)

    # ---------------- DOCUMENTS TABLE ---------------- #

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER,

            filename TEXT,

            extracted_text TEXT,

            verification_status TEXT,

            document_type TEXT,

            confidence_score INTEGER,

            summary TEXT,

            download_count INTEGER DEFAULT 0,

            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    cursor.execute("PRAGMA table_info(users)")
    existing_columns = {row["name"] for row in cursor.fetchall()}

    if "mobile" not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN mobile TEXT DEFAULT ''")

    if "address" not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN address TEXT DEFAULT ''")

    if "profile_photo" not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN profile_photo TEXT DEFAULT ''")

    conn.commit()
    conn.close()


create_database()


# ============================================================
# OCR FUNCTION
# ============================================================

def extract_text(filepath):

    text = ""

    extension = os.path.splitext(filepath)[1].lower()

    # ---------------- IMAGE OCR ---------------- #

    if extension in [".jpg", ".jpeg", ".png"]:

        image = Image.open(filepath)

        text = pytesseract.image_to_string(image)

    # ---------------- PDF OCR ---------------- #

    elif extension == ".pdf":

        pdf = fitz.open(filepath)

        for page_number, page in enumerate(pdf):

            pix = page.get_pixmap()

            temp_image = os.path.join(
                UPLOAD_FOLDER,
                f"temp_page_{page_number}.png"
            )

            pix.save(temp_image)

            image = Image.open(temp_image)

            text += pytesseract.image_to_string(image)

            image.close()

            if os.path.exists(temp_image):

                os.remove(temp_image)

        pdf.close()

    return text


# ============================================================
# DOCUMENT TYPE DETECTION
# ============================================================

def detect_document_type(text):
    """Detect the uploaded document type using OCR text."""

    text = text or ""
    normalized = text.lower()

    # ---------------- AADHAAR CARD ---------------- #
    # Handles Aadhaar / Aadhar / UIDAI and common OCR variations.
    aadhaar_keywords = (
        "uidai" in normalized
        or "aadhaar" in normalized
        or "aadhar" in normalized
        or "unique identification authority" in normalized
        or "government of india" in normalized
    )

    # Aadhaar numbers are 12 digits, often written as 4-4-4 groups.
    aadhaar_number = re.search(
        r"(?<!\d)(?:\d{4}[ -]?\d{4}[ -]?\d{4})(?!\d)",
        text
    )

    if aadhaar_keywords and aadhaar_number:
        return "Aadhaar Card"

    # If OCR misses the keyword but clearly reads a 12-digit Aadhaar-style
    # number, classify it as Aadhaar only when the text is otherwise short
    # enough to avoid confusing ordinary documents.
    if aadhaar_number and len(re.sub(r"\s+", " ", text).strip()) < 2500:
        if (
            "dob" in normalized
            or "date of birth" in normalized
            or "male" in normalized
            or "female" in normalized
            or "address" in normalized
        ):
            return "Aadhaar Card"

    # ---------------- PAN CARD ---------------- #
    pan_keywords = (
        "permanent account number" in normalized
        or "income tax" in normalized
        or "pan card" in normalized
        or re.search(r"\bpan\s*(?:no|number|card)?\b", normalized)
    )

    pan_number = re.search(
        r"(?<![A-Za-z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Za-z0-9])",
        text.upper()
    )

    if pan_keywords and pan_number:
        return "PAN Card"

    # A valid PAN pattern is highly distinctive. If OCR reads the PAN number
    # but misses the heading, accept it as PAN Card.
    if pan_number:
        return "PAN Card"

    # ---------------- RESUME ---------------- #
    if (
        "resume" in normalized
        or "skills" in normalized
        or "education" in normalized
    ):
        return "Resume"

    # ---------------- INVOICE ---------------- #
    if (
        "invoice" in normalized
        or "gst" in normalized
    ):
        return "Invoice"

    # ---------------- MARKSHEET ---------------- #
    if (
        "marksheet" in normalized
        or "mark sheet" in normalized
        or "roll no" in normalized
    ):
        return "Marksheet"

    return "Unknown"


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def calculate_confidence(document_type):

    if document_type == "Aadhaar Card":

        return random.randint(95, 99)

    elif document_type == "PAN Card":

        return random.randint(94, 99)

    elif document_type == "Resume":

        return random.randint(90, 97)

    elif document_type == "Invoice":

        return random.randint(88, 96)

    elif document_type == "Marksheet":

        return random.randint(90, 98)

    else:

        return random.randint(30, 60)


# ============================================================
# SMART VERIFICATION
# ============================================================

def smart_verification(text, document_type):

    text = text.lower()

    # ---------------- RESUME ---------------- #

    if document_type == "Resume":

        if (
            "education" in text
            and "skills" in text
            and "project" in text
        ):

            return "Verified"

        return "Rejected"

    # ---------------- AADHAAR ---------------- #

    elif document_type == "Aadhaar Card":

        # Accept 12 digits with spaces, hyphens, or no separators.
        aadhaar = re.findall(
            r"(?<!\d)(?:\d{4}[ -]?\d{4}[ -]?\d{4})(?!\d)",
            text
        )

        has_aadhaar_keyword = (
            "uidai" in text
            or "aadhaar" in text
            or "aadhar" in text
            or "unique identification authority" in text
        )

        if len(aadhaar) > 0 and has_aadhaar_keyword:
            return "Verified"

        return "Rejected"

    # ---------------- PAN ---------------- #

    elif document_type == "PAN Card":

        pan = re.findall(
            r"(?<![A-Z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z0-9])",
            text.upper()
        )

        if len(pan) > 0:
            return "Verified"

        return "Rejected"

    # ---------------- INVOICE ---------------- #

    elif document_type == "Invoice":

        if (
            "invoice" in text
            and "gst" in text
        ):

            return "Verified"

        return "Rejected"

    return "Rejected"


# ============================================================
# AI SUMMARY
# ============================================================

def generate_summary(text):

    lines = text.split("\n")

    clean_lines = []

    for line in lines:

        line = line.strip()

        if line != "":

            clean_lines.append(line)

    # ---------------- CANDIDATE NAME ---------------- #

    if len(clean_lines) > 0:

        candidate_name = clean_lines[0]

    else:

        candidate_name = "Not Found"

    # ---------------- QUALIFICATION ---------------- #

    qualification = "Not Found"

    if "b.tech" in text.lower():

        qualification = "B.Tech"

    elif "bca" in text.lower():

        qualification = "BCA"

    elif "mca" in text.lower():

        qualification = "MCA"

    elif "b.sc" in text.lower():

        qualification = "B.Sc"

    # ---------------- SKILLS ---------------- #

    skill_list = []

    keywords = [

        "Python",
        "Java",
        "C++",
        "C",
        "Flask",
        "Machine Learning",
        "HTML",
        "CSS",
        "JavaScript",
        "SQL",
        "Bootstrap",
        "React",
        "Django"

    ]

    for skill in keywords:

        if skill.lower() in text.lower():

            skill_list.append(skill)

    if len(skill_list) == 0:

        skills = "Not Found"

    else:

        skills = ", ".join(skill_list)

    summary = f"""
Candidate Name : {candidate_name}

Qualification : {qualification}

Skills : {skills}
"""

    return summary



# ============================================================
# NAME-BASED DOCUMENT FILENAME
# ============================================================

def create_name_based_filename(extracted_text, document_type, original_filename):
    """Create a safe filename using the document owner's name from OCR text.
    If the name cannot be detected, keep the original filename.
    """
    text = extracted_text or ""
    name = ""

    # Resume: the existing summary already treats the first non-empty
    # OCR line as the candidate name, so use the same existing logic.
    if document_type == "Resume":
        for line in text.splitlines():
            line = line.strip()
            if line:
                name = line
                break

    # Other documents: prefer an explicit Name field when OCR contains one.
    if not name:
        match = re.search(
            r"(?:^|\n)\s*(?:name|full\s+name|holder\s+name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{2,60})",
            text,
            flags=re.IGNORECASE
        )
        if match:
            name = match.group(1).strip()

    # Clean OCR noise and characters that are not suitable for filenames.
    name = re.sub(r"[^A-Za-z0-9 ._-]", "", name)
    name = re.sub(r"\s+", " ", name).strip(" ._-")

    if len(name) < 2 or len(name) > 60:
        return original_filename

    # Avoid accidentally using generic OCR headings as a person's name.
    blocked = {
        "resume", "curriculum vitae", "curriculum vitae cv",
        "government of india", "income tax department",
        "permanent account number", "aadhaar", "aadhar"
    }
    if name.lower() in blocked:
        return original_filename

    extension = os.path.splitext(original_filename)[1].lower()
    base_name = secure_filename(name)

    if not base_name:
        return original_filename

    new_filename = f"{base_name}{extension}"
    counter = 2

    # Keep filenames unique so an existing document is never overwritten.
    while os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], new_filename)):
        new_filename = f"{base_name}_{counter}{extension}"
        counter += 1

    return new_filename


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]

        email = request.form["email"]

        password = request.form["password"]

        hash_password = generate_password_hash(password)

        conn = get_db()

        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        )

        user = cursor.fetchone()

        if user:

            conn.close()

            flash("Email Already Exists")

            return redirect("/register")

        cursor.execute(
            """
            INSERT INTO users(name, email, password)
            VALUES(?,?,?)
            """,
            (
                name,
                email,
                hash_password
            )
        )

        conn.commit()

        conn.close()

        flash("Registration Successful")

        return redirect("/login")

    return render_template("register.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]

        password = request.form["password"]

        conn = get_db()

        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        )

        user = cursor.fetchone()

        conn.close()

        if user is not None:

            if check_password_hash(
                user["password"],
                password
            ):

                session["user_id"] = user["id"]

                session["user_name"] = user["name"]

                return redirect("/dashboard")

        flash("Invalid Email or Password")

    return render_template("login.html")


# ============================================================
# DASHBOARD
# ============================================================

# ============================================================
# REPAIR EXISTING DOCUMENT NAMES
# ============================================================
def repair_existing_document_names(conn, user_id):
    """
    Extra feature: rename already-uploaded documents using the
    document owner's name found in the saved OCR text.

    Existing upload/verification logic is not changed.
    This only repairs old database records whose filename still
    contains the original uploaded filename.
    """
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, filename, extracted_text, document_type
        FROM documents
        WHERE user_id=?
        """,
        (user_id,)
    )

    rows = cursor.fetchall()
    changed = 0

    for doc in rows:
        current_filename = doc["filename"] or ""
        text = doc["extracted_text"] or ""
        document_type = doc["document_type"] or ""

        if not text.strip():
            continue

        # --------------------------------------------------------
        # FIND OWNER NAME FROM OCR
        # --------------------------------------------------------
        name = ""

        if document_type == "Resume":
            for raw_line in text.splitlines():
                candidate = re.sub(r"\s+", " ", raw_line).strip()
                if candidate:
                    name = candidate
                    break

        if not name:
            match = re.search(
                r"(?:^|\n)\s*(?:name|full\s+name|holder\s+name|"
                r"candidate\s+name|applicant\s+name)\s*[:\-]?\s*"
                r"([A-Za-z][A-Za-z .'-]{2,60})",
                text,
                flags=re.IGNORECASE
            )
            if match:
                name = match.group(1).strip()

        # --------------------------------------------------------
        # CLEAN NAME
        # --------------------------------------------------------
        name = re.sub(r"[^A-Za-z0-9 ._-]", "", name)
        name = re.sub(r"\s+", " ", name).strip(" ._-")

        blocked = {
            "resume",
            "cv",
            "curriculum vitae",
            "curriculum vitae cv",
            "profile",
            "personal details",
            "contact",
            "contact details",
            "skills",
            "technical skills",
            "education",
            "experience",
            "projects",
            "certifications",
            "objective",
            "career objective",
            "government of india",
            "income tax department",
            "permanent account number",
            "aadhaar",
            "aadhar",
            "uidai",
        }

        if (
            len(name) < 2
            or len(name) > 60
            or name.lower() in blocked
        ):
            continue

        safe_name = secure_filename(name)

        if not safe_name:
            continue

        extension = os.path.splitext(current_filename)[1].lower()

        if not extension:
            extension = ".pdf"

        target_filename = f"{safe_name}{extension}"

        current_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            current_filename
        )

        target_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            target_filename
        )

        # --------------------------------------------------------
        # IF ALREADY CORRECT, DO NOTHING
        # --------------------------------------------------------
        if current_filename.lower() == target_filename.lower():
            continue

        # --------------------------------------------------------
        # AVOID OVERWRITING ANOTHER DOCUMENT
        # --------------------------------------------------------
        if os.path.exists(target_path):
            counter = 2

            while True:
                candidate_filename = (
                    f"{safe_name}_{counter}{extension}"
                )

                candidate_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    candidate_filename
                )

                if (
                    candidate_filename.lower()
                    == current_filename.lower()
                ):
                    target_filename = candidate_filename
                    target_path = candidate_path
                    break

                if not os.path.exists(candidate_path):
                    target_filename = candidate_filename
                    target_path = candidate_path
                    break

                counter += 1

        # --------------------------------------------------------
        # RENAME PHYSICAL FILE + UPDATE DATABASE
        # --------------------------------------------------------
        try:
            if os.path.exists(current_path):
                os.replace(current_path, target_path)

            cursor.execute(
                """
                UPDATE documents
                SET filename=?
                WHERE id=?
                """,
                (
                    target_filename,
                    doc["id"]
                )
            )

            changed += 1

        except Exception as e:
            print(
                "DOCUMENT NAME REPAIR ERROR:",
                current_filename,
                e
            )

    conn.commit()

    return changed



@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect("/login")

    search = request.args.get("search", "").strip()
    skill = request.args.get("skill", "").strip()

    # Multiple required skills can be entered like:
    # Java, Python, SQL
    required_skills_input = request.args.get(
        "required_skills", ""
    ).strip()

    # Optional job description. The existing dashboard does not need to change:
    # if a job description is supplied through the URL, supported skills are
    # automatically extracted and used for candidate matching.
    job_description = request.args.get("job_description", "").strip()
    if job_description and not required_skills_input:
        required_skills_input = ", ".join(
            extract_skills_from_job_description(job_description)
        )

    conn = get_db()
    cursor = conn.cursor()

    # ============================================================
    # EXTRA: REPAIR OLD DOCUMENT FILENAMES FROM OCR NAMES
    # ============================================================
    # This runs automatically when the dashboard opens.
    # Existing upload/verification features remain unchanged.
    repair_existing_document_names(
        conn,
        session["user_id"]
    )

    # ---------------- LOGGED-IN USER PROFILE ---------------- #
    cursor.execute(
        """
        SELECT id, name, email, mobile, address, profile_photo
        FROM users
        WHERE id=?
        """,
        (session["user_id"],)
    )
    current_user = cursor.fetchone()

    # ---------------- DOCUMENT SEARCH ---------------- #

    if search:

        cursor.execute(
            """
            SELECT *
            FROM documents
            WHERE user_id=?
            AND (
                filename LIKE ?
                OR document_type LIKE ?
            )
            ORDER BY id DESC
            """,
            (
                session["user_id"],
                "%" + search + "%",
                "%" + search + "%"
            )
        )

    else:

        cursor.execute(
            """
            SELECT *
            FROM documents
            WHERE user_id=?
            ORDER BY id DESC
            """,
            (session["user_id"],)
        )

    documents = cursor.fetchall()

    # ---------------- ANALYTICS ---------------- #

    total_documents = len(documents)

    verified_documents = len(
        [
            doc
            for doc in documents
            if doc["verification_status"] == "Verified"
        ]
    )

    rejected_documents = len(
        [
            doc
            for doc in documents
            if doc["verification_status"] == "Rejected"
        ]
    )

    resume_count = len(
        [
            doc
            for doc in documents
            if doc["document_type"] == "Resume"
        ]
    )

    aadhaar_count = len(
        [
            doc
            for doc in documents
            if doc["document_type"] == "Aadhaar Card"
        ]
    )

    pan_count = len(
        [
            doc
            for doc in documents
            if doc["document_type"] == "PAN Card"
        ]
    )

    if total_documents > 0:

        average_confidence = sum(
            doc["confidence_score"] or 0
            for doc in documents
        ) / total_documents

    else:

        average_confidence = 0

    average_confidence = round(
        average_confidence,
        2
    )

    recent_documents = documents[:5]

    # --------------------------------------------------------
    # SKILL FINDER
    # --------------------------------------------------------
    # User can enter: C, Java, Python, Swastik
    # Every item is checked separately.
    # Unknown words are NEVER treated as skills; they become Not Found.

    skill_results = []
    skill_search_message = ""
    skill_name = ""

    if skill:
        raw_skills = [
            item.strip()
            for item in skill.split(",")
            if item.strip()
        ]

        # Remove duplicates but preserve user order.
        unique_raw_skills = []
        seen = set()
        for item in raw_skills:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                unique_raw_skills.append(item)

        # Convert only known skills to their canonical names.
        searched_skills = []
        for item in unique_raw_skills:
            canonical = canonicalize_skill(item)
            searched_skills.append({
                "input": item,
                "canonical": canonical or item,
                "supported": canonical is not None
            })

        skill_name = ", ".join(item["input"] for item in searched_skills)

        # Always search ALL resumes belonging to this user.
        # This means every resume gets a Found/Not Found status.
        cursor.execute(
            """
            SELECT *
            FROM documents
            WHERE user_id=?
            AND document_type='Resume'
            ORDER BY id DESC
            """,
            (session["user_id"],)
        )
        resumes = cursor.fetchall()

        for resume in resumes:
            resume_text = resume["extracted_text"] or ""
            skill_statuses = []
            found_skills = []
            not_found_skills = []

            for item in searched_skills:
                display_skill = item["canonical"]

                # Unsupported search text (e.g. Swastik) is always Not Found.
                if not item["supported"]:
                    found = False
                else:
                    found = skill_found(resume_text, display_skill)

                skill_statuses.append({
                    "skill": display_skill,
                    "found": found
                })

                if found:
                    found_skills.append(display_skill)
                else:
                    not_found_skills.append(display_skill)

            # Show ALL resumes, not just resumes where one skill was found.
            # The UI can therefore display Found and Not Found for every search.
            skill_results.append({
                "id": resume["id"],
                "filename": resume["filename"],
                "document_type": resume["document_type"],
                "verification_status": resume["verification_status"],
                "confidence_score": resume["confidence_score"],
                "upload_date": resume["upload_date"],
                "skill": ", ".join(found_skills),
                "found": bool(found_skills),
                "matched_skills": found_skills,
                "not_found_skills": not_found_skills,
                "skill_statuses": skill_statuses
            })

        if not resumes:
            skill_search_message = "No resume has been uploaded yet."
        elif not any(result["matched_skills"] for result in skill_results):
            skill_search_message = (
                "No searched skill was found in the uploaded resumes. "
                "Each requested skill is marked Not Found below."
            )

    # --------------------------------------------------------
    # MULTI-SKILL CANDIDATE MATCHING
    # --------------------------------------------------------
    candidate_results = []
    required_skills = [
        item.strip()
        for item in required_skills_input.split(",")
        if item.strip()
    ]

    canonical_required = []
    for item in required_skills:
        canonical = canonicalize_skill(item)
        if canonical and canonical not in canonical_required:
            canonical_required.append(canonical)

    if canonical_required:
        cursor.execute(
            """
            SELECT *
            FROM documents
            WHERE user_id=?
            AND document_type='Resume'
            ORDER BY id DESC
            """,
            (session["user_id"],)
        )
        resumes = cursor.fetchall()

        for resume in resumes:
            matched, missing, score = calculate_skill_match(
                resume["extracted_text"] or "",
                canonical_required
            )

            candidate_results.append({
                "id": resume["id"],
                "filename": resume["filename"],
                "matched": matched,
                "missing": missing,
                "score": score,
                "resume_score": calculate_resume_score(
                    resume["extracted_text"] or "",
                    resume["confidence_score"] or 0
                )[0],
                "resume_score_label": calculate_resume_score(
                    resume["extracted_text"] or "",
                    resume["confidence_score"] or 0
                )[1],
                "recommendation": build_skill_recommendation(missing),
                "verification_status": resume["verification_status"],
                "confidence_score": resume["confidence_score"]
            })

        candidate_results.sort(
            key=lambda item: (
                item["score"],
                item["confidence_score"] or 0
            ),
            reverse=True
        )

    # --------------------------------------------------------
    # JOB DESCRIPTION MATCHER
    # --------------------------------------------------------
    # Uses the existing job_description parameter and the same
    # supported-skill detection already used elsewhere.
    job_candidate_results = []

    if job_description:
        job_required_skills = extract_skills_from_job_description(
            job_description
        )

        # If the user has also entered required_skills, combine both
        # sources without duplicates.
        for item in canonical_required:
            if item not in job_required_skills:
                job_required_skills.append(item)

        if job_required_skills:
            cursor.execute(
                """
                SELECT *
                FROM documents
                WHERE user_id=?
                AND document_type='Resume'
                ORDER BY id DESC
                """,
                (session["user_id"],)
            )

            job_resumes = cursor.fetchall()

            for resume in job_resumes:
                resume_text = resume["extracted_text"] or ""

                matched, missing, match_score = calculate_skill_match(
                    resume_text,
                    job_required_skills
                )

                resume_score, resume_score_label, _ = calculate_resume_score(
                    resume_text,
                    resume["confidence_score"] or 0
                )

                job_candidate_results.append({
                    "id": resume["id"],
                    "filename": resume["filename"],
                    "matched": matched,
                    "missing": missing,
                    "score": match_score,
                    "resume_score": resume_score,
                    "resume_score_label": resume_score_label,
                    "recommendation": build_skill_recommendation(missing),
                    "verification_status": resume["verification_status"],
                    "confidence_score": resume["confidence_score"]
                })

            # Highest job match first, then resume score, then OCR confidence.
            job_candidate_results.sort(
                key=lambda item: (
                    item["score"],
                    item["resume_score"],
                    item["confidence_score"] or 0
                ),
                reverse=True
            )

            for rank, candidate in enumerate(
                job_candidate_results,
                start=1
            ):
                candidate["rank"] = rank

    else:
        job_required_skills = []

    conn.close()

    return render_template(
        "dashboard.html",

        name=current_user["name"] if current_user else session["user_name"],
        email=current_user["email"] if current_user else "",
        mobile=current_user["mobile"] if current_user else "",
        address=current_user["address"] if current_user else "",
        profile_photo=current_user["profile_photo"] if current_user else "",
        user_id=current_user["id"] if current_user else session["user_id"],


        documents=documents,

        search=search,

        total_documents=total_documents,

        verified_documents=verified_documents,

        rejected_documents=rejected_documents,

        resume_count=resume_count,

        aadhaar_count=aadhaar_count,

        pan_count=pan_count,

        average_confidence=average_confidence,

        recent_documents=recent_documents,

        # Skill Finder data
        skill=skill,

        skill_name=skill_name,

        skill_results=skill_results,

        skill_search_message=skill_search_message,

        # Multi-skill candidate matching
        required_skills_input=required_skills_input,

        required_skills=required_skills,

        candidate_results=candidate_results,

        # Job Description Matcher
        job_description=job_description,
        job_description_skills=job_required_skills,
        job_candidate_results=job_candidate_results,

        # Overall ranking information for future/compatible dashboard sections.
        top_candidate=candidate_results[0] if candidate_results else None
    )


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@app.route("/upload", methods=["GET", "POST"])
def upload():

    if "user_id" not in session:

        return redirect("/login")

    if request.method == "POST":

        # ---------------- CHECK FILE ---------------- #

        if "document" not in request.files:

            flash("No file selected!")

            return redirect("/upload")

        file = request.files["document"]

        if file.filename == "":

            flash("Please select a file!")

            return redirect("/upload")

        # ====================================================
        # SECURE ORIGINAL FILENAME
        # ====================================================

        original_filename = secure_filename(
            file.filename
        )

        # ====================================================
        # CHECK FILE TYPE
        # ====================================================

        if not allowed_file(original_filename):

            flash(
                "Invalid file type! "
                "Only PDF, JPG, JPEG and PNG are allowed."
            )

            return redirect("/upload")

        # ====================================================
        # CHECK FILE SIZE
        # ====================================================

        file.seek(
            0,
            os.SEEK_END
        )

        file_size = file.tell()

        file.seek(0)

        if file_size > MAX_FILE_SIZE:

            flash(
                "File is too large! "
                "Maximum allowed size is 5 MB."
            )

            return redirect("/upload")

        # ====================================================
        # CREATE UNIQUE FILENAME
        # ====================================================

        name, extension = os.path.splitext(
            original_filename
        )

        filename = (
            f"{name}_"
            f"{session['user_id']}_"
            f"{random.randint(1000, 9999)}"
            f"{extension}"
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        # ====================================================
        # SAVE FILE
        # ====================================================

        file.save(filepath)

        # ====================================================
        # OCR
        # ====================================================

        try:

            extracted_text = extract_text(
                filepath
            )

        except Exception as e:

            if os.path.exists(filepath):

                os.remove(filepath)

            flash(
                "OCR processing failed. "
                "Please check the document."
            )

            print("OCR ERROR:", e)

            return redirect("/upload")

        # ====================================================
        # DOCUMENT TYPE
        # ====================================================

        document_type = detect_document_type(
            extracted_text
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        summary = generate_summary(
            extracted_text
        )


        # ====================================================
        # NAME-BASED FILENAME
        # ====================================================

        new_filename = create_name_based_filename(
            extracted_text,
            document_type,
            original_filename
        )

        if new_filename != filename:
            new_filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                new_filename
            )

            os.replace(filepath, new_filepath)

            filename = new_filename
            filepath = new_filepath

        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = calculate_confidence(
            document_type
        )

        # ====================================================
        # VERIFICATION
        # ====================================================

        status = smart_verification(
            extracted_text,
            document_type
        )

        # ====================================================
        # DATABASE INSERT
        # ====================================================

        conn = get_db()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO documents(

                user_id,
                filename,
                extracted_text,
                verification_status,
                document_type,
                confidence_score,
                summary,
                download_count

            )

            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                session["user_id"],
                filename,
                extracted_text,
                status,
                document_type,
                confidence,
                summary,
                0
            )
        )

        conn.commit()

        conn.close()

        flash(
            "Document Uploaded Successfully!"
        )

        return redirect("/dashboard")

    return render_template(
        "upload.html"
    )


# ============================================================
# VIEW DOCUMENT
# ============================================================

@app.route("/document/<int:id>")
def view_document(id):

    if "user_id" not in session:

        return redirect("/login")

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM documents

        WHERE id=?
        AND user_id=?
        """,
        (
            id,
            session["user_id"]
        )
    )

    document = cursor.fetchone()

    conn.close()

    if document is None:

        flash(
            "Document Not Found"
        )

        return redirect("/dashboard")

    return render_template(
        "view_document.html",
        document=document
    )


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.route("/delete/<int:id>")
def delete_document(id):

    if "user_id" not in session:

        return redirect("/login")

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT filename
        FROM documents

        WHERE id=?
        AND user_id=?
        """,
        (
            id,
            session["user_id"]
        )
    )

    document = cursor.fetchone()

    if document:

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            document["filename"]
        )

        if os.path.exists(filepath):

            os.remove(filepath)

        cursor.execute(
            """
            DELETE FROM documents
            WHERE id=?
            """,
            (id,)
        )

        conn.commit()

    conn.close()

    flash(
        "Document Deleted Successfully!"
    )

    return redirect("/dashboard")


# ============================================================
# DOWNLOAD DOCUMENT
# ============================================================

@app.route("/download/<filename>")
def download_document(filename):

    if "user_id" not in session:

        return redirect("/login")

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM documents

        WHERE filename=?
        AND user_id=?
        """,
        (
            filename,
            session["user_id"]
        )
    )

    document = cursor.fetchone()

    if document is None:

        conn.close()

        flash(
            "Document Not Found"
        )

        return redirect("/dashboard")

    # ---------------- INCREASE DOWNLOAD COUNT ---------------- #

    cursor.execute(
        """
        UPDATE documents

        SET download_count =
            download_count + 1

        WHERE id=?
        """,
        (document["id"],)
    )

    conn.commit()

    conn.close()

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
        as_attachment=True
    )


# ============================================================
# PDF REPORT
# ============================================================

@app.route("/report/<int:id>")
def generate_report(id):

    if "user_id" not in session:

        return redirect("/login")

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM documents

        WHERE id=?
        AND user_id=?
        """,
        (
            id,
            session["user_id"]
        )
    )

    document = cursor.fetchone()

    conn.close()

    if document is None:

        flash(
            "Document Not Found"
        )

        return redirect("/dashboard")

    filename = f"Report_{id}.pdf"

    filepath = os.path.join(
        REPORT_FOLDER,
        filename
    )

    pdf = SimpleDocTemplate(
        filepath
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "<b>Document Verification Report</b>",
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            "<br/>",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Filename:</b> "
            f"{document['filename']}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Document Type:</b> "
            f"{document['document_type']}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Status:</b> "
            f"{document['verification_status']}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Confidence:</b> "
            f"{document['confidence_score']}%",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Upload Date:</b> "
            f"{document['upload_date']}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            "<br/>",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            "<b>OCR Extracted Text</b>",
            styles["Heading2"]
        )
    )

    extracted_text = document["extracted_text"] or "No text extracted."

    # Basic HTML safety for ReportLab Paragraph
    extracted_text = (
        extracted_text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )

    story.append(
        Paragraph(
            extracted_text,
            styles["Normal"]
        )
    )

    pdf.build(story)

    return send_from_directory(
        REPORT_FOLDER,
        filename,
        as_attachment=True
    )


# ============================================================
# JOB DESCRIPTION MATCH API
# ============================================================

@app.route("/api/job-match", methods=["GET", "POST"])
def job_match_api():
    """Return ranked resume matches for a job description.
    Accepts GET ?job_description=... or JSON/form field job_description.
    """
    if "user_id" not in session:
        return {"error": "Login required"}, 401

    job_description = ""
    if request.method == "POST":
        if request.is_json:
            data = request.get_json(silent=True) or {}
            job_description = str(data.get("job_description", "")).strip()
        else:
            job_description = request.form.get("job_description", "").strip()
    else:
        job_description = request.args.get("job_description", "").strip()

    required = extract_skills_from_job_description(job_description)
    if not required:
        return {
            "job_skills": [],
            "message": "No supported skills were detected in the job description.",
            "candidates": []
        }

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT * FROM documents
           WHERE user_id=? AND document_type='Resume'
           ORDER BY id DESC""",
        (session["user_id"],)
    )
    resumes = cursor.fetchall()
    conn.close()

    candidates = []
    for resume in resumes:
        matched, missing, score = calculate_skill_match(
            resume["extracted_text"] or "", required
        )
        resume_score, label, _ = calculate_resume_score(
            resume["extracted_text"] or "", resume["confidence_score"] or 0
        )
        candidates.append({
            "id": resume["id"],
            "filename": resume["filename"],
            "job_match": score,
            "matched_skills": matched,
            "missing_skills": missing,
            "resume_score": resume_score,
            "resume_score_label": label,
            "verification_status": resume["verification_status"],
            "confidence": resume["confidence_score"]
        })

    candidates.sort(
        key=lambda c: (c["job_match"], c["resume_score"], c["confidence"] or 0),
        reverse=True
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    return {
        "job_skills": required,
        "candidate_count": len(candidates),
        "candidates": candidates
    }


# ============================================================
# CANDIDATE REPORT / JOB MATCH REPORT
# ============================================================

@app.route("/candidate-report/<int:id>")
def candidate_report(id):
    """Generate a PDF recruitment report for one resume.
    Existing document/report routes remain unchanged.
    Optional query: ?skills=Java,Python,SQL
    """
    if "user_id" not in session:
        return redirect("/login")

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM documents
        WHERE id=? AND user_id=? AND document_type='Resume'
        """,
        (id, session["user_id"])
    )
    document = cursor.fetchone()
    conn.close()

    if document is None:
        flash("Resume Not Found")
        return redirect("/dashboard")

    requested = request.args.get("skills", "").strip()
    requested_skills = [x.strip() for x in requested.split(",") if x.strip()]
    if not requested_skills:
        requested_skills = detect_skills(document["extracted_text"] or "")

    matched, missing, match_score = calculate_skill_match(
        document["extracted_text"] or "", requested_skills
    )
    resume_score, resume_label, score_reasons = calculate_resume_score(
        document["extracted_text"] or "", document["confidence_score"] or 0
    )

    filename = f"Candidate_Report_{id}.pdf"
    filepath = os.path.join(REPORT_FOLDER, filename)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>DocVerify - Candidate Evaluation Report</b>", styles["Title"]))
    story.append(Paragraph("<br/>", styles["Normal"]))
    story.append(Paragraph(f"<b>Resume:</b> {html.escape(document['filename'] or '')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Document Status:</b> {html.escape(document['verification_status'] or '')}", styles["Normal"]))
    story.append(Paragraph(f"<b>OCR Confidence:</b> {document['confidence_score'] or 0}%", styles["Normal"]))
    story.append(Paragraph(f"<b>Resume Score:</b> {resume_score}/100 ({resume_label})", styles["Normal"]))
    story.append(Paragraph("<br/><b>Requested Skills</b>", styles["Heading2"]))
    story.append(Paragraph(html.escape(", ".join(requested_skills) if requested_skills else "No skills specified"), styles["Normal"]))
    story.append(Paragraph(f"<b>Skill Match:</b> {match_score}%", styles["Normal"]))
    story.append(Paragraph("<br/><b>Matched Skills</b>", styles["Heading2"]))
    story.append(Paragraph(html.escape(", ".join(matched) if matched else "None"), styles["Normal"]))
    story.append(Paragraph("<b>Missing Skills</b>", styles["Heading2"]))
    story.append(Paragraph(html.escape(", ".join(missing) if missing else "None"), styles["Normal"]))
    story.append(Paragraph("<b>Recommendation</b>", styles["Heading2"]))
    story.append(Paragraph(html.escape(build_skill_recommendation(missing)), styles["Normal"]))
    story.append(Paragraph("<b>Resume Score Factors</b>", styles["Heading2"]))
    story.append(Paragraph(html.escape("; ".join(score_reasons) if score_reasons else "No additional factors detected."), styles["Normal"]))

    pdf = SimpleDocTemplate(filepath)
    pdf.build(story)
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@app.route("/admin")
def admin():

    if "user_id" not in session:

        return redirect("/login")

    conn = get_db()

    cursor = conn.cursor()

    # ---------------- TOTAL USERS ---------------- #

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM users
        """
    )

    total_users = cursor.fetchone()["total"]

    # ---------------- TOTAL DOCUMENTS ---------------- #

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM documents
        """
    )

    total_documents = cursor.fetchone()["total"]

    # ---------------- VERIFIED ---------------- #

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM documents

        WHERE verification_status='Verified'
        """
    )

    verified_documents = cursor.fetchone()["total"]

    # ---------------- REJECTED ---------------- #

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM documents

        WHERE verification_status='Rejected'
        """
    )

    rejected_documents = cursor.fetchone()["total"]

    # ---------------- REGISTERED USERS ---------------- #

    cursor.execute(
        """
        SELECT id, name, email
        FROM users

        ORDER BY id DESC
        """
    )

    users = cursor.fetchall()

    # ---------------- ALL DOCUMENTS ---------------- #

    cursor.execute(
        """
        SELECT

            documents.id,

            documents.user_id,

            documents.filename,

            documents.document_type,

            documents.verification_status,

            documents.confidence_score,

            documents.upload_date,

            users.name

        FROM documents

        LEFT JOIN users

        ON documents.user_id = users.id

        ORDER BY documents.id DESC
        """
    )

    documents = cursor.fetchall()

    conn.close()

    return render_template(
        "admin.html",

        total_users=total_users,

        total_documents=total_documents,

        verified_documents=verified_documents,

        rejected_documents=rejected_documents,

        users=users,

        documents=documents
    )


# ============================================================
# ADMIN DELETE DOCUMENT
# ============================================================

@app.route("/admin/delete/<int:id>")
def admin_delete_document(id):

    if "user_id" not in session:

        return redirect("/login")

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT filename
        FROM documents

        WHERE id=?
        """,
        (id,)
    )

    document = cursor.fetchone()

    if document:

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            document["filename"]
        )

        if os.path.exists(filepath):

            os.remove(filepath)

        cursor.execute(
            """
            DELETE FROM documents
            WHERE id=?
            """,
            (id,)
        )

        conn.commit()

        flash(
            "Document deleted successfully!"
        )

    else:

        flash(
            "Document not found!"
        )

    conn.close()

    return redirect("/admin")


# ============================================================
# FILE TOO LARGE ERROR
# ============================================================

@app.errorhandler(413)
def request_entity_too_large(error):

    flash(
        "File is too large! Maximum allowed size is 5 MB."
    )

    return redirect("/upload")


# ============================================================
# USER PROFILE UPDATE
# ============================================================

@app.route("/profile/update", methods=["POST"])
def update_profile():

    if "user_id" not in session:
        return redirect("/login")

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    mobile = request.form.get("mobile", "").strip()
    address = request.form.get("address", "").strip()

    if not name or not email:
        flash("Name and email are required.")
        return redirect("/dashboard")

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id FROM users
        WHERE email=? AND id!=?
        """,
        (email, session["user_id"])
    )

    if cursor.fetchone():
        conn.close()
        flash("This email is already registered with another account.")
        return redirect("/dashboard")

    new_photo = None
    photo = request.files.get("profile_photo")

    if photo and photo.filename:
        safe_name = secure_filename(photo.filename)
        ext = os.path.splitext(safe_name)[1].lower()

        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            conn.close()
            flash("Profile photo must be JPG, JPEG, PNG or WEBP.")
            return redirect("/dashboard")

        photo.seek(0, os.SEEK_END)
        size = photo.tell()
        photo.seek(0)


        new_photo = (
            f"profile_{session['user_id']}_"
            f"{random.randint(1000, 9999)}{ext}"
        )

        photo.save(
            os.path.join(
                app.config["PROFILE_FOLDER"],
                new_photo
            )
        )

        cursor.execute(
            "SELECT profile_photo FROM users WHERE id=?",
            (session["user_id"],)
        )
        old = cursor.fetchone()

        if old and old["profile_photo"]:
            old_path = os.path.join(
                app.config["PROFILE_FOLDER"],
                old["profile_photo"]
            )
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass

    if new_photo:
        cursor.execute(
            """
            UPDATE users
            SET name=?, email=?, mobile=?, address=?, profile_photo=?
            WHERE id=?
            """,
            (name, email, mobile, address, new_photo, session["user_id"])
        )
    else:
        cursor.execute(
            """
            UPDATE users
            SET name=?, email=?, mobile=?, address=?
            WHERE id=?
            """,
            (name, email, mobile, address, session["user_id"])
        )

    conn.commit()
    conn.close()

    session["user_name"] = name
    flash("Profile updated successfully.")

    return redirect("/dashboard")


@app.route("/profile/photo/remove", methods=["POST"])
def remove_profile_photo():

    if "user_id" not in session:
        return redirect("/login")

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT profile_photo FROM users WHERE id=?",
        (session["user_id"],)
    )
    user = cursor.fetchone()

    if user and user["profile_photo"]:
        path = os.path.join(
            app.config["PROFILE_FOLDER"],
            user["profile_photo"]
        )
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    cursor.execute(
        "UPDATE users SET profile_photo='' WHERE id=?",
        (session["user_id"],)
    )

    conn.commit()
    conn.close()

    flash("Profile photo removed.")
    return redirect("/dashboard")


@app.route("/profile/photo/<filename>")
def profile_photo(filename):

    if "user_id" not in session:
        return redirect("/login")

    safe_filename = secure_filename(filename)

    if not safe_filename:
        return redirect("/dashboard")

    return send_from_directory(
        app.config["PROFILE_FOLDER"],
        safe_filename,
        max_age=0
    )


# Compatibility route for dashboard/profile image URLs.
# This does not change any existing functionality.
@app.route("/profile-photo/<filename>")
def profile_photo_compat(filename):

    if "user_id" not in session:
        return redirect("/login")

    safe_filename = secure_filename(filename)

    if not safe_filename:
        return redirect("/dashboard")

    return send_from_directory(
        app.config["PROFILE_FOLDER"],
        safe_filename,
        max_age=0
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")



# ============================================================
# EXTRA FEATURE 1 + 2
# ACTIVITY TIMELINE + NOTIFICATIONS
# ============================================================
# NOTE:
# These blocks are ADDITIVE. Existing routes, functions and
# document-verification logic above are kept unchanged.
# ============================================================

def create_activity_notification_tables():
    """Create the extra tables required for Activity Timeline and Notifications."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            notification_type TEXT DEFAULT 'info',
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def add_activity(user_id, action, description):
    """Add one item to the user's Activity Timeline."""
    if not user_id:
        return

    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO activity_logs(
                user_id, action, description
            )
            VALUES(?,?,?)
            """,
            (
                user_id,
                action,
                description
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("ACTIVITY LOG ERROR:", e)


def add_notification(
    user_id,
    title,
    message,
    notification_type="info"
):
    """Create one notification for the logged-in user."""
    if not user_id:
        return

    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO notifications(
                user_id,
                title,
                message,
                notification_type,
                is_read
            )
            VALUES(?,?,?,?,0)
            """,
            (
                user_id,
                title,
                message,
                notification_type
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("NOTIFICATION ERROR:", e)


# Create the extra tables when this code is loaded.
create_activity_notification_tables()


# ------------------------------------------------------------
# ADDITIVE EVENT TRACKING
# ------------------------------------------------------------
# This hook watches existing routes without replacing or editing
# those routes. It only creates Activity/Notification records
# after successful actions.
# ------------------------------------------------------------

@app.after_request
def extra_feature_activity_tracker(response):
    try:
        user_id = session.get("user_id")

        if not user_id:
            return response

        route_path = request.path
        method = request.method

        # ---------------- DOCUMENT UPLOAD ----------------
        if (
            route_path == "/upload"
            and method == "POST"
            and response.status_code in (200, 302, 303)
        ):
            conn = get_db()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT filename, document_type, verification_status
                FROM documents
                WHERE user_id=?
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,)
            )

            latest = cursor.fetchone()
            conn.close()

            if latest:
                filename = latest["filename"] or "Document"
                document_type = latest["document_type"] or "Document"
                status = latest["verification_status"] or "Processed"

                add_activity(
                    user_id,
                    "Document Uploaded",
                    f"{filename} was uploaded and processed as {document_type}."
                )

                notification_type = (
                    "success"
                    if status == "Verified"
                    else "warning"
                )

                add_notification(
                    user_id,
                    "Document Processed",
                    f"{filename} has been processed. Verification status: {status}.",
                    notification_type
                )

        # ---------------- DOCUMENT DELETE ----------------
        elif (
            route_path.startswith("/delete/")
            and method == "GET"
            and response.status_code in (200, 302, 303)
        ):
            document_id = route_path.rsplit("/", 1)[-1]

            add_activity(
                user_id,
                "Document Deleted",
                f"Document #{document_id} was deleted from your workspace."
            )

            add_notification(
                user_id,
                "Document Deleted",
                f"Document #{document_id} was removed from your workspace.",
                "warning"
            )

        # ---------------- PROFILE UPDATE ----------------
        elif (
            route_path == "/profile/update"
            and method == "POST"
            and response.status_code in (200, 302, 303)
        ):
            add_activity(
                user_id,
                "Profile Updated",
                "Your profile information was updated successfully."
            )

            add_notification(
                user_id,
                "Profile Updated",
                "Your profile information was updated successfully.",
                "success"
            )

        # ---------------- PROFILE PHOTO REMOVE ----------------
        elif (
            route_path == "/profile/photo/remove"
            and method == "POST"
            and response.status_code in (200, 302, 303)
        ):
            add_activity(
                user_id,
                "Profile Photo Removed",
                "Your profile photo was removed."
            )

            add_notification(
                user_id,
                "Profile Photo Removed",
                "Your profile photo has been removed.",
                "info"
            )

    except Exception as e:
        # Never allow the extra feature to break the existing app.
        print("EXTRA FEATURE TRACKER ERROR:", e)

    return response


# ============================================================
# NOTIFICATIONS API
# ============================================================

@app.route("/api/notifications")
def api_notifications():
    """Return recent notifications and unread count as JSON."""
    if "user_id" not in session:
        return {"error": "Login required"}, 401

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            title,
            message,
            notification_type,
            is_read,
            created_at
        FROM notifications
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT 20
        """,
        (session["user_id"],)
    )

    rows = cursor.fetchall()

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM notifications
        WHERE user_id=?
        AND is_read=0
        """,
        (session["user_id"],)
    )

    unread = cursor.fetchone()["total"]

    conn.close()

    return {
        "unread_count": unread,
        "notifications": [
            {
                "id": row["id"],
                "title": row["title"],
                "message": row["message"],
                "type": row["notification_type"],
                "is_read": bool(row["is_read"]),
                "created_at": row["created_at"]
            }
            for row in rows
        ]
    }


@app.route("/api/notifications/read/<int:id>", methods=["POST"])
def mark_notification_read(id):
    """Mark one notification as read."""
    if "user_id" not in session:
        return {"error": "Login required"}, 401

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE notifications
        SET is_read=1
        WHERE id=?
        AND user_id=?
        """,
        (
            id,
            session["user_id"]
        )
    )

    conn.commit()
    conn.close()

    return {
        "success": True,
        "notification_id": id
    }


@app.route("/api/notifications/read-all", methods=["POST"])
def mark_all_notifications_read():
    """Mark all notifications of the current user as read."""
    if "user_id" not in session:
        return {"error": "Login required"}, 401

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE notifications
        SET is_read=1
        WHERE user_id=?
        AND is_read=0
        """,
        (session["user_id"],)
    )

    conn.commit()
    conn.close()

    return {
        "success": True,
        "message": "All notifications marked as read."
    }


# ============================================================
# ACTIVITY TIMELINE API
# ============================================================

@app.route("/api/activity")
def api_activity():
    """Return the current user's recent Activity Timeline."""
    if "user_id" not in session:
        return {"error": "Login required"}, 401

    limit = request.args.get("limit", 20, type=int)

    # Keep the API safe even if someone enters a very large number.
    limit = max(1, min(limit, 100))

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        f"""
        SELECT
            id,
            action,
            description,
            created_at
        FROM activity_logs
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT {limit}
        """,
        (session["user_id"],)
    )

    rows = cursor.fetchall()
    conn.close()

    return {
        "activities": [
            {
                "id": row["id"],
                "action": row["action"],
                "description": row["description"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]
    }


# ============================================================
# ADDITIONAL PAGE DATA FOR DASHBOARD UI
# ============================================================

@app.context_processor
def extra_dashboard_feature_data():
    """
    Make notification/activity data available to dashboard.html
    without changing the existing dashboard route.
    """
    if "user_id" not in session:
        return {
            "dashboard_notifications": [],
            "dashboard_unread_notifications": 0,
            "dashboard_activity": []
        }

    try:
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                title,
                message,
                notification_type,
                is_read,
                created_at
            FROM notifications
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT 10
            """,
            (session["user_id"],)
        )

        notification_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM notifications
            WHERE user_id=?
            AND is_read=0
            """,
            (session["user_id"],)
        )

        unread_row = cursor.fetchone()

        cursor.execute(
            """
            SELECT
                id,
                action,
                description,
                created_at
            FROM activity_logs
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT 10
            """,
            (session["user_id"],)
        )

        activity_rows = cursor.fetchall()

        conn.close()

        return {
            "dashboard_notifications": notification_rows,
            "dashboard_unread_notifications": (
                unread_row["total"] if unread_row else 0
            ),
            "dashboard_activity": activity_rows
        }

    except Exception as e:
        print("DASHBOARD EXTRA DATA ERROR:", e)

        return {
            "dashboard_notifications": [],
            "dashboard_unread_notifications": 0,
            "dashboard_activity": []
        }


# ============================================================
# END OF ADDITIONAL FEATURES
# ============================================================



# ============================================================
# EXTRA FEATURE 3 + 4
# DUPLICATE DOCUMENT DETECTION + AI-STYLE RISK SCORE
# ============================================================
# These features are additive. Existing upload, OCR, verification,
# notifications and activity code are kept intact.
# ============================================================

import hashlib


def create_document_intelligence_table():
    """Create storage for calculated duplicate/risk information."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_intelligence(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL UNIQUE,
            content_hash TEXT,
            risk_score REAL DEFAULT 0,
            risk_level TEXT DEFAULT 'Low Risk',
            duplicate_flag INTEGER DEFAULT 0,
            duplicate_of INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


create_document_intelligence_table()


def normalize_document_text(text):
    """Normalize OCR text so harmless spacing/case differences do not
    prevent duplicate detection."""
    value = text or ""
    value = value.lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-z0-9 ]", "", value)
    return value.strip()


def calculate_document_hash(text):
    """Return a stable SHA-256 hash of normalized OCR text."""
    normalized = normalize_document_text(text)

    if len(normalized) < 30:
        return ""

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def calculate_document_risk(
    document,
    duplicate_flag=False
):
    """
    Calculate a transparent document risk percentage.
    Higher percentage = higher verification risk.

    This is a rules-based project feature, not a claim of
    real-world fraud detection.
    """
    risk = 0.0

    confidence = document["confidence_score"] or 0
    status = (document["verification_status"] or "").lower()
    doc_type = (document["document_type"] or "").lower()
    text = document["extracted_text"] or ""

    # OCR confidence
    if confidence < 50:
        risk += 30
    elif confidence < 70:
        risk += 20
    elif confidence < 85:
        risk += 10
    elif confidence < 95:
        risk += 4

    # Verification result
    if "rejected" in status:
        risk += 35
    elif "pending" in status:
        risk += 12
    elif "verified" in status:
        risk += 0

    # Document type detection
    if not doc_type or "unknown" in doc_type:
        risk += 20

    # OCR text quality
    if len(normalize_document_text(text)) < 80:
        risk += 15

    # Exact duplicate
    if duplicate_flag:
        risk += 25

    risk = max(0.0, min(100.0, round(risk, 2)))

    if risk >= 60:
        level = "High Risk"
    elif risk >= 30:
        level = "Medium Risk"
    else:
        level = "Low Risk"

    return risk, level


def refresh_document_intelligence(user_id):
    """
    Recalculate duplicate and risk information for all documents
    belonging to the current user.
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM documents
        WHERE user_id=?
        ORDER BY id ASC
    """, (user_id,))

    documents = cursor.fetchall()

    hashes = {}

    for doc in documents:
        content_hash = calculate_document_hash(
            doc["extracted_text"] or ""
        )

        if content_hash:
            hashes.setdefault(
                content_hash,
                []
            ).append(doc["id"])

    for doc in documents:
        content_hash = calculate_document_hash(
            doc["extracted_text"] or ""
        )

        matching_ids = (
            hashes.get(content_hash, [])
            if content_hash
            else []
        )

        # The earliest document is treated as the original.
        original_id = (
            matching_ids[0]
            if matching_ids
            else None
        )

        duplicate_flag = bool(
            len(matching_ids) > 1
            and doc["id"] != original_id
        )

        duplicate_of = (
            original_id
            if duplicate_flag
            else None
        )

        risk_score, risk_level = calculate_document_risk(
            doc,
            duplicate_flag
        )

        cursor.execute("""
            INSERT INTO document_intelligence(
                document_id,
                content_hash,
                risk_score,
                risk_level,
                duplicate_flag,
                duplicate_of,
                updated_at
            )
            VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(document_id)
            DO UPDATE SET
                content_hash=excluded.content_hash,
                risk_score=excluded.risk_score,
                risk_level=excluded.risk_level,
                duplicate_flag=excluded.duplicate_flag,
                duplicate_of=excluded.duplicate_of,
                updated_at=CURRENT_TIMESTAMP
        """, (
            doc["id"],
            content_hash,
            risk_score,
            risk_level,
            1 if duplicate_flag else 0,
            duplicate_of
        ))

    conn.commit()
    conn.close()


@app.route("/api/document-intelligence")
def api_document_intelligence():
    """Return risk and duplicate information for the user's documents."""
    if "user_id" not in session:
        return {"error": "Login required"}, 401

    refresh_document_intelligence(
        session["user_id"]
    )

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            d.id,
            d.filename,
            d.document_type,
            d.verification_status,
            d.confidence_score,
            d.upload_date,
            i.risk_score,
            i.risk_level,
            i.duplicate_flag,
            i.duplicate_of
        FROM documents d
        LEFT JOIN document_intelligence i
            ON i.document_id=d.id
        WHERE d.user_id=?
        ORDER BY d.id DESC
    """, (session["user_id"],))

    rows = cursor.fetchall()
    conn.close()

    documents = []

    for row in rows:
        documents.append({
            "id": row["id"],
            "filename": row["filename"],
            "document_type": row["document_type"],
            "verification_status": row["verification_status"],
            "confidence_score": row["confidence_score"],
            "upload_date": row["upload_date"],
            "risk_score": row["risk_score"] or 0,
            "risk_level": row["risk_level"] or "Low Risk",
            "duplicate": bool(row["duplicate_flag"]),
            "duplicate_of": row["duplicate_of"]
        })

    return {
        "documents": documents,
        "duplicate_count": sum(
            1 for item in documents
            if item["duplicate"]
        ),
        "high_risk_count": sum(
            1 for item in documents
            if item["risk_level"] == "High Risk"
        ),
        "medium_risk_count": sum(
            1 for item in documents
            if item["risk_level"] == "Medium Risk"
        ),
        "low_risk_count": sum(
            1 for item in documents
            if item["risk_level"] == "Low Risk"
        )
    }


@app.route("/api/document-intelligence/<int:id>")
def api_single_document_intelligence(id):
    """Return detailed risk/duplicate information for one document."""
    if "user_id" not in session:
        return {"error": "Login required"}, 401

    refresh_document_intelligence(
        session["user_id"]
    )

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            d.id,
            d.filename,
            d.document_type,
            d.verification_status,
            d.confidence_score,
            d.upload_date,
            i.risk_score,
            i.risk_level,
            i.duplicate_flag,
            i.duplicate_of
        FROM documents d
        LEFT JOIN document_intelligence i
            ON i.document_id=d.id
        WHERE d.id=?
        AND d.user_id=?
    """, (
        id,
        session["user_id"]
    ))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"error": "Document not found"}, 404

    return {
        "id": row["id"],
        "filename": row["filename"],
        "document_type": row["document_type"],
        "verification_status": row["verification_status"],
        "confidence_score": row["confidence_score"],
        "upload_date": row["upload_date"],
        "risk_score": row["risk_score"] or 0,
        "risk_level": row["risk_level"] or "Low Risk",
        "duplicate": bool(row["duplicate_flag"]),
        "duplicate_of": row["duplicate_of"]
    }


# ============================================================
# END DUPLICATE + RISK SCORE
# ============================================================



# ============================================================
# NEW FEATURE: COMPLETE DOCUMENT VERIFICATION CENTER
# ============================================================
# IMPORTANT:
# The existing project code above is intentionally kept unchanged.
# This section is additive and provides a separate page containing
# every uploaded document with duplicate, similarity, suspicious
# information, tampering-indicator and final-result analysis.
# ============================================================

from difflib import SequenceMatcher


def calculate_text_similarity(text_a, text_b):
    """Return OCR-text similarity percentage for two documents."""
    a = normalize_document_text(text_a)
    b = normalize_document_text(text_b)

    if len(a) < 30 or len(b) < 30:
        return 0

    return round(SequenceMatcher(None, a, b).ratio() * 100, 2)


def document_authenticity_analysis(document, all_documents):
    """
    Project-level authenticity analysis.

    This does not claim to prove real-world forgery. It flags documents
    as suspicious when the available OCR/verification evidence is weak,
    inconsistent, duplicated or highly similar to another document.
    """
    text = document["extracted_text"] or ""
    normalized = normalize_document_text(text)
    doc_type = document["document_type"] or "Unknown"
    status = (document["verification_status"] or "").lower()
    confidence = document["confidence_score"] or 0

    exact_duplicate = False
    duplicate_of = None
    highest_similarity = 0
    similar_document_id = None

    current_hash = calculate_document_hash(text)

    for other in all_documents:
        if other["id"] == document["id"]:
            continue

        other_text = other["extracted_text"] or ""

        # Exact normalized OCR duplicate
        if current_hash:
            other_hash = calculate_document_hash(other_text)
            if other_hash and current_hash == other_hash:
                if other["id"] < document["id"]:
                    exact_duplicate = True
                    duplicate_of = other["id"]

        # Similar-but-not-identical OCR content
        similarity = calculate_text_similarity(text, other_text)
        if similarity > highest_similarity:
            highest_similarity = similarity
            similar_document_id = other["id"]

    suspicious_reasons = []

    if exact_duplicate:
        suspicious_reasons.append("Exact OCR content matches an earlier document")

    if highest_similarity >= 85 and not exact_duplicate:
        suspicious_reasons.append(
            f"Very high content similarity ({highest_similarity:.0f}%) "
            "with another uploaded document"
        )

    if "rejected" in status:
        suspicious_reasons.append("Existing verification rules rejected the document")

    if confidence < 70:
        suspicious_reasons.append("Low OCR/document confidence")

    if doc_type == "Unknown":
        suspicious_reasons.append("Document type could not be identified")

    if len(normalized) < 80:
        suspicious_reasons.append("Very little readable information was extracted")

    # Document-specific information checks.
    # These are consistency checks, not government database verification.
    information_issue = False

    if doc_type == "Aadhaar Card":
        has_number = bool(re.search(
            r"(?<!\d)(?:\d{4}[ -]?\d{4}[ -]?\d{4})(?!\d)",
            text
        ))
        has_keyword = any(
            word in (text.lower())
            for word in ["aadhaar", "aadhar", "uidai"]
        )
        if not has_number or not has_keyword:
            information_issue = True
            suspicious_reasons.append(
                "Aadhaar identifying information is incomplete or inconsistent"
            )

    elif doc_type == "PAN Card":
        has_pan = bool(re.search(
            r"(?<![A-Za-z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Za-z0-9])",
            text.upper()
        ))
        if not has_pan:
            information_issue = True
            suspicious_reasons.append(
                "PAN identifying information is incomplete or inconsistent"
            )

    elif doc_type == "Invoice":
        invoice_text = text.lower()
        if "invoice" not in invoice_text or "gst" not in invoice_text:
            information_issue = True
            suspicious_reasons.append(
                "Expected invoice/GST information is incomplete"
            )

    elif doc_type == "Resume":
        resume_text = text.lower()
        required_resume_parts = [
            "education",
            "skills",
            "project"
        ]
        missing_parts = [
            part for part in required_resume_parts
            if part not in resume_text
        ]
        if missing_parts:
            information_issue = True
            suspicious_reasons.append(
                "Expected resume sections are missing: "
                + ", ".join(missing_parts)
            )

    elif doc_type == "Marksheet":
        marksheet_text = text.lower()
        if "roll" not in marksheet_text and "marks" not in marksheet_text:
            information_issue = True
            suspicious_reasons.append(
                "Expected marksheet information is incomplete"
            )

    # Tampering is intentionally described as an indicator.
    # OCR-only data cannot prove image manipulation.
    tampering_indicator = (
        highest_similarity >= 85
        or confidence < 60
        or information_issue
        or "rejected" in status
    )

    if exact_duplicate:
        final_result = "Duplicate Document"
        result_class = "danger"
    elif "rejected" in status and (information_issue or confidence < 70):
        final_result = "Possible Fake / Suspicious"
        result_class = "danger"
    elif tampering_indicator:
        final_result = "Suspicious / Possible Tampering"
        result_class = "warning"
    else:
        final_result = "Not Duplicate / No Major Suspicion"
        result_class = "success"

    return {
        "exact_duplicate": exact_duplicate,
        "duplicate_of": duplicate_of,
        "similarity": highest_similarity,
        "similar_document_id": similar_document_id,
        "information_issue": information_issue,
        "tampering_indicator": tampering_indicator,
        "suspicious_reasons": suspicious_reasons,
        "final_result": final_result,
        "result_class": result_class
    }


@app.route("/document-verification")
def document_verification_center():
    """Show a complete verification analysis for every uploaded document."""
    if "user_id" not in session:
        return redirect("/login")

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM documents
        WHERE user_id=?
        ORDER BY id DESC
    """, (session["user_id"],))

    rows = cursor.fetchall()

    # Convert rows to dictionaries so the analysis can safely compare
    # documents without changing the existing documents table.
    all_documents = [dict(row) for row in rows]

    results = []

    for document in all_documents:
        analysis = document_authenticity_analysis(
            document,
            all_documents
        )

        # Combine the existing project risk score with the new checks.
        existing_risk = 0
        cursor.execute("""
            SELECT risk_score
            FROM document_intelligence
            WHERE document_id=?
        """, (document["id"],))
        intelligence = cursor.fetchone()

        if intelligence:
            existing_risk = float(intelligence["risk_score"] or 0)

        extra_risk = 0

        if analysis["exact_duplicate"]:
            extra_risk += 30

        if analysis["similarity"] >= 85:
            extra_risk += 20
        elif analysis["similarity"] >= 70:
            extra_risk += 10

        if analysis["information_issue"]:
            extra_risk += 15

        if analysis["tampering_indicator"]:
            extra_risk += 10

        final_risk = min(
            100,
            round(max(existing_risk, 0) + extra_risk, 2)
        )

        if final_risk >= 60:
            final_risk_level = "High Risk"
        elif final_risk >= 30:
            final_risk_level = "Medium Risk"
        else:
            final_risk_level = "Low Risk"

        result = dict(document)
        result.update(analysis)
        result["risk_score"] = final_risk
        result["risk_level"] = final_risk_level

        results.append(result)

    conn.close()

    counts = {
        "total": len(results),
        "duplicate": sum(
            1 for item in results if item["exact_duplicate"]
        ),
        "suspicious": sum(
            1 for item in results
            if item["final_result"] != "Not Duplicate / No Major Suspicion"
            and not item["exact_duplicate"]
        ),
        "safe": sum(
            1 for item in results
            if item["final_result"] == "Not Duplicate / No Major Suspicion"
        )
    }

    return render_template(
        "document_verification.html",
        documents=results,
        counts=counts
    )


# ============================================================
# END NEW DOCUMENT VERIFICATION CENTER
# ============================================================


# ============================================================
# ADDED FEATURE: SIGNATURE REGISTRATION + VERIFICATION
# Existing application code above is intentionally untouched.
# This is a project-level signature similarity screening feature.
# ============================================================

try:
    import cv2
    import numpy as np
    SIGNATURE_ENGINE_AVAILABLE = True
except Exception as signature_import_error:
    cv2 = None
    np = None
    SIGNATURE_ENGINE_AVAILABLE = False
    print("SIGNATURE ENGINE NOT AVAILABLE:", signature_import_error)

SIGNATURE_FOLDER = os.path.join(UPLOAD_FOLDER, "signatures")
os.makedirs(SIGNATURE_FOLDER, exist_ok=True)

SIGNATURE_ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def create_signature_tables():
    """Create signature reference and verification tables without altering
    the existing users/documents tables."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signature_profiles(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            person_name TEXT NOT NULL,
            person_identifier TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, person_identifier)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signature_references(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            sample_label TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signature_verifications(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            document_id INTEGER,
            submitted_image_path TEXT NOT NULL,
            similarity_score REAL DEFAULT 0,
            result TEXT NOT NULL,
            details TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


create_signature_tables()


def _signature_extension(filename):
    extension = os.path.splitext(filename or "")[1].lower().lstrip(".")
    return extension if extension in SIGNATURE_ALLOWED_EXTENSIONS else None


def _signature_unique_filename(original_name, prefix="signature"):
    extension = _signature_extension(original_name) or "png"
    safe = secure_filename(os.path.splitext(original_name or "signature")[0])
    safe = safe[:40] or prefix
    token = f"{random.randint(100000, 999999)}"
    return f"{prefix}_{safe}_{token}.{extension}"


def _signature_crop_and_normalize(image_path, target_width=320, target_height=160):
    """Convert a signature image into a centered binary ink mask."""
    if not SIGNATURE_ENGINE_AVAILABLE:
        raise RuntimeError("OpenCV/NumPy is not installed")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("The uploaded signature image could not be read.")

    # Keep very large photos manageable.
    h, w = image.shape[:2]
    if max(h, w) > 1800:
        scale = 1800.0 / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove tiny scanner/camera noise.
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    coords = cv2.findNonZero(binary)
    if coords is None:
        raise ValueError("No visible signature ink was detected. Upload a clear signature image.")

    x, y, bw, bh = cv2.boundingRect(coords)
    pad = max(8, int(max(bw, bh) * 0.08))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(binary.shape[1], x + bw + pad)
    y2 = min(binary.shape[0], y + bh + pad)
    cropped = binary[y1:y2, x1:x2]

    # Preserve aspect ratio while fitting a fixed canvas.
    canvas = np.zeros((target_height, target_width), dtype=np.uint8)
    ch, cw = cropped.shape[:2]
    scale = min((target_width - 20) / max(cw, 1), (target_height - 20) / max(ch, 1))
    nw = max(1, int(cw * scale))
    nh = max(1, int(ch * scale))
    resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)
    ox = (target_width - nw) // 2
    oy = (target_height - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _signature_similarity(reference_path, submitted_path):
    """Return a 0-100 similarity score using normalized shape + pixel evidence."""
    reference = _signature_crop_and_normalize(reference_path)
    submitted = _signature_crop_and_normalize(submitted_path)

    # Pixel/ink overlap after centering and normalization.
    ref_float = reference.astype(np.float32) / 255.0
    sub_float = submitted.astype(np.float32) / 255.0
    pixel_error = float(np.mean(np.abs(ref_float - sub_float)))
    pixel_score = max(0.0, min(100.0, (1.0 - pixel_error) * 100.0))

    # Contour shape comparison gives some tolerance to stroke thickness.
    ref_contours, _ = cv2.findContours(reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sub_contours, _ = cv2.findContours(submitted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_score = 0.0
    if ref_contours and sub_contours:
        ref_contour = max(ref_contours, key=cv2.contourArea)
        sub_contour = max(sub_contours, key=cv2.contourArea)
        ref_area = cv2.contourArea(ref_contour)
        sub_area = cv2.contourArea(sub_contour)
        if ref_area > 5 and sub_area > 5:
            match = float(cv2.matchShapes(ref_contour, sub_contour, cv2.CONTOURS_MATCH_I1, 0.0))
            shape_score = max(0.0, min(100.0, (1.0 - min(match / 0.50, 1.0)) * 100.0))

    score = round((pixel_score * 0.75) + (shape_score * 0.25), 2)
    return score


def _signature_result(score):
    if score >= 85:
        return "SIGNATURE VERIFIED", "High similarity with registered genuine signature samples."
    if score >= 70:
        return "MANUAL REVIEW REQUIRED", "Moderate similarity. The signature should be reviewed before acceptance."
    return "POSSIBLE FORGED SIGNATURE", "Low similarity with the registered genuine signature samples."


def _current_user_id():
    return session.get("user_id")


@app.route("/signature/register", methods=["GET", "POST"])
def register_signature():
    if "user_id" not in session:
        return redirect("/login")

    if request.method == "GET":
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.id, p.person_name, p.person_identifier,
                   COUNT(r.id) AS sample_count
            FROM signature_profiles p
            LEFT JOIN signature_references r ON r.profile_id=p.id
            WHERE p.user_id=?
            GROUP BY p.id
            ORDER BY p.id DESC
        """, (session["user_id"],))
        profiles = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return render_template("document_verification.html", signature_mode="register", signature_profiles=profiles)

    if not SIGNATURE_ENGINE_AVAILABLE:
        flash("Signature engine unavailable. Install opencv-python-headless and numpy first.")
        return redirect("/document-verification")

    person_name = request.form.get("person_name", "").strip()
    person_identifier = request.form.get("person_identifier", "").strip()
    files = [f for f in request.files.getlist("signature_samples") if f and f.filename]

    if not person_name or not person_identifier:
        flash("Person name and unique ID are required for signature registration.")
        return redirect("/document-verification")

    if len(files) < 3 or len(files) > 5:
        flash("Please register 3 to 5 genuine signature samples for a reliable project-level profile.")
        return redirect("/document-verification")

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id FROM signature_profiles
        WHERE user_id=? AND person_identifier=?
    """, (session["user_id"], person_identifier))
    profile = cursor.fetchone()

    if profile:
        profile_id = profile["id"]
        cursor.execute("DELETE FROM signature_references WHERE profile_id=?", (profile_id,))
    else:
        cursor.execute("""
            INSERT INTO signature_profiles(user_id, person_name, person_identifier)
            VALUES(?,?,?)
        """, (session["user_id"], person_name, person_identifier))
        profile_id = cursor.lastrowid

    saved = 0
    try:
        for index, uploaded in enumerate(files, start=1):
            if not _signature_extension(uploaded.filename):
                raise ValueError(f"Invalid signature format: {uploaded.filename}")
            filename = _signature_unique_filename(uploaded.filename, "reference")
            relative_path = os.path.join("signatures", filename).replace("\\", "/")
            absolute_path = os.path.join(UPLOAD_FOLDER, relative_path)
            uploaded.save(absolute_path)
            _signature_crop_and_normalize(absolute_path)  # validate before registration
            cursor.execute("""
                INSERT INTO signature_references(profile_id, user_id, image_path, sample_label)
                VALUES(?,?,?,?)
            """, (profile_id, session["user_id"], relative_path, f"Genuine Sample {index}"))
            saved += 1

        cursor.execute("""
            UPDATE signature_profiles SET person_name=? WHERE id=? AND user_id=?
        """, (person_name, profile_id, session["user_id"]))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("SIGNATURE REGISTRATION ERROR:", e)
        flash(f"Signature registration failed: {e}")
        conn.close()
        return redirect("/document-verification")

    conn.close()
    flash(f"Signature profile registered successfully with {saved} genuine samples.")
    return redirect("/document-verification")


@app.route("/signature/verify", methods=["POST"])
def verify_signature():
    if "user_id" not in session:
        return redirect("/login")

    if not SIGNATURE_ENGINE_AVAILABLE:
        flash("Signature engine unavailable. Install opencv-python-headless and numpy first.")
        return redirect("/document-verification")

    profile_id = request.form.get("profile_id", type=int)
    document_id = request.form.get("document_id", type=int)
    submitted = request.files.get("submitted_signature")

    if not profile_id or not submitted or not submitted.filename:
        flash("Select a registered person and upload the submitted signature.")
        return redirect("/document-verification")

    if not _signature_extension(submitted.filename):
        flash("Only PNG, JPG, JPEG or WEBP signature images are supported.")
        return redirect("/document-verification")

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM signature_profiles
        WHERE id=? AND user_id=?
    """, (profile_id, session["user_id"]))
    profile = cursor.fetchone()

    if not profile:
        conn.close()
        flash("Registered signature profile not found.")
        return redirect("/document-verification")

    cursor.execute("""
        SELECT * FROM signature_references
        WHERE profile_id=? AND user_id=?
        ORDER BY id ASC
    """, (profile_id, session["user_id"]))
    references = cursor.fetchall()

    if not references:
        conn.close()
        flash("No genuine reference signatures are registered for this person.")
        return redirect("/document-verification")

    filename = _signature_unique_filename(submitted.filename, "submitted")
    relative_path = os.path.join("signatures", filename).replace("\\", "/")
    absolute_path = os.path.join(UPLOAD_FOLDER, relative_path)
    submitted.save(absolute_path)

    scores = []
    try:
        for reference in references:
            reference_path = os.path.join(UPLOAD_FOLDER, reference["image_path"])
            if os.path.exists(reference_path):
                scores.append(_signature_similarity(reference_path, absolute_path))
    except Exception as e:
        print("SIGNATURE VERIFICATION ERROR:", e)
        conn.close()
        flash(f"Signature verification failed: {e}")
        return redirect("/document-verification")

    if not scores:
        conn.close()
        flash("Reference signature files are missing. Please register the genuine samples again.")
        return redirect("/document-verification")

    scores.sort(reverse=True)
    # Best match is useful for natural signing variation; average the best two
    # when two or more genuine samples exist to reduce a single-sample outlier.
    top_scores = scores[:2]
    final_score = round(sum(top_scores) / len(top_scores), 2)
    result, details = _signature_result(final_score)

    cursor.execute("""
        INSERT INTO signature_verifications(
            profile_id, user_id, document_id, submitted_image_path,
            similarity_score, result, details
        ) VALUES(?,?,?,?,?,?,?)
    """, (
        profile_id,
        session["user_id"],
        document_id,
        relative_path,
        final_score,
        result,
        details
    ))
    conn.commit()
    conn.close()

    flash(f"{result} — Similarity Score: {final_score}%")
    return redirect("/document-verification")


@app.route("/api/signature-summary")
def signature_summary():
    if "user_id" not in session:
        return {"error": "login_required"}, 401

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) AS profile_count
        FROM signature_profiles WHERE user_id=?
    """, (session["user_id"],))
    profile_count = cursor.fetchone()["profile_count"]

    cursor.execute("""
        SELECT COUNT(*) AS sample_count
        FROM signature_references WHERE user_id=?
    """, (session["user_id"],))
    sample_count = cursor.fetchone()["sample_count"]

    cursor.execute("""
        SELECT
            COUNT(*) AS total_checks,
            SUM(CASE WHEN result='SIGNATURE VERIFIED' THEN 1 ELSE 0 END) AS verified,
            SUM(CASE WHEN result='POSSIBLE FORGED SIGNATURE' THEN 1 ELSE 0 END) AS suspicious,
            SUM(CASE WHEN result='MANUAL REVIEW REQUIRED' THEN 1 ELSE 0 END) AS review
        FROM signature_verifications WHERE user_id=?
    """, (session["user_id"],))
    row = cursor.fetchone()

    cursor.execute("""
        SELECT v.similarity_score, v.result, v.created_at,
               p.person_name, p.person_identifier, v.document_id
        FROM signature_verifications v
        JOIN signature_profiles p ON p.id=v.profile_id
        WHERE v.user_id=?
        ORDER BY v.id DESC LIMIT 8
    """, (session["user_id"],))
    recent = [dict(item) for item in cursor.fetchall()]
    conn.close()

    return {
        "engine_available": SIGNATURE_ENGINE_AVAILABLE,
        "profile_count": profile_count or 0,
        "sample_count": sample_count or 0,
        "total_checks": row["total_checks"] or 0,
        "verified": row["verified"] or 0,
        "suspicious": row["suspicious"] or 0,
        "review": row["review"] or 0,
        "recent": recent
    }


# ============================================================
# ADDED FEATURE: SIGNATURE PROFILE TEMPLATE CONTEXT
# This supplies registered profiles to the existing verification page.
# ============================================================
@app.context_processor
def signature_profile_template_context():
    profiles = []
    try:
        user_id = session.get("user_id")
        if user_id:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.id, p.person_name, p.person_identifier,
                       COUNT(r.id) AS sample_count
                FROM signature_profiles p
                LEFT JOIN signature_references r ON r.profile_id=p.id
                WHERE p.user_id=?
                GROUP BY p.id
                ORDER BY p.id DESC
            """, (user_id,))
            profiles = [dict(row) for row in cursor.fetchall()]
            conn.close()
    except Exception as e:
        print("SIGNATURE CONTEXT ERROR:", e)
    return {"signature_profiles": profiles}


# ============================================================
# END ADDED SIGNATURE FEATURE
# ============================================================



# ============================================================
# ============================================================
# ============================================================
# ADDED FEATURE: EMAIL OTP TWO-STEP LOGIN
# Existing project code above is kept unchanged.
# ============================================================
import secrets
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta

OTP_EMAIL_SENDER = os.environ.get("DOCVERIFY_EMAIL_SENDER", "").strip()
OTP_EMAIL_APP_PASSWORD = os.environ.get("DOCVERIFY_EMAIL_APP_PASSWORD", "").strip()
OTP_SMTP_HOST = os.environ.get("DOCVERIFY_SMTP_HOST", "smtp.gmail.com").strip()

try:
    OTP_SMTP_PORT = int(os.environ.get("DOCVERIFY_SMTP_PORT", "587"))
except ValueError:
    OTP_SMTP_PORT = 587

OTP_EXPIRY_MINUTES = 5
OTP_MAX_ATTEMPTS = 5


def _get_docverify_otp_email_config():
    sender = os.environ.get(
        "DOCVERIFY_EMAIL_SENDER", OTP_EMAIL_SENDER
    ).strip()

    # Google displays App Passwords with spaces; remove them safely.
    app_password = os.environ.get(
        "DOCVERIFY_EMAIL_APP_PASSWORD", OTP_EMAIL_APP_PASSWORD
    ).strip().replace(" ", "")

    host = os.environ.get(
        "DOCVERIFY_SMTP_HOST", OTP_SMTP_HOST
    ).strip()

    try:
        port = int(os.environ.get(
            "DOCVERIFY_SMTP_PORT", str(OTP_SMTP_PORT)
        ))
    except ValueError:
        port = 587

    return sender, app_password, host, port


def _send_docverify_login_otp(recipient_email, recipient_name, otp_code):
    sender, app_password, host, port = _get_docverify_otp_email_config()

    if not sender or not app_password:
        raise RuntimeError(
            "Gmail OTP configuration is missing. Set "
            "DOCVERIFY_EMAIL_SENDER and DOCVERIFY_EMAIL_APP_PASSWORD."
        )

    message = EmailMessage()
    message["Subject"] = "DocVerify Login Verification Code"
    message["From"] = sender
    message["To"] = recipient_email
    message.set_content(
        f"""Hello {recipient_name},

Your DocVerify login verification code is: {otp_code}

This code is valid for {OTP_EXPIRY_MINUTES} minutes.
If you did not try to log in, you can safely ignore this email.

Regards,
DocVerify Security Team"""
    )

    with smtplib.SMTP(host, port, timeout=20) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender, app_password)
        server.send_message(message)


def _clear_docverify_otp_session():
    session.pop("otp_pending_user_id", None)
    session.pop("otp_pending_email", None)
    session.pop("otp_code", None)
    session.pop("otp_expires_at", None)
    session.pop("otp_attempts", None)


def _create_docverify_login_otp(user):
    otp_code = f"{secrets.randbelow(1000000):06d}"
    expires_at = datetime.now() + timedelta(minutes=OTP_EXPIRY_MINUTES)

    email_sent = False

    try:
        _send_docverify_login_otp(
            user["email"],
            user["name"],
            otp_code
        )
        email_sent = True
        print(f"EMAIL OTP SENT SUCCESSFULLY TO: {user['email']}")

    except Exception as exc:
        # IMPORTANT FIX:
        # Do NOT redirect back to /login when SMTP fails.
        # Keep the OTP flow alive so login cannot get stuck in a loop.
        print("EMAIL OTP ERROR:", exc)
        print("Use a Gmail App Password, not your normal Gmail password.")
        print("=" * 60)
        print("DOCVERIFY LOCAL OTP FALLBACK")
        print(f"Email: {user['email']}")
        print(f"OTP  : {otp_code}")
        print(f"Valid for: {OTP_EXPIRY_MINUTES} minutes")
        print("Fix Gmail SMTP credentials to receive OTP by email.")
        print("=" * 60)

    # Ensure normal authentication happens only after OTP verification.
    session.pop("user_id", None)
    session.pop("user_name", None)

    session["otp_pending_user_id"] = user["id"]
    session["otp_pending_email"] = user["email"]
    session["otp_code"] = otp_code
    session["otp_expires_at"] = expires_at.isoformat()
    session["otp_attempts"] = 0
    session.modified = True

    return email_sent


@app.before_request
def docverify_email_otp_login_gate():
    """Require email OTP after valid email/password login."""

    if request.endpoint == "static":
        return None

    # Intercept only the existing login form submission.
    if request.path != "/login" or request.method != "POST":
        return None

    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    if not email or not password:
        flash("Please enter your email and password.")
        return redirect("/login")

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE LOWER(email)=LOWER(?)",
        (email,)
    )
    user = cursor.fetchone()
    conn.close()

    if user is None or not check_password_hash(
        user["password"],
        password
    ):
        flash("Invalid Email or Password")
        return redirect("/login")

    # Valid credentials -> create OTP and go to OTP page.
    _create_docverify_login_otp(user)

    flash(
        f"A 6-digit verification code was generated for "
        f"{user['email']}."
    )

    return redirect("/verify-login-otp")


@app.route("/verify-login-otp", methods=["GET", "POST"])
def verify_login_otp():
    """Verify OTP and create the authenticated dashboard session."""

    pending_user_id = session.get("otp_pending_user_id")
    pending_email = session.get("otp_pending_email")
    saved_otp = session.get("otp_code")
    expires_raw = session.get("otp_expires_at")

    if (
        not pending_user_id
        or not pending_email
        or not saved_otp
        or not expires_raw
    ):
        flash("Please log in with your email and password first.")
        return redirect("/login")

    try:
        expires_at = datetime.fromisoformat(expires_raw)
    except (ValueError, TypeError):
        expires_at = datetime.min

    if datetime.now() >= expires_at:
        _clear_docverify_otp_session()
        flash("The verification code has expired. Please log in again.")
        return redirect("/login")

    if request.method == "POST":
        entered_otp = request.form.get("otp", "").strip()

        attempts = int(session.get("otp_attempts", 0)) + 1
        session["otp_attempts"] = attempts

        if (
            len(entered_otp) == 6
            and entered_otp.isdigit()
            and secrets.compare_digest(entered_otp, saved_otp)
        ):
            conn = get_db()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id, name, email FROM users WHERE id=?",
                (pending_user_id,)
            )
            user = cursor.fetchone()
            conn.close()

            if user is None:
                session.clear()
                flash(
                    "User account could not be found. "
                    "Please log in again."
                )
                return redirect("/login")

            _clear_docverify_otp_session()

            # This is the only point where authenticated session is created.
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session.modified = True

            flash("Email verified successfully. Login successful.")
            return redirect("/dashboard")

        if attempts >= OTP_MAX_ATTEMPTS:
            _clear_docverify_otp_session()
            flash(
                "Too many incorrect verification attempts. "
                "Please log in again."
            )
            return redirect("/login")

        remaining = OTP_MAX_ATTEMPTS - attempts
        flash(
            f"Incorrect verification code. "
            f"{remaining} attempts remaining."
        )

    return render_template(
        "verify_login_otp.html",
        otp_email=pending_email,
        otp_expiry_minutes=OTP_EXPIRY_MINUTES
    )


@app.route("/resend-login-otp", methods=["POST"])
def resend_login_otp():
    """Generate and send a fresh login OTP."""

    pending_user_id = session.get("otp_pending_user_id")

    if not pending_user_id:
        flash("Please log in with your email and password first.")
        return redirect("/login")

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, name, email FROM users WHERE id=?",
        (pending_user_id,)
    )
    user = cursor.fetchone()
    conn.close()

    if user is None:
        session.clear()
        flash(
            "User account could not be found. "
            "Please log in again."
        )
        return redirect("/login")

    _create_docverify_login_otp(user)

    flash(
        "A new OTP was generated. "
        "If Gmail SMTP is configured correctly, it has been sent to your email."
    )

    return redirect("/verify-login-otp")


# ============================================================
# END ADDED EMAIL OTP LOGIN FEATURE
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
