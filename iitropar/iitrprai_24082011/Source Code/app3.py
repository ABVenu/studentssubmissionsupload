import os
import sqlite3
import streamlit as st
from openai import OpenAI
import PyPDF2
import hashlib
from dotenv import load_dotenv

# ---------- CONFIG ----------
DB_PATH = "db/tutor.db"
os.makedirs("db", exist_ok=True)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- DATABASE ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            raw_text TEXT NOT NULL,
            layer0 TEXT,
            layer1 TEXT,
            layer2 TEXT,
            layer3 TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            content_id INTEGER,
            layer INTEGER,
            signal TEXT,
            understood INTEGER DEFAULT 0,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS quiz (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            content_id INTEGER,
            layer INTEGER,
            question TEXT,
            options TEXT,
            correct_answer TEXT,
            user_answer TEXT,
            is_correct INTEGER,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# ---------- STUDENT AUTH ----------
def register_student(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO students (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def login_student(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM students WHERE username=? AND password_hash=?", (username, password_hash))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

# ---------- DB OPERATIONS ----------
def save_content(student_id, raw_text, layers):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO content (student_id, raw_text, layer0, layer1, layer2, layer3) VALUES (?, ?, ?, ?, ?, ?)",
        (student_id, raw_text, layers[0], layers[1], layers[2], layers[3])
    )
    content_id = c.lastrowid
    conn.commit()
    conn.close()
    return content_id

def save_feedback(student_id, content_id, layer, signal, understood=0):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO feedback (student_id, content_id, layer, signal, understood) VALUES (?, ?, ?, ?, ?)",
        (student_id, content_id, layer, signal, understood)
    )
    conn.commit()
    conn.close()

def save_quiz(student_id, content_id, layer, question, options, correct, user_ans, is_correct):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO quiz (student_id, content_id, layer, question, options, correct_answer, user_answer, is_correct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (student_id, content_id, layer, question, options, correct, user_ans, is_correct))
    conn.commit()
    conn.close()

def get_dashboard_data(student_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT c.id, substr(c.raw_text, 1, 60) as snippet,
               COUNT(f.id) as interactions,
               SUM(f.understood) as understood_count,
               AVG(q.is_correct) as avg_quiz
        FROM content c
        LEFT JOIN feedback f ON c.id = f.content_id AND f.student_id=?
        LEFT JOIN quiz q ON c.id = q.content_id AND q.student_id=?
        WHERE c.student_id=?
        GROUP BY c.id
        ORDER BY c.id DESC
    """, (student_id, student_id, student_id))
    rows = c.fetchall()
    conn.close()
    return rows

def get_feedback_history(student_id, content_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT layer, signal, understood, ts FROM feedback WHERE student_id=? AND content_id=? ORDER BY ts DESC", (student_id, content_id))
    rows = c.fetchall()
    conn.close()
    return rows

def get_quiz_history(student_id, content_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT question, user_answer, correct_answer, is_correct, ts FROM quiz WHERE student_id=? AND content_id=? ORDER BY ts DESC", (student_id, content_id))
    rows = c.fetchall()
    conn.close()
    return rows

# ---------- PDF EXTRACT ----------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ---------- OPENAI CALLS ----------
def generate_layers(text):
    prompts = [
        "Explain this text to a child in very simple terms:\n\n",
        "Explain this text to a high-school student:\n\n",
        "Explain this text to a college student, including definitions:\n\n",
        "Explain this text to an expert, with technical detail:\n\n"
    ]
    layers = []
    for p in prompts:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful tutor."},
                {"role": "user", "content": p + text}
            ],
            temperature=0.7
        )
        layers.append(resp.choices[0].message.content)
    return layers

def generate_quiz(layer_text, num_qs=5):
    prompt = f"""
    Create {num_qs} multiple-choice questions from this explanation:
    {layer_text}
    Format exactly like this:
    Q: <question text>
    A) option1
    B) option2
    C) option3
    D) option4
    Answer: B
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    return resp.choices[0].message.content

# ---------- STREAMLIT APP ----------
init_db()
st.set_page_config(page_title="Cognitive Distillation Tutor", layout="wide")

if "student_id" not in st.session_state:
    st.session_state.student_id = None

if not st.session_state.student_id:
    st.title("🔐 Student Login")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            sid = login_student(username, password)
            if sid:
                st.session_state.student_id = sid
                st.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
    with tab2:
        new_user = st.text_input("New Username", key="reg_user")
        new_pass = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register"):
            if register_student(new_user, new_pass):
                st.success("🎉 Registration successful! Please log in.")
            else:
                st.error("⚠️ Username already exists")

else:
    st.sidebar.write(f"👤 Logged in as Student #{st.session_state.student_id}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    page = st.sidebar.radio("Go to:", ["Tutor", "Dashboard"])

    # ---------- TUTOR ----------
    if page == "Tutor":
        st.title("🧠 Cognitive Distillation Tutor")

        if "current_layer" not in st.session_state:
            st.session_state.current_layer = 0
        if "layers" not in st.session_state:
            st.session_state.layers = []
        if "content_id" not in st.session_state:
            st.session_state.content_id = None

        st.subheader("Input Content")
        user_text = st.text_area("Paste text here:", height=150)
        uploaded_pdf = st.file_uploader("Or upload a PDF", type=["pdf"])

        if st.button("Distill Content"):
            raw_text = extract_text_from_pdf(uploaded_pdf) if uploaded_pdf else user_text.strip()
            if not raw_text:
                st.warning("Please provide text or upload a PDF!")
            else:
                with st.spinner("Distilling into multiple layers..."):
                    layers = generate_layers(raw_text)
                    content_id = save_content(st.session_state.student_id, raw_text, layers)
                    st.session_state.layers = layers
                    st.session_state.current_layer = 0
                    st.session_state.content_id = content_id
                st.success("✅ Distillation complete!")

        if st.session_state.layers:
            i = st.session_state.current_layer
            st.subheader(f"Layer {i} Explanation")
            st.write(st.session_state.layers[i])

            col1, col2, col3 = st.columns(3)
            if col1.button("Too Simple"):
                if i < 3:
                    st.session_state.current_layer += 1
                save_feedback(st.session_state.student_id, st.session_state.content_id, i, "too_simple")
            if col2.button("Understood"):
                st.success("🎉 Marked as understood!")
                save_feedback(st.session_state.student_id, st.session_state.content_id, i, "understood", understood=1)
            if col3.button("Too Complex"):
                if i > 0:
                    st.session_state.current_layer -= 1
                save_feedback(st.session_state.student_id, st.session_state.content_id, i, "too_complex")

            # QUIZ
            st.subheader("📝 Quick Quiz")
            if st.button("Generate Quiz for this layer"):
                quiz_text = generate_quiz(st.session_state.layers[i], num_qs=5)
                st.text(quiz_text)

    # ---------- DASHBOARD ----------
    elif page == "Dashboard":
        st.title("📊 Learning Dashboard")
        data = get_dashboard_data(st.session_state.student_id)
        if not data:
            st.info("No learning history yet. Distill some content first!")
        else:
            for row in data:
                content_id, snippet, interactions, understood_count, avg_quiz = row
                mastery_score = 0 if interactions == 0 else round(understood_count / interactions, 2)
                quiz_score = round(avg_quiz, 2) if avg_quiz is not None else "N/A"
                with st.expander(f"Content #{content_id} – {snippet}..."):
                    st.write(f"**Interactions:** {interactions}")
                    st.write(f"**Understood count:** {understood_count}")
                    st.write(f"**Mastery score:** {mastery_score}")
                    st.write(f"**Average quiz accuracy:** {quiz_score}")

                    history = get_feedback_history(st.session_state.student_id, content_id)
                    if history:
                        st.markdown("**Feedback history:**")
                        for h in history:
                            layer, signal, understood, ts = h
                            st.write(f"- Layer {layer}, Signal: {signal}, Understood: {understood}, Time: {ts}")

                    q_hist = get_quiz_history(st.session_state.student_id, content_id)
                    if q_hist:
                        st.markdown("**Quiz history:**")
                        for qh in q_hist:
                            q, ua, ca, ic, ts = qh
                            result = "✅" if ic else "❌"
                            st.write(f"- {result} Q: {q} | Your answer: {ua} | Correct: {ca} | {ts}")
