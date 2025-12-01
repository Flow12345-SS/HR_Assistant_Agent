import numpy as np
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------ GLOBAL SETUP ------------------

st.set_page_config(page_title="TalentSphere AI", page_icon="🧠", layout="wide")

st.sidebar.title("🧠 TalentSphere AI")
st.sidebar.write("Smart Multi-Agent System for HR & People Operations")

# Load embeddings (shared by all agents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Try loading vector store for HR policy
hr_db = None
try:
    hr_db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    st.sidebar.error(
        "HR Vector Store not found.\n\n"
        "👉 Run `python ingest.py` in your terminal after placing `hr_policy.pdf` in the `data/` folder."
    )

# Sidebar navigation
agent_choice = st.sidebar.radio(
    "Choose an Agent:",
    [
        "HR Policy Assistant",
        "Resume Screening Agent",
        "Interview Agent",
        "Onboarding Assistant",
    ]
)

st.title("🧠 TalentSphere AI – Multi-Agent HR System")
st.caption("Category: People & HR – Policy, Hiring, Interviews & Onboarding in one AI-powered workspace.")


# HR Policy Agent - Only latest question & answer
def hr_policy_agent():
    st.subheader("🏢 HR Policy Assistant")
    st.write("Ask any question related to leave, benefits, timings, rules, etc.")

    if hr_db is None:
        st.warning("Vector store not loaded. Please run `python ingest.py` first.")
        return

    query = st.text_input("Ask your HR policy question here:")

    if st.button("Ask HR Agent 🚀", key="ask_hr") and query:

        docs = hr_db.similarity_search_with_score(query, k=5)

        best_answer = ""
        best_score = 999

        for doc, score in docs:
            if score < best_score:
                best_score = score
                best_answer = doc.page_content

        if not best_answer.strip():
            best_answer = "Sorry, I couldn't find the exact answer in the HR policy."

        # 🔴 Clear previous question and add only new one
        st.session_state.hr_chat = [
            ("You", query),
            ("HR Assistant", best_answer)
        ]

    # Show only the latest question/answer
    if "hr_chat" in st.session_state:
        for speaker, msg in st.session_state.hr_chat:
            st.markdown(f"**{speaker}:** {msg}")


# ==================================================
# 2️⃣ RESUME SCREENING AGENT
# ==================================================
def resume_screening_agent():
    st.subheader("📄 Resume Screening Agent")
    st.write("Paste the **job description** and multiple **resumes**. The agent will rank resumes by best fit.")

    jd_text = st.text_area("Job Description", height=150, placeholder="Paste the job description here...")

    st.markdown("**Candidate Resumes** (separate each resume with a line containing `---`):")
    resumes_raw = st.text_area(
        "Resumes",
        height=250,
        placeholder="Resume 1 text...\n\n---\n\nResume 2 text...\n\n---\n\nResume 3 text...",
    )

    if st.button("Rank Resumes ✅"):
        if not jd_text.strip() or not resumes_raw.strip():
            st.warning("Please provide both job description and at least one resume.")
            return

        # Split resumes by ---
        resumes = [r.strip() for r in resumes_raw.split("---") if r.strip()]
        if len(resumes) == 0:
            st.warning("No valid resumes found. Make sure you separate them using `---`.")
            return

        # Get embeddings
        jd_embedding = np.array(embeddings.embed_query(jd_text))
        resume_embeddings = embeddings.embed_documents(resumes)

        scores = []
        for idx, emb in enumerate(resume_embeddings):
            emb = np.array(emb)
            # cosine similarity
            sim = float(np.dot(jd_embedding, emb) / (np.linalg.norm(jd_embedding) * np.linalg.norm(emb)))
            scores.append((idx + 1, sim, resumes[idx]))

        # Sort by score desc
        scores.sort(key=lambda x: x[1], reverse=True)

        st.markdown("### 🏆 Ranked Resumes (Best match on top)")
        for rank, (idx, sim, resume_text) in enumerate(scores, start=1):
            st.markdown(f"**#{rank} – Candidate {idx}**  \nSimilarity score: `{sim:.3f}`")
            with st.expander(f"Show Resume {idx}"):
                st.write(resume_text)


# ==================================================
# 3️⃣ INTERVIEW AGENT
# ==================================================
def interview_agent():
    st.subheader("🎤 Interview Agent")
    st.write("Choose a role, then the agent will ask questions one by one and give simple feedback at the end.")

    roles = {
        "Software Engineer": [
            "Tell me about a challenging bug you fixed.",
            "How do you ensure code quality in a large project?",
            "Explain the difference between multithreading and multiprocessing.",
        ],
        "Data Analyst": [
            "How do you handle missing values in a dataset?",
            "Explain a project where you used data to drive a business decision.",
            "What is the difference between correlation and causation?",
        ],
        "HR Executive": [
            "How do you handle conflicts between employees?",
            "Describe a time you improved an HR process.",
            "How would you design an employee engagement initiative?",
        ],
    }

    role = st.selectbox("Select role for mock interview:", list(roles.keys()))

    if "interview_state" not in st.session_state:
        st.session_state.interview_state = {
            "active": False,
            "role": None,
            "q_index": 0,
            "answers": [],
        }

    state = st.session_state.interview_state

    if st.button("Start / Restart Interview 🔁"):
        state["active"] = True
        state["role"] = role
        state["q_index"] = 0
        state["answers"] = []

    if not state["active"]:
        st.info("Click **Start / Restart Interview** to begin.")
        return

    questions = roles[state["role"]]
    q_index = state["q_index"]

    if q_index < len(questions):
        st.markdown(f"**Question {q_index + 1}:** {questions[q_index]}")
        answer = st.text_area("Your answer:", key=f"answer_{q_index}")

        if st.button("Submit Answer ✅", key=f"submit_{q_index}"):
            if not answer.strip():
                st.warning("Please type an answer before submitting.")
            else:
                state["answers"].append(answer.strip())
                state["q_index"] += 1
                st.rerun()
    else:
        # All questions answered – simple evaluation
        st.success("Interview completed! 🎉")
        st.markdown("### 📊 Feedback Summary")

        total_score = 0
        detailed_feedback = []

        for i, ans in enumerate(state["answers"]):
            length = len(ans.split())
            if length < 20:
                fb = "Answer is too short. Try to give more details and examples."
                score = 5
            elif length < 60:
                fb = "Decent answer, but you can add more structure and concrete examples."
                score = 7
            else:
                fb = "Strong, detailed answer with good explanation."
                score = 9

            total_score += score
            detailed_feedback.append((i + 1, score, fb))

        avg_score = total_score / len(state["answers"])

        st.markdown(f"**Overall Score:** `{avg_score:.1f} / 10`")

        for q_no, score, fb in detailed_feedback:
            st.markdown(f"**Q{q_no} – Score:** {score}/10")
            st.write(f"Feedback: {fb}")
            st.write("---")


# ==================================================
# 4️⃣ ONBOARDING ASSISTANT
# ==================================================
def onboarding_agent():
    st.subheader("🧭 Employee Onboarding Assistant")
    st.write("Helps new hires understand what to do in their first days at the company.")

    st.markdown("### ✅ Standard Onboarding Checklist")

    steps = [
        "Complete HR forms (personal details, bank, ID proof, tax forms).",
        "Submit educational and experience documents.",
        "Receive laptop/system and set up company accounts (email, HR portal, Slack/Teams).",
        "Read HR policy, code of conduct, and security guidelines.",
        "Meet your manager and team; understand your role and expectations.",
        "Get access to required tools (Jira, Git, internal tools etc.).",
        "Complete mandatory trainings (security, compliance, workplace safety).",
        "Schedule 1:1 with manager at end of first week for feedback.",
    ]

    for i, step in enumerate(steps, start=1):
        st.markdown(f"- **Day 1–{i+1}:** {step}")

    st.markdown("### ❓Ask a specific onboarding question")

    faq = [
        (["documents", "docs", "submit"], "You usually need to submit ID proof, address proof, PAN, bank details, education certificates, and previous employment documents."),
        (["laptop", "system", "device"], "IT will generally provide a laptop on Day 1 or 2. If delayed, contact the IT helpdesk or your manager."),
        (["probation", "confirmation"], "Most companies keep new hires on probation for 3–6 months. Confirmation depends on performance and manager feedback."),
        (["leave", "vacation", "holiday"], "During probation, leave may be limited. Check HR policy or ask your HR SPOC for exact details."),
    ]

    question = st.text_input("Your onboarding question:")

    if st.button("Ask Onboarding Agent 💬"):
        if not question.strip():
            st.warning("Please type a question.")
            return

        q_low = question.lower()
        answer = None
        for keywords, resp in faq:
            if any(k in q_low for k in keywords):
                answer = resp
                break

        if answer is None:
            answer = "I don't have an exact answer for that, but you can check with your HR SPOC or refer to the onboarding email/HR portal."

        st.markdown(f"**Onboarding Agent:** {answer}")


# ==================================================
# ROUTING – SHOW SELECTED AGENT
# ==================================================
if agent_choice == "HR Policy Assistant":
    hr_policy_agent()
elif agent_choice == "Resume Screening Agent":
    resume_screening_agent()
elif agent_choice == "Interview Agent":
    interview_agent()
elif agent_choice == "Onboarding Assistant":
    onboarding_agent()

