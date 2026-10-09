"""
EduGuard AI: Early Warning System & AI Study Coach
SDG 4: Quality Education

Run on your computer:   streamlit run app.py
Run online:             deploy this folder from GitHub on Streamlit Community Cloud

Files this app needs next to it:
  model.pkl, model_info.json, StudentPerformanceFactors.csv   (made by the Colab notebook)
  sample_class.csv                                            (demo class list for the Teacher Dashboard)
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------------------
# 1. PAGE SETUP
# ------------------------------------------------------------------------------------------
st.set_page_config(page_title="EduGuard AI", page_icon="🎓", layout="wide")

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
RISK_COLORS = {"High risk": "#d03b3b", "Needs attention": "#e0a100", "On track": "#1f9d55"}

# Gemini models to try, in order (all have a free tier). If Google renames models later,
# set GEMINI_MODEL in Streamlit secrets and that one is tried first.
GEMINI_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash"]

# The 19 inputs the model expects, and the choices for each text input
CHOICES = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["No", "Yes"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Internet_Access": ["No", "Yes"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "School_Type": ["Public", "Private"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Learning_Disabilities": ["No", "Yes"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
    "Gender": ["Male", "Female"],
}
FEATURES = ["Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
            "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
            "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality", "School_Type",
            "Peer_Influence", "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
            "Distance_from_Home", "Gender"]
# A "typical" student, used to fill any column missing from an uploaded class list
TYPICAL = {"Hours_Studied": 20, "Attendance": 80, "Sleep_Hours": 7, "Previous_Scores": 75,
           "Tutoring_Sessions": 1, "Physical_Activity": 3, "Parental_Involvement": "Medium",
           "Access_to_Resources": "Medium", "Extracurricular_Activities": "Yes", "Motivation_Level": "Medium",
           "Internet_Access": "Yes", "Family_Income": "Medium", "Teacher_Quality": "Medium",
           "School_Type": "Public", "Peer_Influence": "Neutral", "Learning_Disabilities": "No",
           "Parental_Education_Level": "High School", "Distance_from_Home": "Near", "Gender": "Female"}


# ------------------------------------------------------------------------------------------
# 2. LOAD THE MODEL (trained in Colab). If it can't be loaded, retrain it quickly from the CSV.
# ------------------------------------------------------------------------------------------
@st.cache_resource          # load only once, not on every click
def load_model():
    try:
        model = joblib.load("model.pkl")
        with open("model_info.json") as f:
            info = json.load(f)
        return model, info
    except Exception:
        # Backup plan: train a Linear Regression model right here (takes ~1 second)
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        df = pd.read_csv("StudentPerformanceFactors.csv")
        for col in ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]:
            df[col] = df[col].fillna(df[col].mode()[0])
        df["Exam_Score"] = df["Exam_Score"].clip(upper=100)
        X, y = df[FEATURES], df["Exam_Score"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        text_cols = [c for c in FEATURES if c in CHOICES]
        model = Pipeline([("prep", ColumnTransformer([("text", OneHotEncoder(handle_unknown="ignore"), text_cols)],
                                                     remainder="passthrough")),
                          ("model", LinearRegression())]).fit(X_tr, y_tr)
        p = model.predict(X_te)
        info = {"best_model": "Linear Regression", "r2": round(r2_score(y_te, p), 3),
                "mae": round(mean_absolute_error(y_te, p), 2), "n_students": len(df),
                "at_risk_cutoff": float(y.quantile(0.25)), "watch_cutoff": float(y.quantile(0.5)),
                "all_results": [], "feature_importance": {}}
        return model, info


@st.cache_data
def load_scores():
    """All real exam scores, used for the 'where you stand' chart."""
    try:
        return pd.read_csv("StudentPerformanceFactors.csv")["Exam_Score"].clip(upper=100)
    except Exception:
        return pd.Series(dtype=float)


model, info = load_model()
AT_RISK, WATCH = info["at_risk_cutoff"], info["watch_cutoff"]


# ------------------------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------------------------------------
def predict(student: dict) -> float:
    """Predict the exam score for one student (a dict of the 19 inputs)."""
    return float(np.clip(model.predict(pd.DataFrame([student])[FEATURES])[0], 0, 100))


def risk_level(score: float) -> str:
    if score <= AT_RISK:
        return "High risk"
    if score <= WATCH:
        return "Needs attention"
    return "On track"


def step_up(value, options):
    """Move a Low/Medium/High style value one step up (if possible)."""
    i = options.index(value)
    return options[min(i + 1, len(options) - 1)]


def what_if(student: dict):
    """Try realistic improvements one at a time and measure how much each one raises the predicted score."""
    base = predict(student)
    changes = {
        "Study 5 more hours per week": ("Hours_Studied", min(student["Hours_Studied"] + 5, 44)),
        "Raise attendance by 10%": ("Attendance", min(student["Attendance"] + 10, 100)),
        "Attend 2 more tutoring sessions a month": ("Tutoring_Sessions", min(student["Tutoring_Sessions"] + 2, 8)),
        "Sleep 7-8 hours a night": ("Sleep_Hours", 8 if student["Sleep_Hours"] < 7 else student["Sleep_Hours"]),
        "Boost motivation one level": ("Motivation_Level", step_up(student["Motivation_Level"], CHOICES["Motivation_Level"])),
        "Get parents more involved": ("Parental_Involvement", step_up(student["Parental_Involvement"], CHOICES["Parental_Involvement"])),
        "Use more learning resources (library, online)": ("Access_to_Resources", step_up(student["Access_to_Resources"], CHOICES["Access_to_Resources"])),
        "Spend time with study-focused friends": ("Peer_Influence", step_up(student["Peer_Influence"], CHOICES["Peer_Influence"])),
        "Join an extracurricular activity": ("Extracurricular_Activities", "Yes"),
        "Add 1 more hour of exercise per week": ("Physical_Activity", min(student["Physical_Activity"] + 1, 6)),
    }
    rows = []
    for action, (col, new_value) in changes.items():
        if new_value == student[col]:
            continue                              # already at the best level, nothing to suggest
        changed = {**student, col: new_value}
        gain = predict(changed) - base
        if gain > 0.05:
            rows.append({"Action": action, "Score gain": round(gain, 2)})
    return pd.DataFrame(rows).sort_values("Score gain", ascending=False) if rows else pd.DataFrame(columns=["Action", "Score gain"])


def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return st.session_state.get("user_api_key", "")


def ask_gemini(prompt: str) -> str:
    """Send a prompt to Google Gemini and return the reply text. Tries several models in turn."""
    from google import genai                      # Google's official library: pip install google-genai
    client = genai.Client(api_key=get_api_key())
    models = list(GEMINI_MODELS)
    try:
        if "GEMINI_MODEL" in st.secrets:
            models.insert(0, st.secrets["GEMINI_MODEL"])
    except Exception:
        pass
    last_error = None
    for name in models:
        try:
            reply = client.models.generate_content(model=name, contents=prompt)
            if reply.text:
                return reply.text
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Gemini did not respond: {last_error}")


def offline_plan(student, score, level, tips_df, name, weeks, subjects):
    """A rule-based study plan, used when no AI key is set or the AI is unavailable."""
    lines = [f"### Personalised plan for {name or 'you'}",
             f"**Predicted score:** {score:.1f} · **Status:** {level} · **Exam in:** {weeks} week(s)", ""]
    lines.append("**Your top 3 priorities**")
    for _, r in tips_df.head(3).iterrows():
        lines.append(f"- {r['Action']} (about +{r['Score gain']:.1f} marks)")
    lines += ["", "**Weekly routine**",
              f"- Study at least **{max(student['Hours_Studied'] + 5, 20)} hours/week**: about {max(student['Hours_Studied'] + 5, 20) // 6} hours a day, 6 days a week, with one rest day.",
              "- Use 45-minute focus blocks with 10-minute breaks (Pomodoro).",
              f"- Spend extra time on: **{subjects or 'your weakest subjects'}**. Start each session with them while you are fresh.",
              "- Attend every class. Attendance is the strongest predictor in our model.",
              "- Sleep 7-8 hours. Revise lightly before bed, not late-night cramming.",
              "- End each week with a self-test (past papers or 20 practice questions).", "",
              "*Generated in offline mode (no AI connection). Add a working Gemini API key in the sidebar for a fully AI-written plan and chat.*"]
    return "\n".join(lines)


def student_summary(student, score, level, tips_df):
    tips = "; ".join(f"{r['Action']} (+{r['Score gain']:.1f})" for _, r in tips_df.head(5).iterrows())
    details = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in student.items())
    return (f"Student details: {details}.\nML model predicted exam score: {score:.1f}/100. Risk level: {level} "
            f"(High risk ≤ {AT_RISK:.0f}, Needs attention ≤ {WATCH:.0f}).\n"
            f"Improvements ranked by the ML model's what-if analysis: {tips}.")


# ------------------------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------------------
st.sidebar.title("🎓 EduGuard AI")
st.sidebar.caption("Early warning + AI study coach · SDG 4 Quality Education")
page = st.sidebar.radio("Go to", ["🏠 Home", "🎯 Student Risk Check", "🤖 AI Study Coach",
                                  "🏫 Teacher Dashboard", "📊 Model Insights"])
st.sidebar.divider()
def secret_key_set():
    try:
        return "GEMINI_API_KEY" in st.secrets
    except Exception:
        return False


if not secret_key_set():
    # The text box always stays on screen (if it disappeared, Streamlit would forget the key)
    st.sidebar.text_input("Gemini API key (optional)", type="password", key="user_api_key",
                          help="Free key from aistudio.google.com. Without it, the coach runs in offline mode.")
st.sidebar.caption("🤖 AI coach: " + ("key added ✅" if get_api_key() else "offline mode"))
st.sidebar.caption(f"Model: {info['best_model']} · trained on {info['n_students']:,} students")


# ------------------------------------------------------------------------------------------
# 5. PAGES
# ------------------------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("EduGuard AI 🎓")
    st.subheader("Spot struggling students early. Give every student a personal study coach.")
    st.write("Many students fall behind quietly. By the time results come out, it is too late to help. "
             "**EduGuard AI** uses machine learning to predict a student's exam score *weeks before the exam*, "
             "flags who is at risk, explains **why**, and an AI coach turns that into a personal study plan.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students in training data", f"{info['n_students']:,}")
    c2.metric("Model accuracy (R²)", f"{info['r2']:.2f}")
    c3.metric("Average error", f"±{info['mae']:.1f} marks")
    c4.metric("Factors analysed", "19")
    st.divider()
    a, b, c, d = st.columns(4)
    a.markdown("#### 🎯 Risk Check\nEnter study habits and get a predicted score and risk level.")
    b.markdown("#### 🔍 What-if Analysis\nSee which changes would raise your score the most.")
    c.markdown("#### 🤖 AI Study Coach\nGet a personalised plan and ask follow-up questions.")
    d.markdown("#### 🏫 Teacher Dashboard\nUpload a class list and see every at-risk student at once.")
    st.info("👈 Start with **Student Risk Check** in the sidebar.")

elif page == "🎯 Student Risk Check":
    st.title("🎯 Student Risk Check")
    st.write("Fill in the details and click **Predict**. Nothing is stored.")
    with st.form("student_form"):
        name = st.text_input("Student name (optional)", value=st.session_state.get("name", ""))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📚 Study habits**")
            hours = st.slider("Hours studied per week", 1, 44, 15)
            attendance = st.slider("Attendance (%)", 60, 100, 75)
            previous = st.slider("Previous exam score", 50, 100, 70)
            tutoring = st.slider("Tutoring sessions per month", 0, 8, 1)
            motivation = st.select_slider("Motivation level", CHOICES["Motivation_Level"], "Medium")
        with col2:
            st.markdown("**🌙 Lifestyle**")
            sleep = st.slider("Sleep hours per night", 4, 10, 7)
            activity = st.slider("Physical activity (hours/week)", 0, 6, 3)
            extra = st.radio("Extracurricular activities?", CHOICES["Extracurricular_Activities"], 1, horizontal=True)
            peers = st.select_slider("Friends' influence on studies", CHOICES["Peer_Influence"], "Neutral")
            disability = st.radio("Learning disability?", CHOICES["Learning_Disabilities"], 0, horizontal=True)
        with col3:
            st.markdown("**🏠 Home & school**")
            parents = st.select_slider("Parental involvement", CHOICES["Parental_Involvement"], "Medium")
            resources = st.select_slider("Access to resources", CHOICES["Access_to_Resources"], "Medium")
            internet = st.radio("Internet access?", CHOICES["Internet_Access"], 1, horizontal=True)
            income = st.select_slider("Family income", CHOICES["Family_Income"], "Medium")
            teacher = st.select_slider("Teacher quality", CHOICES["Teacher_Quality"], "Medium")
            school = st.radio("School type", CHOICES["School_Type"], 0, horizontal=True)
            par_edu = st.selectbox("Parents' education", CHOICES["Parental_Education_Level"])
            distance = st.select_slider("Distance from home", CHOICES["Distance_from_Home"], "Near")
            gender = st.radio("Gender", CHOICES["Gender"], 1, horizontal=True)
        submitted = st.form_submit_button("🔮 Predict", type="primary", width="stretch")

    if submitted:
        student = {"Hours_Studied": hours, "Attendance": attendance, "Parental_Involvement": parents,
                   "Access_to_Resources": resources, "Extracurricular_Activities": extra, "Sleep_Hours": sleep,
                   "Previous_Scores": previous, "Motivation_Level": motivation, "Internet_Access": internet,
                   "Tutoring_Sessions": tutoring, "Family_Income": income, "Teacher_Quality": teacher,
                   "School_Type": school, "Peer_Influence": peers, "Physical_Activity": activity,
                   "Learning_Disabilities": disability, "Parental_Education_Level": par_edu,
                   "Distance_from_Home": distance, "Gender": gender}
        score = predict(student)
        # remember this student for the AI Coach page
        st.session_state.update(student=student, score=score, name=name, chat=[], plan=None)

    if "student" in st.session_state:
        student, score = st.session_state["student"], st.session_state["score"]
        level = risk_level(score)
        tips = what_if(student)
        st.divider()
        left, right = st.columns([1, 1.4])
        with left:
            st.metric("Predicted exam score", f"{score:.1f} / 100")
            icon = {"High risk": "🔴", "Needs attention": "🟠", "On track": "🟢"}[level]
            msg = {"High risk": "This student is in the bottom 25%. Early support is strongly recommended.",
                   "Needs attention": "Below the average student. Small changes can make a big difference.",
                   "On track": "Above average. Keep up the good habits!"}[level]
            {"High risk": st.error, "Needs attention": st.warning, "On track": st.success}[level](f"{icon} **{level}**: {msg}")
        with right:
            scores = load_scores()
            if len(scores):
                fig, ax = plt.subplots(figsize=(6, 2.4))
                ax.hist(scores, bins=40, range=(55, 100), color="#c9daf3", edgecolor="white")
                ax.axvline(score, color=ORANGE, linewidth=2.5)
                ax.text(score + 0.6, ax.get_ylim()[1] * 0.85, "You", color=ORANGE, fontweight="bold")
                ax.set_title("Where this student stands among 6,600+ students", fontsize=10, color=GREY)
                ax.set_yticks([]); ax.set_xlabel("Exam score", color=GREY)
                for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
                st.pyplot(fig, width="stretch")
        st.subheader("🔍 What-if: which changes would help most?")
        if len(tips):
            fig, ax = plt.subplots(figsize=(8, 0.45 * len(tips) + 0.6))
            t = tips.iloc[::-1]
            ax.barh(t["Action"], t["Score gain"], color=BLUE, height=0.6)
            for i, v in enumerate(t["Score gain"]): ax.text(v + 0.03, i, f"+{v:.1f}", va="center", color=GREY, fontsize=9)
            ax.set_xlabel("Predicted increase in exam score (marks)", color=GREY)
            for s in ["top", "right"]: ax.spines[s].set_visible(False)
            st.pyplot(fig, width="stretch")
        else:
            st.write("This student is already doing everything the model can suggest. 🎉")
        st.info("👉 Open **🤖 AI Study Coach** in the sidebar to turn this into a personal study plan.")

elif page == "🤖 AI Study Coach":
    st.title("🤖 AI Study Coach")
    if "student" not in st.session_state:
        st.warning("First run a prediction on the **🎯 Student Risk Check** page. The coach uses those results.")
        st.stop()
    student, score = st.session_state["student"], st.session_state["score"]
    level, tips = risk_level(score), what_if(student)
    name = st.session_state.get("name", "")
    st.write(f"Coaching **{name or 'this student'}** · predicted score **{score:.1f}** · status **{level}**")

    c1, c2 = st.columns(2)
    weeks = c1.number_input("Weeks until the exam", 1, 26, 4)
    subjects = c2.text_input("Weak subjects or topics", placeholder="e.g. Maths (calculus), Physics")
    if st.button("✨ Generate my study plan", type="primary"):
        context = student_summary(student, score, level, tips)
        prompt = (f"You are EduGuard AI, a kind and practical study coach for students in India.\n{context}\n"
                  f"Student name: {name or 'the student'}. Weeks until exam: {weeks}. Weak subjects: {subjects or 'not given'}.\n"
                  "Write a personalised study plan in Markdown: (1) a 2-line encouraging summary of where they stand, "
                  "(2) top 3 priorities based on the what-if analysis, (3) a week-by-week plan until the exam, "
                  "(4) a sample daily timetable, (5) 3 quick tips for staying motivated. Keep it under 400 words, "
                  "simple English, no medical advice.")
        with st.spinner("Your coach is writing your plan..."):
            if get_api_key():
                try:
                    st.session_state["plan"] = ask_gemini(prompt)
                except Exception as e:
                    st.warning(f"AI unavailable, so showing the offline plan instead. ({str(e)[:120]})")
                    st.session_state["plan"] = offline_plan(student, score, level, tips, name, weeks, subjects)
            else:
                st.session_state["plan"] = offline_plan(student, score, level, tips, name, weeks, subjects)
        st.session_state["context"] = context

    if st.session_state.get("plan"):
        with st.container(border=True):
            st.markdown(st.session_state["plan"])
        st.download_button("⬇️ Download plan", st.session_state["plan"], file_name="study_plan.md")

        st.subheader("💬 Ask your coach")
        st.session_state.setdefault("chat", [])
        for msg in st.session_state["chat"]:
            st.chat_message(msg["role"]).markdown(msg["text"])
        question = st.chat_input("e.g. How do I stop procrastinating? How should I revise for maths?")
        if question:
            st.chat_message("user").markdown(question)
            st.session_state["chat"].append({"role": "user", "text": question})
            if get_api_key():
                history = "\n".join(f"{m['role']}: {m['text']}" for m in st.session_state["chat"][-8:])
                prompt = (f"You are EduGuard AI, a friendly study coach. Context: {st.session_state.get('context', '')}\n"
                          f"Study plan given earlier:\n{st.session_state['plan']}\n\nConversation so far:\n{history}\n"
                          "Reply to the student's last message helpfully and briefly (under 150 words).")
                try:
                    answer = ask_gemini(prompt)
                except Exception as e:
                    answer = f"Sorry, the AI is unavailable right now ({str(e)[:100]}). Please try again in a minute."
            else:
                answer = ("I'm in offline mode, so I can't chat yet. Add a free Gemini API key in the sidebar to unlock "
                          "the AI chat. Meanwhile, focus on your top priority: **" +
                          (tips.iloc[0]["Action"] if len(tips) else "keep your current routine") + "**.")
            st.chat_message("assistant").markdown(answer)
            st.session_state["chat"].append({"role": "assistant", "text": answer})

elif page == "🏫 Teacher Dashboard":
    st.title("🏫 Teacher Dashboard")
    st.write("Upload a class list (CSV) to predict every student's score and see who needs help first. "
             "Columns should match the dataset (e.g. `Hours_Studied`, `Attendance`, ...). An optional `Name` column is shown too. "
             "Missing columns are filled with typical values.")
    up = st.file_uploader("Upload class CSV", type="csv")
    use_sample = st.button("Or try with a sample class of 30 students")
    if up is not None:
        st.session_state["class_df"] = pd.read_csv(up)
    elif use_sample:
        st.session_state["class_df"] = pd.read_csv("sample_class.csv")

    if "class_df" in st.session_state:
        cls = st.session_state["class_df"].copy()
        for col in FEATURES:
            if col not in cls.columns:
                cls[col] = TYPICAL[col]
            cls[col] = cls[col].fillna(TYPICAL[col])
        cls["Predicted score"] = np.clip(model.predict(cls[FEATURES]), 0, 100).round(1)
        cls["Risk level"] = cls["Predicted score"].apply(risk_level)
        cls = cls.sort_values("Predicted score")
        counts = cls["Risk level"].value_counts()

        a, b, c, d = st.columns(4)
        a.metric("Students", len(cls))
        b.metric("🔴 High risk", int(counts.get("High risk", 0)))
        c.metric("🟠 Needs attention", int(counts.get("Needs attention", 0)))
        d.metric("🟢 On track", int(counts.get("On track", 0)))

        left, right = st.columns([1, 2])
        with left:
            fig, ax = plt.subplots(figsize=(4, 3))
            labels = ["High risk", "Needs attention", "On track"]
            vals = [int(counts.get(l, 0)) for l in labels]
            ax.bar(labels, vals, color=[RISK_COLORS[l] for l in labels], width=0.6)
            for i, v in enumerate(vals): ax.text(i, v + 0.2, str(v), ha="center", color=GREY)
            ax.set_title("Class risk overview", fontsize=11)
            ax.tick_params(axis="x", labelsize=8)
            for s in ["top", "right"]: ax.spines[s].set_visible(False)
            st.pyplot(fig, width="stretch")
        with right:
            show = [c for c in ["Name", "Predicted score", "Risk level", "Attendance", "Hours_Studied",
                                "Previous_Scores", "Motivation_Level"] if c in cls.columns]
            st.dataframe(cls[show].style.map(lambda v: f"color: {RISK_COLORS.get(v, '')}; font-weight: 600"
                                             if v in RISK_COLORS else "", subset=["Risk level"]),
                         hide_index=True, width="stretch", height=330,
                         column_config={"Predicted score": st.column_config.NumberColumn(format="%.1f")})
        st.download_button("⬇️ Download report (CSV)", cls.to_csv(index=False), file_name="class_risk_report.csv")

elif page == "📊 Model Insights":
    st.title("📊 Model Insights")
    st.write("How the machine learning model was built and how well it works.")
    a, b, c = st.columns(3)
    a.metric("Best model", info["best_model"])
    b.metric("R² score", f"{info['r2']:.3f}", help="Share of score variation explained by the model (1.0 = perfect)")
    c.metric("Mean absolute error", f"{info['mae']:.2f} marks", help="Average gap between predicted and real score")
    if info.get("all_results"):
        st.subheader("Models compared")
        st.dataframe(pd.DataFrame(info["all_results"]), hide_index=True)
    col1, col2 = st.columns(2)
    for col, img, cap in [(col1, "chart5_feature_importance.png", "Which factors matter most"),
                          (col2, "chart4_predicted_vs_actual.png", "Predicted vs actual scores")]:
        if os.path.exists(img):
            col.image(img, caption=cap, width="stretch")
    st.subheader("How it works")
    st.markdown("""
1. **Data:** 6,607 student records with 19 factors (study habits, attendance, home and school environment).
2. **Cleaning:** missing values filled with the most common value; scores capped at 100.
3. **Preparation:** text categories (Low/Medium/High...) converted to numbers with one-hot encoding.
4. **Training:** 3 algorithms trained on 80% of the students (Linear Regression, Random Forest, Gradient Boosting).
5. **Testing:** each model tested on the other 20% it had never seen; the best R² wins.
6. **What-if analysis:** the model is re-run with one habit improved at a time to rank the most helpful changes.
7. **AI coach:** the prediction and what-if results are sent to Google Gemini (a large language model), which writes the personal plan.
""")
    st.caption("Dataset: Student Performance Factors (Kaggle). Predictions are guidance, not a judgement of any student.")
