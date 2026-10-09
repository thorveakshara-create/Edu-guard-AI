# EduGuard AI: Step-by-Step Build Guide

Follow these steps in order. No coding experience is needed. Every step happens in your web browser.
Total time: about **3–4 hours** spread over 2 days.

**What's in this folder**

| File | What it is |
|---|---|
| `EduGuard_Model_Training.ipynb` | The Colab notebook that trains the ML model (Part B) |
| `app.py` | The web app: Risk Check, What-if, AI Coach, Teacher Dashboard, Model Insights (Part C) |
| `requirements.txt` | The list of libraries the app needs |
| `StudentPerformanceFactors.csv` | The dataset (6,607 students) |
| `sample_class.csv` | A demo class of 30 students for the Teacher Dashboard |
| `model.pkl`, `model_info.json`, `chart*.png` | A ready-trained model and charts, so the app works straight away. **You will replace these with your own from Colab.** |

---

## Part A: Accounts (tonight, about 20 minutes)

Create these free accounts. Use the same Google account wherever possible.

1. **Google account**: you need it for Colab and Gemini.
2. **GitHub**: go to https://github.com/signup. GitHub stores your code online.
3. **Streamlit Community Cloud**: go to https://share.streamlit.io and click **"Continue with GitHub"**. It hosts your app for free.
4. **Gemini API key** (powers the AI Coach):
   - Go to https://aistudio.google.com/apikey and sign in with Google.
   - Click **Create API key** and copy the key (it starts with `AIza...`).
   - Save it somewhere private. **Never paste it into your code or share it publicly.**
5. *(Optional)* **Kaggle**: go to https://www.kaggle.com. You only need it if the notebook can't download the data automatically.

---

## Part B: Train the model in Google Colab (Day 2, about 1 hour)

1. Go to https://colab.research.google.com.
2. Click **File → Upload notebook** and choose `EduGuard_Model_Training.ipynb` from this folder.
3. *(Recommended)* Click the 📁 folder icon on the left, then the upload icon, and upload
   `StudentPerformanceFactors.csv`. The notebook then won't need to download anything.
4. Click the first code cell and press **Shift + Enter**. Keep pressing Shift + Enter to run each cell in order.
   **Read the text and the `#` comments as you go.** Your mentor may ask you about them.
5. When it asks you to, allow downloads. At the end, the last cell downloads these files to your computer:
   `model.pkl`, `model_info.json`, `requirements.txt`, `StudentPerformanceFactors.csv`, and 5 chart images.
6. **Write down** the best model's name, its R² score and its MAE. These go in your PPT.
   (Expect Linear Regression to win, with R² ≈ 0.77 and an average error of about half a mark.)

**If something goes wrong:** read the red error message, then run cells again from the top
(**Runtime → Run all**). The most common cause is skipping a cell.

### Words you should be able to explain (your mentor may ask)

- **Features / target:** the inputs we know about a student, and the exam score we predict.
- **Train/test split:** the model learns from 80% of students and is tested on the other 20%, which it has never seen.
- **One-hot encoding:** turning text like "Low/Medium/High" into 0/1 columns, because models need numbers.
- **Linear Regression:** finds the best combination of factors (weighted sum) to predict the score.
- **Random Forest:** many decision trees that vote on the answer.
- **Gradient Boosting:** trees built one after another, each fixing the last one's mistakes.
- **R²:** how much of the variation in scores the model explains (1.0 = perfect).
- **MAE:** the average number of marks the prediction is off by.
- **Permutation importance:** shuffle one factor and see how much worse the model gets. A big drop means the factor matters a lot.
- **What-if analysis:** re-run the model with one habit improved, to see how much the score would rise.
- **Where the AI comes in:** the ML model makes the *numbers* (prediction and what-if). The large language model
  (Gemini) turns those numbers into a *personalised plan* in plain language and answers the student's questions.

---

## Part C: Put the app online (Day 3, about 1 hour)

### C1. Create a GitHub repository
1. On GitHub, click **+ → New repository**.
2. Name it `EduGuard-AI`, set it to **Public**, tick **Add a README file**, then click **Create repository**.
3. Click **Add file → Upload files** and drag in these files:
   - `app.py`, `sample_class.csv`, `STEP_BY_STEP_GUIDE.md` (from this folder)
   - `model.pkl`, `model_info.json`, `requirements.txt`, `StudentPerformanceFactors.csv`,
     `chart4_predicted_vs_actual.png`, `chart5_feature_importance.png` (**your versions from Colab**)
   - `EduGuard_Model_Training.ipynb` (so your mentor can see the training code)
4. Click **Commit changes**.

### C2. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and click **Create app → Deploy a public app from GitHub**.
2. Repository: `your-username/EduGuard-AI` · Branch: `main` · Main file path: `app.py`.
3. Click **Advanced settings → Secrets** and paste this, using your own key:
   ```
   GEMINI_API_KEY = "AIza...your key..."
   ```
4. Click **Deploy**. The first start takes 2–5 minutes.
5. You get a link like `https://eduguard-ai-yourname.streamlit.app`. **This is your working model.** Put it in the PPT and concept note.

### C3. Test every feature
- **🎯 Student Risk Check:** move the sliders, click Predict, and check that the score, risk level and what-if chart appear.
- **🤖 AI Study Coach:** type weak subjects, click Generate, then ask a question in the chat box.
- **🏫 Teacher Dashboard:** click "try with a sample class" and check the counts, chart and download button.
- **📊 Model Insights:** check that the metrics and charts show.

**Troubleshooting**
- *"AI unavailable"*: check the key in Secrets (no extra spaces, inside quotes). Then go to **⋮ → Reboot app**.
  If Google has renamed its models, add the line `GEMINI_MODEL = "model-name"` to Secrets. Current model names are listed at https://ai.google.dev/gemini-api/docs/models.
- *Module not found / install error*: check that `requirements.txt` is in the repo.
- *The app shows the offline plan*: the AI isn't connected, but everything else still works. That is fine for a demo, but try to get the AI working.
- *The app falls asleep after a few days without visits*: open the link and click "Yes, get this app back up". Open it the morning you present.

---

## Part D: Screenshots for your PPT (Day 4)

Take these from your **live** app. On Windows press `Win + Shift + S`; on Mac press `Cmd + Shift + 4`.

1. Home page (title and metric cards)
2. Risk Check: the filled-in form
3. Risk Check: predicted score, risk level and "where you stand" chart
4. What-if chart
5. AI Coach: the generated study plan
6. AI Coach: a chat question and answer
7. Teacher Dashboard with the sample class
8. Model Insights page
9. From Colab: the model comparison chart and the feature importance chart

**Tip:** For screenshots 3–6, use a "struggling student" (attendance 65%, 8 study hours, low motivation).
The high-risk result and the coach's advice tell a stronger story.

---

## Run it on your own computer (optional)
If you have Python installed:
```
pip install -r requirements.txt
streamlit run app.py
```
