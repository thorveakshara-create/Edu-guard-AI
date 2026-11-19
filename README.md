EduGuard AI

Spot struggling students early. Give every student a personal study coach.

Live Demo | GitHub Repository

EduGuard AI Screenshot

The Problem

Many students fall behind quietly. By the time exam results come out, it is too late to help.

The Solution

EduGuard AI uses machine learning to predict a student's exam score weeks before the exam. It flags students who are at risk, explains why, and uses an AI coach to turn those insights into a personal study plan.

Features
Student Risk Check: Enter a student's details to get a predicted exam score and a risk flag, along with an explanation of the factors behind the prediction.
AI Study Coach: Powered by the Google Gemini API, it converts the risk insights into a personalized study plan.
Teacher Dashboard: Lets teachers monitor performance and identify at-risk students at a glance.
Tech Stack
Area	Tools
Language	Python
Machine Learning	Model trained and tested in Google Colab
Web App	Streamlit
AI Coach	Google Gemini API
Deployment	Streamlit Community Cloud
Version Control	Git and GitHub
How It Works
A machine learning model is trained on student data to predict exam scores.
The app takes a student's inputs and returns a predicted score and risk level.
The model's prediction is explained so users understand which factors matter.
The Gemini-powered AI Study Coach turns this into a personalized study plan.
Teachers can view results across students in the dashboard.
Run Locally
bash
# 1. Clone the repository
git clone https://github.com/thorveakshara-create/Edu-guard-AI.git
cd Edu-guard-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
export GEMINI_API_KEY="your_api_key_here"

# 4. Run the app
streamlit run app.py

Note: replace app.py with the name of your main Streamlit file if it is different. On Windows, set the key with set GEMINI_API_KEY=your_api_key_here.

If you deploy on Streamlit Cloud, add the key under App settings > Secrets instead of putting it in the code.

Project Status

Built as an internship project. The app is currently a working prototype.

Author

Akshara Thorve GitHub: thorveakshara-create
