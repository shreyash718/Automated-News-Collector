# 🚀 Automated News Collector (Streamlit + LangChain)

A modern AI-powered news aggregator built with **Streamlit**, **LangChain**, **RSS**, and optional **LLM summarization** (ChatGPT / Claude / Gemini).  
Designed to be simple enough for beginners, yet powerful enough for advanced users.

---

## ✨ Features

- 🔍 Search any topic  
- 🔥 Trending topics (Google Trends → NewsAPI → RSS fallback)  
- 🤖 AI Summaries using:
  - ChatGPT (OpenAI)
  - Claude (Anthropic)
  - Gemini (Google)
- 📰 RSS-only mode (works without API keys!)  
- 📄 Combined summary + key takeaways  
- 📑 Pagination (Prev / Next, fully stable)  
- 💡 Clean HTML-free article extraction  
- 🧠 Local extractive summarizer (offline fallback)  
- 🛠️ Safe Streamlit `session_state` (no weird reruns)

---

# 📦 Installation (Beginner Friendly)

## 1️⃣ Install Python (3.10+ recommended)
Download from:  
https://www.python.org/downloads/

Check version:
```bash
python --version
```

---

## 2️⃣ Create a virtual environment
```bash
python -m venv myvenv
```

Activate it:

### Windows:
```bash
myvenv\Scripts\activate
```

### Mac/Linux:
```bash
source myvenv/bin/activate
```

---

## 3️⃣ Install dependencies
```bash
pip install streamlit requests python-dotenv feedparser pytrends google-generativeai langchain langchain-openai langchain-anthropic langchain-google-vertexai
```

(If some libraries fail, the app still works using fallbacks.)

---

# 🔑 API Keys Setup (Optional)

Create a file named **`.env`** in your project folder:

```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
NEWSAPI_KEY=your_key_here
```

> **Note:**  
> You do NOT need any API key if you enable **RSS-only mode** inside the app.

**NEVER upload `.env` to GitHub.**

---

# ▶️ Running the App

Inside your virtual environment:

```bash
streamlit run app.py
```

The app opens automatically at:

```
http://localhost:8501
```

---

# 🕹️ How to Use

### 🔍 Searching
1. Enter any topic (e.g., "AI", "Elections", "Startups")  
2. Choose:
   - Model: ChatGPT / Claude / Gemini  
   - Articles per page  
   - RSS-Only Mode  
3. Click **Search**

---

### 🔥 Trending Topics
Click any trending chip →  
The app auto-fetches news + summaries.

---

### 📑 Pagination
Use **Prev** and **Next** buttons.  
Smooth, stable, no experimental reruns.

---

### 🧠 Summaries
Click **Combined Summary & Takeaways** to get:
- 3-sentence summary  
- 3 key takeaways  

If no API key → automatic **local summary**.

---

# 📂 Project Structure

```
project/
│── app.py
│── .env               # API keys (ignored by Git)
│── .gitignore
│── requirements.txt   # optional
```

---

# 🤝 Contributing

Pull requests are welcome!  
Suggestions for UI improvements, new sources, or better summarization are appreciated.

---

# 🛡️ Important Security Note

Do **NOT** commit your API keys.  
GitHub will block your push instantly using secret-scanning.

---

# ⭐ If you like this project…

Give it a star ⭐ on GitHub — it motivates further improvements!

---

# ❤️ Acknowledgements

Built using:
- Streamlit  
- LangChain  
- OpenAI / Anthropic / Gemini SDKs  
- NewsAPI  
- Google Trends  
- Feedparser  

---

Enjoy using the **Automated News Collector** 🚀  
