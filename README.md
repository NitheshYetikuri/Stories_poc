
# Project Name

DevStoryAI

## Prerequisites

Make sure you have Python installed (preferably Python 3.10+).

## Setup Instructions

1. **Install UV**

```bash
pip install uv
```

2. **Create a virtual environment**

```bash
UV venv
```

3. **Activate the virtual environment**
   Windows:

```bash
.venv\Scripts\activate
```

Linux / Mac:

```bash
source .venv/bin/activate
```

4. **Sync UV dependencies**

```bash
uv sync
```

5. **Run the Streamlit app**

```bash
streamlit run main.py
```

## Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_API_KEY1=your_api_key_here
GROQ_API_KEY=your_api_key_here
GROQ_API_KEY1=your_api_key_here
```

## Notes

* Make sure to replace the placeholders in the `.env` file with your actual API keys.
* Activate the virtual environment every time before running the app.

---

