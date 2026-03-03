# Career Coach Recommender

A certificate recommendation system built for a learning platform. Given a user's profile and purchase history, it recommends the most relevant certifications using content-based filtering, co-occurrence analysis, and interaction signals.

---

## How it works

The system combines three sources of signal to rank certificates:

- **Content-based filtering** — matches the user's skills and career goal against each certificate's skills and description. Supports both exact string matching and **semantic similarity via sentence embeddings** (see AI Setup below)
- **Co-occurrence** — recommends certificates that are frequently purchased together
- **Interaction signals** — uses clicks, add-to-cart events, and purchases from `signals.csv` to boost relevant certificates

Prerequisites are enforced automatically — a certificate won't be recommended unless the user already holds the required prerequisite.

For new users with no history, the system falls back to recommending the most popular certificates (cold-start handling).

---

## Project structure

```
career-coach-recommender/
├── data/               # Raw CSV datasets
├── notebooks/
│   └── 01_eda.ipynb   # Exploratory data analysis and evaluation metrics
├── src/
│   ├── data_loader.py # Load and clean all datasets
│   ├── recommender.py # Core recommendation logic
│   └── api.py         # FastAPI endpoint
├── tests/
│   └── test_recommender.py
├── Dockerfile
├── requirements.txt
├── requirements-ai.txt  # Optional: semantic similarity dependencies
├── requirements-api.txt
└── writeup.md
```

---

## Setup

```bash
git clone <repo-url>
cd career-coach-recommender
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## AI Setup (Semantic Similarity)

To enable **semantic skill and goal matching** using sentence embeddings (`all-MiniLM-L6-v2`):

```bash
pip install -r requirements-ai.txt
```

The model downloads automatically on first run (~80 MB). Without this step, the system falls back to exact string matching — all functionality remains intact.

---

## Run the API

```bash
cd src
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

---

## API Usage

**Get recommendations for a user:**

```
GET /recommend?user_id=U0001
```

**With custom weights:**

```
GET /recommend?user_id=U0001&cbr_weight=0.7&co_weight=0.2&sig_weight=0.1
```

**Example response:**

```json
[
  {
    "cert_id": "PRINCE2F",
    "name": "PRINCE2 Foundation",
    "score": 0.6,
    "reason": "Matches your skills and goals"
  },
  {
    "cert_id": "DevOpsP",
    "name": "DevOps Practitioner",
    "score": 0.8,
    "reason": "Matches your skills and frequently bought with your certificates"
  }
]
```

---

## Run with Docker

```bash
docker build -t career-coach .
docker run -p 8000:8000 career-coach
```

---

## Run tests

```bash
cd tests
python test_recommender.py
```
