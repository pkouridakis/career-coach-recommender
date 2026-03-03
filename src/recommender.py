import pandas as pd
from data_loader import load_catalog, load_users, load_purchases, load_signals
from itertools import combinations
from collections import Counter

# Optional: sentence-transformers for semantic similarity
# Falls back to exact matching if not installed
try:
    from sentence_transformers import SentenceTransformer, util
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    SEMANTIC_ENABLED = True 
except ImportError:
    SEMANTIC_ENABLED = False

# Load all datasets
data_path = "../data/"

users_list = load_users()
cert_list = load_catalog()
purchases_list = load_purchases()
signals_list = load_signals()


def find_purchases(user_id, purchase_df):
    if user_id in purchase_df["user_id"].values:
        return purchase_df[purchase_df["user_id"] == user_id]["cert_id"].tolist()
    return []


def check_prerequisites(cert_id, catalog_df, user_purchases):
    if cert_id in catalog_df["cert_id"].values:
        cert = catalog_df[catalog_df["cert_id"] == cert_id].iloc[0]
        if cert["prerequisites"] == []:
            return True
        prerequisite = cert["prerequisites"][0]
        return prerequisite in user_purchases
    print("Certificate doesnt exists")


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using sentence embeddings.
    Returns a float in [0, 1]. Falls back to 0.0 if model is unavailable.
    """
    if not SEMANTIC_ENABLED or not text_a or not text_b:
        return 0.0
    emb_a = _model.encode(text_a, convert_to_tensor=True)
    emb_b = _model.encode(text_b, convert_to_tensor=True)
    score = util.cos_sim(emb_a, emb_b).item()
    return max(0.0, score)  # clamp negatives to 0


def _skill_similarity(user_skills: list, cert_skills: list) -> float:
    """
    Semantic similarity between user's skills (joined) and cert's skills (joined).
    Falls back to Jaccard exact matching when sentence-transformers is unavailable.
    """
    if SEMANTIC_ENABLED:
        user_text = ", ".join(user_skills)
        cert_text = ", ".join(cert_skills)
        return _semantic_similarity(user_text, cert_text)
    # Exact-match fallback: Jaccard-like overlap score
    common = set(user_skills).intersection(set(cert_skills))
    return len(common)


def content_based_recommend(user_id, users_df, catalog_df):
    if user_id not in users_df["user_id"].values:
        return []

    user = users_df[users_df["user_id"] == user_id].iloc[0]
    user_purchases = find_purchases(user_id, purchases_list)

    scores = {}
    for _, cert in catalog_df.iterrows():
        cert_id = cert["cert_id"]

        if cert_id in user_purchases:
            continue
        if not check_prerequisites(cert_id, catalog_df, user_purchases):
            continue
    
        skill_score = _skill_similarity(user["skills"], cert["skills"])

        if SEMANTIC_ENABLED:
            goal_score = _semantic_similarity(user["goal"], cert["short_desc"])
        else:
            
            user_goal_words = [w for w in user["goal"].split() if len(w) > 3]
            goal_score = sum(1 for w in user_goal_words if w in cert["short_desc"])

        scores[cert_id] = skill_score + goal_score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]


def co_occurrence_recommend(user_id, purchases_df):
    user_certs = purchases_df.groupby("user_id")["cert_id"].apply(list)

    pair_counts = Counter()
    for cert_list_u in user_certs:
        for pair in combinations(sorted(cert_list_u), 2):
            if pair[0] != pair[1]:
                pair_counts[pair] += 1

    user_certs_list = find_purchases(user_id, purchases_df)
    recommendations = {}

    for pair, count in pair_counts.items():
        if pair[0] in user_certs_list and pair[1] not in user_certs_list:
            recommendations[pair[1]] = count
        if pair[1] in user_certs_list and pair[0] not in user_certs_list:
            recommendations[pair[0]] = count

    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:5]


def signal_score(user_id, cert_id, signal_df):
    weights = {
        "purchase": 0.4,
        "add_to_cart": 0.3,
        "click": 0.2,
        "impression": 0.1,
    }
    if user_id not in signal_df["user_id"].values:
        return 0
    user_signals = signal_df[
        (signal_df["user_id"] == user_id) & (signal_df["cert_id"] == cert_id)
    ]
    return sum(weights[row["event"]] for _, row in user_signals.iterrows())


def cold_start_recommend(catalog_df, purchases_df, signals_df, user_id):
    already_seen = signals_df[signals_df["user_id"] == user_id]["cert_id"].tolist()
    popular = purchases_df["cert_id"].value_counts()

    result = []
    for cert_id, count in popular.items():
        if cert_id not in already_seen:
            cert = catalog_df[catalog_df["cert_id"] == cert_id].iloc[0]
            result.append(
                {
                    "cert_id": cert_id,
                    "name": cert["name"],
                    "score": count,
                    "reason": "Trending certificate among learners",
                }
            )
        if len(result) == 5:
            break
    return result[:5]


def recommend(
    user_id,
    users_df,
    catalog_df,
    purchases_df,
    signals_df,
    cbr_weight=0.6,
    co_weight=0.2,
    sig_weight=0.2,
):
    cbr_scores = dict(content_based_recommend(user_id, users_df, catalog_df))
    co_scores = dict(co_occurrence_recommend(user_id, purchases_df))

    if not cbr_scores and not co_scores:
        return cold_start_recommend(catalog_df, purchases_df, signals_df, user_id)

    max_cbr = max(cbr_scores.values()) if cbr_scores and max(cbr_scores.values()) > 0 else 1
    max_co = max(co_scores.values()) if co_scores and max(co_scores.values()) > 0 else 1

    all_certs = set(cbr_scores.keys()) | set(co_scores.keys())

    final_scores = {}
    for cert_id in all_certs:
        cbr = cbr_scores.get(cert_id, 0) / max_cbr
        co = co_scores.get(cert_id, 0) / max_co
        sig = signal_score(user_id, cert_id, signals_df)
        final_scores[cert_id] = (cbr * cbr_weight) + (co * co_weight) + (sig * sig_weight)

    top5 = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    result = []
    for cert_id, score in top5:
        cert = catalog_df[catalog_df["cert_id"] == cert_id].iloc[0]

        if co_scores.get(cert_id, 0) > 0 and cbr_scores.get(cert_id, 0) > 0:
            reason = "Matches your skills and frequently bought with your certificates"
        elif co_scores.get(cert_id, 0) > 0:
            reason = "Frequently bought with your certificates"
        else:
            reason = "Matches your skills and goals"

        result.append(
            {
                "cert_id": cert_id,
                "name": cert["name"],
                "score": round(score, 3),
                "reason": reason,
            }
        )

    return result
