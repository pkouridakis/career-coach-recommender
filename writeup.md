# Career Coach Recommender — Writeup

## Approach

The goal was to build a certificate recommendation system that feels relevant to each user not just a generic popularity list. I combined three complementary signals to achieve this:

**Content-based filtering** matches the user's skills and career goal against each certificate's skills and short description. For example, a user whose goal is "become a project manager" will score higher on PRINCE2 certificates because the word "project" appears in the description. This handles cold-start partially — even a new user with a profile gets meaningful recommendations.

**Co-occurrence analysis** looks at purchase history across all users to find certificates that are frequently bought together. If 34 users bought DevOps Foundation and then DevOps Practitioner, the system learns that pattern and applies it to new users who own DevOps Foundation. This captures learning path logic without explicitly programming it.

**Interaction signals** (clicks, add-to-cart, purchases) from `signals.csv` are weighted and added to the final score. A user who added a certificate to their cart is showing stronger intent than one who just viewed it, so the weights reflect this:

| Event | Weight |
|---|---|
| Purchase | 0.4 |
| Add to cart | 0.3 |
| Click | 0.2 |
| Impression | 0.1 |

---

## Ranking Formula

The three scores are normalized to the same 0–1 scale before combining, so that co-occurrence counts (0–35) don't dominate content-based scores (0–3):

```
final_score = (cbr_score * 0.6) + (co_score * 0.2) + (signal_score * 0.2)
```

Content-based filtering gets the highest weight (0.6) because it directly incorporates the user's stated skills and career goal. Co-occurrence and signals each get 0.2 as supporting signals. These weights are configurable via the API.

---

## Semantic Similarity (Stretch AI Feature)

The content-based recommender now supports **semantic similarity** via `sentence-transformers` (`all-MiniLM-L6-v2`). When the library is installed, both skill matching and goal-to-description matching use cosine similarity over sentence embeddings instead of exact string overlap.

This solves a key limitation of the original approach: "sprint planning" and "planning" were treated as completely different skills. With embeddings, semantically related concepts score high even without exact string matches.

The implementation is **graceful-degradation**: if `sentence-transformers` is not installed, the system automatically falls back to the original exact-match logic. This means the API works out-of-the-box with the minimal `requirements.txt`, and the AI-enhanced version is opt-in via `requirements-ai.txt`.

```python
# skill matching: user skills joined → cert skills joined → cosine similarity
skill_score = _skill_similarity(user["skills"], cert["skills"])

# goal matching: user goal text → cert short_desc → cosine similarity
goal_score = _semantic_similarity(user["goal"], cert["short_desc"])
```

---

## Rules

Prerequisites are enforced before any recommendation is returned. A certificate is excluded from the candidate list if the user does not already own the required prerequisite. For example, ITIL Practitioner will never be recommended to a user who hasn't purchased ITIL Foundation.

---

## Known Limitations

**Co-occurrence and career goals:** The co-occurrence model recommends certificates based on purchase patterns, with no awareness of the user's career goal. A user who wants to become a Project Manager might be recommended DevOps Practitioner simply because they own DevOps Foundation — even though it doesn't match their stated goal. Normalizing scores (0.6 weight on content-based) reduces this effect, but doesn't eliminate it.

**Recall@K evaluation:** Proper offline evaluation of Recall@K requires a train/test split where each user's last purchase is held out. Since the system filters out already-purchased certificates from recommendations, a naive Recall@K calculation always returns 0. This was identified as a limitation; a future evaluation would implement a proper holdout evaluation pipeline.

**Stop words in goal matching (exact fallback):** The exact-match fallback filters out words shorter than 4 characters to avoid noise (e.g., "a", "in"). A more robust approach would use a standard stop word list. This limitation does not apply when semantic similarity is enabled.

---

## Future Improvements

- **Train/test evaluation:** Implement holdout evaluation with Recall@K and NDCG to measure recommendation quality objectively
- **Dynamic CTR weighting:** High CTR certificates like MoVF (97% CTR) could receive a dynamic boost in the ranking formula
- **Temporal signals:** Weight recent interactions more heavily than older ones using time decay
- **Collaborative filtering:** Matrix factorization (e.g. SVD) as an alternative candidate generation method
- **Exclude repeated impressions:** Currently the system can show the same recommendations to existing users on repeated API calls. Filtering already-shown impressions for all users (not just cold-start) would improve diversity
- **Fine-tuned embeddings:** The current model (`all-MiniLM-L6-v2`) is a general-purpose model. Fine-tuning on certification domain data would improve skill and goal matching further