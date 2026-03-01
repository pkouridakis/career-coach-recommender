from fastapi import FastAPI
from recommender import recommend, users_list, cert_list, purchases_list
from data_loader import load_signals
import csv
from datetime import datetime


def log_impression(user_id, cert_id):
    with open("../data/signals.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.today().date(),user_id,cert_id,"impression"])

app = FastAPI()

@app.get("/recommend")
def get_recommendations(user_id: str, cbr_weight: float = 0.6, co_weight: float = 0.2, sig_weight: float = 0.2):
    signals_fresh = load_signals()
    results = recommend(user_id, users_list, cert_list, purchases_list, signals_fresh, cbr_weight, co_weight, sig_weight)
    for cert in results:
        log_impression(user_id, cert["cert_id"])
    return results