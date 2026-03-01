import pandas as pd
from data_loader import load_catalog, load_users, load_purchases, load_signals
from itertools import combinations
from collections import Counter

# Load all datasets
data_path   = "../data/" #./data/

users_list = load_users()
cert_list = load_catalog()
purchases_list = load_purchases()
signals_list = load_signals()

def find_purchases(user_id, purchase_df):
    if user_id in purchase_df["user_id"].values:
        user_purchases = purchase_df[purchase_df["user_id"] == user_id]["cert_id"].tolist()

        return user_purchases
    return []

def check_prerequisites(cert_id, catalog_df, user_purchases):
    if cert_id in catalog_df["cert_id"].values:
        cert = catalog_df[catalog_df["cert_id"] == cert_id].iloc[0]
        if cert["prerequisites"] == []:
            return True
        else:
            prerequisite = cert["prerequisites"][0]
            if prerequisite in user_purchases:
                return True
            else:
                return False
    else:
        print("Certificate doesnt exists")

def content_based_recommend(user_id, users_df, catalog_df):
    if user_id in users_df["user_id"].values: # check if user exists
        user = users_df[users_df["user_id"] == user_id].iloc[0]
        scores = {}
        scores[user_id] = {}
        for cert_index, cert in catalog_df.iterrows(): # for each cert calculate score
            cert_id = cert["cert_id"]
            cert_skills = set(cert["skills"])
            common = set(user["skills"]).intersection(cert_skills) # retrieve user's skills and finds the commons
            
            user_goal= [word for word in user["goal"].split() if len(word) > 3] # to avoid words like "a"
            cert_desc = cert["short_desc"]
            goal_matches = sum(1 for word in user_goal if word in cert_desc)

            scores[user_id][cert_id] = len(common) + goal_matches
        
        user_purchases = find_purchases(user_id, purchases_list)
        filtered_scores = {}

        for cert_id, score in scores[user_id].items(): 
            #if cert_id not in user_purchases:
            if check_prerequisites(cert_id, catalog_df, user_purchases):
                filtered_scores[cert_id] = score
    else:
        return []

    return sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:5]

#content_based_recommend('U0003', users_list, cert_list)
#find_purchases('U0001', purchases_list)
#check_prerequisites('ITILF', cert_list, find_purchases('U0001', purchases_list))

def co_occurrence_recommend(user_id, purchases_df):
    user_certs = purchases_df.groupby('user_id')['cert_id'].apply(list)
    
    pair_counts = Counter()
    for cert_list in user_certs:
        for pair in combinations(sorted(cert_list), 2):
            if pair[0] != pair[1]:
                pair_counts[pair] += 1
    
    user_certs_list = find_purchases(user_id, purchases_df)
    recommendations = {}

    for pair, count in pair_counts.items():
        if pair[0] in user_certs_list:
            if pair[1] not in user_certs_list:
                recommendations[pair[1]] = count
        if pair[1] in user_certs_list:
            if pair[0] not in user_certs_list:
                recommendations[pair[0]] = count
            
    return sorted(recommendations.items(), key=lambda x: x[1],reverse=True)[:5]
 
def signal_score(user_id, cert_id, signal_df):
    weights = {
        'purchase': 0.4,
        'add_to_cart': 0.3,
        'click': 0.2,
        'impression': 0.1
    }
    if user_id in signal_df["user_id"].values:
        user_signals = signal_df[
            (signal_df["user_id"] == user_id) & 
            (signal_df["cert_id"] == cert_id)
        ]
        score = 0
        for _, row in user_signals.iterrows():
            score += weights[row["event"]]
        return score
    return 0

def cold_start_recommend(catalog_df, purchases_df, signals_df, user_id):
    alread_seen = signals_df[signals_df["user_id"] == user_id]["cert_id"].tolist()
    
    popular = purchases_df["cert_id"].value_counts()
    
    result = []
    for cert_id, count in popular.items():
        if cert_id not in alread_seen:
            cert = catalog_df[catalog_df["cert_id"] == cert_id].iloc[0]
            result.append({
                "cert_id": cert_id,
                "name": cert["name"],
                "score": count,
                "reason": "Trending certificate among learners"
            })
        if len(result) == 5:
            break
    return result[:5]

def recommend(user_id, users_df, catalog_df, purchases_df, signals_df, cbr_weight = 0.6, co_weight = 0.2, sig_weight = 0.2):
    cbr_scores = dict(content_based_recommend(user_id, users_df, catalog_df))
    co_scores  = dict(co_occurrence_recommend(user_id, purchases_df))
    if not cbr_scores and not co_scores :
        return cold_start_recommend(catalog_df, purchases_df, signals_df, user_id)

    # normalize to 0-1 so scores are on the same scale, because co has large numbers like 34 and cbr smalls like 1-3
    max_cbr = max(cbr_scores.values()) if cbr_scores and max(cbr_scores.values()) > 0 else 1
    max_co  = max(co_scores.values())  if co_scores  and max(co_scores.values())  > 0 else 1

    all_certs = set(cbr_scores.keys()) | set(co_scores.keys())

    final_scores = {}
    for cert_id in all_certs:
        cbr = cbr_scores.get(cert_id, 0) / max_cbr
        co  = co_scores.get(cert_id, 0)  / max_co
        sig = signal_score(user_id, cert_id, signals_df)
        
        final_scores[cert_id] = (cbr * cbr_weight) + (co * co_weight) + (sig * sig_weight)

    top5 = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    # build final output with name + reason
    result = []
    for cert_id, score in top5:
        cert = catalog_df[catalog_df["cert_id"] == cert_id].iloc[0]
        
        # decide reason
        if co_scores.get(cert_id, 0) > 0 and cbr_scores.get(cert_id, 0) > 0:
            reason = f"Matches your skills and frequently bought with your certificates"
        elif co_scores.get(cert_id, 0) > 0:
            reason = f"Frequently bought with your certificates"
        else:
            reason = f"Matches your skills and goals"

        result.append({
            "cert_id": cert_id,
            "name":    cert["name"],
            "score":   round(score, 3),
            "reason":  reason
        })

    return result

#print("U0003:", recommend('U0003', users_list, cert_list, purchases_list, signals_list))