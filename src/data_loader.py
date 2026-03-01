import pandas as pd

# Load all datasets
data_path   = "../data/" #./data/

def load_catalog():
    df = pd.read_csv(data_path + "catalog.csv", header=0)
    df["skills"] = df["skills"].str.split('|')
    df["languages"] = df["languages"].str.split('|')
    df["prerequisites"] = df["prerequisites"].fillna('').str.split('|').apply(lambda x: [s.strip() for s in x if s != ''])

    return df

def load_users():
    df = pd.read_csv(data_path + "users.csv", header=0)
    df["skills"] = df["skills"].str.split('|')

    return df

def load_purchases():
    df = pd.read_csv(data_path + "purchases.csv", header=0)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    return df

def load_signals():
    df = pd.read_csv(data_path + "signals.csv", header=0)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df