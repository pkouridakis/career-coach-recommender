import unittest
import sys
sys.path.append('../src')

from recommender import recommend, users_list, cert_list, purchases_list, signals_list, check_prerequisites, find_purchases
from data_loader import load_catalog, load_users, load_purchases, load_signals

catalog   = load_catalog()
users     = load_users()
purchases = load_purchases()
signals   = load_signals()


assert check_prerequisites('ITILF', cert_list, []) == True  # no prereqs
assert check_prerequisites('ITILP', cert_list, []) == False  # missing prereq
assert check_prerequisites('ITILP', cert_list, ['ITILF']) == True  # has prereq
assert find_purchases('U0001', purchases_list) != [] # U0001 has purchases
assert find_purchases('U9999', purchases_list) == [] # User doesn't exists → empty list

print("All tests passed! ✅")