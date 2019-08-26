import pickle


import pandas

with open('TestBed_06_17', 'rb') as o:
    df = pickle.load(o)

print(df)