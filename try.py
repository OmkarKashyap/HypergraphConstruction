import pickle

with open('pickled_datasets/train.pkl', 'rb') as f:
    file = pickle.load(f)

print(file[:10])