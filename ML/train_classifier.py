import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open("./new_data.pickle", 'rb'))
# Check sample lengths
lengths = [len(sample) for sample in data_dict['data']]
print("Unique lengths of samples:", set(lengths))  # Useful debug

# Filter samples of expected length (e.g., 42)
expected_length = 42
filtered_data = []
filtered_labels = []

for sample, label in zip(data_dict['data'], data_dict['labels']):
    if len(sample) == expected_length:
        filtered_data.append(sample)
        filtered_labels.append(label)

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# model = DecisionTreeClassifier()
model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model_2.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
