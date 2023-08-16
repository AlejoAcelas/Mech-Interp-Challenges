import json
import os
import numpy as np
from sklearn.metrics import accuracy_score

reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'

print('Reading prediction')
prediction = np.genfromtxt(os.path.join(prediction_dir, 'prediction'))
truth = np.genfromtxt(os.path.join(reference_dir, 'testing_label'))
with open(os.path.join(prediction_dir, 'metadata.json')) as f:
    duration = json.load(f).get('duration', -1)

print('Checking Accuracy')
accuracy = accuracy_score(truth, prediction)
print('Scores:')
scores = {
    'accuracy': accuracy,
    'duration': duration
}
print(scores)

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
