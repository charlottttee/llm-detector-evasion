import json
import sys

gens_path = sys.argv[1] #except .json
probs_path = sys.argv[2] #except -probs.json
out_path = probs_path
reverse = False

with open(gens_path + '.json', 'rb') as f:
    data = json.load(f)
print('Loaded generations from ' + gens_path + '.json')

with open(probs_path + '-probs.json', 'rb') as f:
    probs = json.load(f)
print('Loaded probs from ' + probs_path + '-probs.json')

prefixes = []
chosen = []
rejected = []

print(len(probs['probs 1']))

for i in range(len(probs['probs 1'])):
    prefix = data['prefixes'][i]
    less = (probs['probs 1'][i] < probs['probs 2'][i])
    more = (probs['probs 1'][i] > probs['probs 2'][i])
    if (more and not reverse) or (less and reverse):
        chosen.append(data['generations 1'][i])
        rejected.append(data['generations 2'][i])
    elif (less and not reverse) or (more and reverse):
        chosen.append(data['generations 2'][i])
        rejected.append(data['generations 1'][i])
    else:
        continue
    prefixes.append(prefix)

with open(out_path + '-dataset.json', 'w') as f:
    json.dump({'prefixes': prefixes, 'chosen': chosen, 'rejected': rejected}, f, indent=2)
print('Saved to ' + out_path + '-dataset.json')
