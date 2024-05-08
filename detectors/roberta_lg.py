import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tqdm import tqdm
import sys

cache_dir = None  # Set Huggingface cache directory if desired

out = sys.argv[1]
in_data = sys.argv[2]
get2 = eval(sys.argv[3])
assert isinstance(get2, bool) # get_2 must be a boolean value
chat = False
first = not os.path.isfile(out + '-probs.json')    # When get2 is off, this file will only overwrite the `roberta_lg` probabilities list at the destination file
start_idx = 0
end_idx = None  # If not set, this defaults to the length of the generations list
batch_size = 250

with open(in_data + '.json', 'rb') as f:
    data_unprocessed = json.load(f)

print('--------------Loaded Data--------------')

detector_mod = AutoModelForSequenceClassification.from_pretrained('roberta-large-openai-detector', cache_dir=cache_dir).cuda()
detector_tok = AutoTokenizer.from_pretrained('roberta-large-openai-detector', cache_dir=cache_dir)
detector = pipeline('text-classification', model=detector_mod, tokenizer=detector_tok, device='cuda:0', batch_size=64)

print('--------------Loaded Detector--------------')

tag = 'generations'
if get2 or tag not in data_unprocessed.keys():
    tag += ' 1'

if chat:
    data = {}
    data[tag] = [item.split('[/INST]')[1].lstrip() for item in data_unprocessed[tag]]
    if get2:
        data['generations 2'] = [item.split('[/INST]')[1].lstrip() for item in data_unprocessed['generations 2']]
else:
    data = data_unprocessed
  
if end_idx == None:
  end_idk = len(data[tag])

probs1 = []
probs2 = []
for i in range(start_idx, end_idx, batch_size):
    print(i)
    out1 = detector(data[tag][i:min(i+batch_size, len(data[tag]))], truncation=True)
    probs1.extend([o['score'] if o['label'] == 'LABEL_1' else 1 - o['score'] for o in out1])
    if get2:
        out2 = detector(data['generations 2'][i:min(i+batch_size, len(data['generations 2']))], truncation=True)
        probs2.extend([o['score'] if o['label'] == 'LABEL_1' else 1 - o['score'] for o in out2])
    dump = {'probs': probs1}
    if get2:        
        dump = {'probs 1': probs1, 'probs 2': probs2}
        with open(out + '-probs.json', 'w') as f:
            json.dump(dump, f)
    else:
        if first:
            out_data = {}
        else:
            with open(out + '-probs.json', 'rb') as f:
                out_data = json.load(f)
        out_data['roberta_lg'] = probs1
        with open(out + '-probs.json', 'w') as f:
            print('saving to ' + out + '-probs.json')
            json.dump(out_data, f)

print('--------------Finished Dump--------------')
