import torch
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
import datasets
import random
import json
import tqdm
import argparse
import pickle as pkl
import sentencepiece as spm
import sys

random.seed(0)
cache_dir = None    # Use this line to set your Huggingface cache if desired
token = None        # You will need to add your own llama2 token for this to work, or switch the model below

batch_size = 125

get2 = eval(sys.argv[1])
assert isinstance(get2, bool) # get_2 must be a boolean value
out = sys.argv[2]
archive = sys.argv[3]
if archive == "None":
    archive = None
data_path = sys.argv[4]

start_idx = 0
end_idx = None      # Unless set to someting, this will default to the length of your prefix data, the end index is not inclusive
temperature = 1.0
top_p = 1.0
ml = 128            # This value is the minumim and maximum sequence length for generations
model_name = "meta-llama/Llama-2-7b-hf"

print("Loading model.")
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, token=token, cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map='auto')

print("Loading archive.")
if archive is not None:
    state_dict = torch.load(archive, map_location='cpu') 
    step, metrics = state_dict['step_idx'], state_dict['metrics']
    print(f'loading reference pre-trained weights at step {step} from {archive} with metrics {json.dumps(metrics, indent=2)}')
    model.load_state_dict(state_dict['state'])

model.to('cuda:0')

print("Loading transformer.")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=token, cache_dir=cache_dir, padding_side='left')
if tokenizer.pad_token is None:
    print('setting pad token to [PAD]')
    tokenizer.pad_token = '[PAD]'

print("Loading data.")
with open (data_path, 'rb') as f:
    prefixes = pkl.load(f)
  
if end_idx == None:
    end_idx = len(prefixes)

print('Overall Start:', start_idx)
print('Overall End:', end_idx)

def get_samples(prompts, seed1 = 0, seed2 = 1):
    inputs = tokenizer(prompts, padding=True, return_tensors='pt').to('cuda:0')

    torch.manual_seed(seed1)
    outputs1 = model.generate(
        **inputs,
        do_sample=True,
        max_length=ml,
        min_length=ml,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature,
        top_p=top_p
    )

    torch.manual_seed(seed2)
    if get2:
        outputs2 = model.generate(
            **inputs,
            do_sample=True,
            max_length=ml,
            min_length=ml,
            pad_token_id=tokenizer.pad_token_id,
            temperature=temperature,
            top_p=top_p
        )
        return tokenizer.batch_decode(outputs1), tokenizer.batch_decode(outputs2)
    
    return tokenizer.batch_decode(outputs1), None

gens1 = []
if get2:
    gens2 = []

batch_num = (end_idx - start_idx) // batch_size

print("Getting generations.")

for j in tqdm.tqdm(list(range(start_idx, end_idx, batch_size))):
    batch = prefixes[j:j+batch_size]

    out1, out2 = get_samples(batch, seed1=2*j, seed2=2*j+1)
    gens1.extend(out1)
    if get2:
        gens2.extend(out2)
        if j % (500) == 0:
            print('Saving Out Data')
            with open(out, 'w') as f:
                json.dump({'prefixes': prefixes, 'generations 1': gens1, 'generations 2': gens2}, f, indent=2)
    elif j % (500) == 0:
        print('Saving Out Data')
        with open(out, 'w') as f:
            json.dump({'prefixes': prefixes, 'generations': gens1}, f, indent=2)

gens1 = [g.replace('<unk>', '').replace('<s> ','') for g in gens1]
if get2:
    gens2 = [g.replace('<unk>', '').replace('<s> ','') for g in gens2]

if get2:
    with open(out, 'w') as f:
        json.dump({'prefixes': prefixes, 'generations 1': gens1, 'generations 2': gens2}, f, indent=2)
else:
    with open(out, 'w') as f:
        json.dump({'prefixes': prefixes, 'generations': gens1}, f, indent=2)
