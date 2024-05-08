
import random
import pickle as pkl
import transformers
import datasets
import pickle as pkl

random.seed(0)

cache_dir = None   # Update this to set your HuggingFace cache directory

data = datasets.load_dataset('Skylion007/openwebtext', split='train[:110000]', cache_dir=cache_dir)
data = data.shuffle(seed=42)

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-xl", cache_dir=cache_dir, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

# Truncate texts to first 8 GPT2 tokens

tokens = tokenizer(data['text'])['input_ids']
prefix_tokens = [t[:8] for t in tokens]
prefixes = tokenizer.batch_decode(prefix_tokens)

with open ('setup/prefixes/eval.pkl', 'wb') as f:
    pkl.dump(prefixes[:10000], f)
with open ('setup/prefixes/train.pkl', 'wb') as f:
    pkl.dump(prefixes[10000:], f)
with open ('setup/prefixes/eval-full-texts.pkl', 'wb') as f:
    pkl.dump(data['text'][:10000], f)
