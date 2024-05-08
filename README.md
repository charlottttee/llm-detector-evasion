# Introduction

This repository contains code related to the paper Language Model Detectors are Easily Optimized Against. It is built on top of the Direct Preference Optimization codebase, which can be accessed at https://github.com/eric-mitchell/direct-preference-optimization.

# Basic Experimental Setup

This section will walk you through tuning Llama2-7b to evade OpenAI's RoBERTa-large detector.

## Getting the Prefixes

You will need training and eval prefixes. You may use your own if you want to train for a specific topic, but we used generic internet texts from https://huggingface.co/datasets/Skylion007/openwebtext. You will need to truncate your prefixes to a length that is short compared to the sequence length. In this setup, we are using a sequence length of `128` Llama2 tokens and a prefix length of `8` GPT2 tokens (we used the same prefixes when training all non-chat models for this task). You can find `.pkl` files containing the training and eval prefixes, as well as the full eval texts for comparison to model-generated texts, in `setup/prefixes`. You can also create these files yourself using `setup/get_prefixes.py`.

## Generating the Training Set

### Getting the Generations

Next, you will need to generate a training set using `setup/get_samples.py`. The command line arguments (all required) are, in order:
1. get2: A boolean value indicating if we want one of two generations per prefix. When creating training datasets, set this to `True`.
2. out: The full path to the file where you would like to store output generations. Note that the code saves to this file periodically throughout generation, not just when it finishes executing.
3. archive: The model archive used for tuning. In this setup, we tune the base Llama2-7b model, so you should set this to `None`.
4. data_path: The full path to a `.pkl` file containing the prefixes. Set this to `setup/prefixes/train.pkl`.

In the file, you can also adjust the batch size (defaults to `125`), start index (defaults to `0`), end index (defaults to the length of the training list), the sequence length (defaults to `128` tokens), the temperature and top-p (default to `1.0`), and the model name (defaults to `"meta-llama/Llama-2-7b-hf"`).

IMPORTANT: YOU NEED TO ADD YOUR LLAMA2 ACCESS TOKEN TO LINE 15.

### Labelling the Generations

To label the generations, run `detectors/roberta_lg.py`. The command line arguments (all required), in order, are:
1. out: The path to the file to which to save the probabilities. `-probs.json` will be appended to the name you specify here.
2. in_data: The path to the generations file created in the previous step. `.json` will be appended to the name you specify here, so do not include this.
3. get2: Similar to get2 above. Set it to True for this step.

In the file, you can change the batch size (defaults to `250`), start index (defaults to `0`), end index (defaults to the length of the generations list).

### Creating the DPO Dataset

Now that we have the generations and human probabilities, you will need to run `setup/create_dpo_dataset.py` to generate a dataset of preference pairs. The command line arguments (all required), in order, are:
1. gens_path: The path to the generations file. `.json` will be appended to the name you specify here, so do not include this.
2. probs_path: The path to the probabilities file generated in the previous step. Omit the `-probs.json` ending. The dataset will be saved to the same path, but with the `-dataset.json` ending.

This file selects the "preferred" element to be the element with a higher probability. This is consistent with the outputs of the RoBERTa-lg detector, but if you are using a different detector, be careful, as some detectors output a higher score for a higher chance of being AI-generated. In this case, set the `reverse` variable in the file to `True`. This file also removes any pairs that have the same probability, so if you use a highly discretized detector, you may see a significant reduction in data from generations to the final dataset.

## Running DPO

Finally, train the model using `train.py`. An example configuration is:

`python3 train.py model=llama7b datasets=[owt] loss=dpo loss.beta=0.5 lr=0.000005 exp_name=[EXPERIMENT NAME] gradient_accumulation_steps=2 batch_size=8 eval_batch_size=16 n_eval_model_samples=64 eval_every=10000 n_epochs=1 model.policy_dtype=bfloat16 +dataset_kwargs.owt.data_path=[FULL PATH TO DPO DATASET] max_prompt_length=16 max_length=128`

A few notes on this command:
1. `datasets` should be set to `[owt]` to match the formatting you used above when creating the dataset.
2. The beta value here is `0.5`. Depending on your preferences on the constraint to the original model, you may consider making this slightly higher. However, to maintain a reasonably low perplexity increase, we recommend not decreating it much.
3. This runs a single epoch over all `100k` training examples. If you want to train for fewer than `100k` examples, switch `n_epochs` to `n_examples`, and set the value to the desired example count.
4. If you wish to train from an existing archive, you can add `model.archive=[PATH TO ARCHIVE]` and/or `model.reference=[PATH TO ARCHIVE]`. The former initializes the policy model to the archive while the latter updates the reference model used by the DPO algorithm.

Note: In WandB, there is logging functionality that checks average detector score at each evaluation. This specifically logs info about the performance of the RoBERTa-lg detector. Please keep this in mind if your training set was created with a different detector.

IMPORTANT: YOU NEED TO ADD YOUR LLAMA2 ACCESS TOKEN TO LINE 17 OF `train.py` AND LINE 47 OF `trainers.py`.
