import torch as t
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # pip install transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import time
import numpy as np
import sklearn
import pickle
import re  # regular expressions, useful for decoding the output


def create_dataset(i_start=0, i_end=50, operation=t.add):
    """(1 pt) Create a dataset of pairs of numbers to calculate an operation on.
    DO NOT USE A FOR LOOP. Use pytorch functions, possibilities include meshgrid, stack, reshape, repeat, tile.
    (Note you'll have to use for loops on string stuff in other functions)

    The dataset should be a tuple of two tensors, X and y, where X is a Nx2 tensor of numbers to add,
    and y is a N tensor of the correct answers.
    E.g., if i_start=0, i_end=2, then X should be tensor([[0,0,1,1],[0,1,0,1]]).T and y should be tensor([0,1,1,2]).
    I recommend doing all pairs of sums involving 0-49, but you may modify this.
    """
    # TODO
    return X, y

def load_LLM(default="EleutherAI/gpt-neo-2.7B", device='cpu'):
    """(1 pt) Load a pretrained LLM and put on device. Default choice is a large-ish GPT-neo-2.7B model on Huggingface.
    Could also consider the "open GPT" from facebook: "facebook/opt-2.7b", or others
    here: https://huggingface.co/models?pipeline_tag=text-generation
    Explicitly load model and tokenizer, don't use the huggingface "pipeline" which hides details of the model
    (and it also has no batch processing, which we need here)
    """
    # TODO
    return model, tokenizer

def encode_problems(X, strategy='baseline'):
    """(1 pts) Encode the problems as strings. For example, if X is [[0,0,1,1],[0,1,0,1]],
    then the baseline output should be ["0+0=", "0+1=", "1+0=", "1+1="]"""
    output_strings = []
    for xi in X:
        if strategy == 'baseline':
            # TODO: encode_string =
        else:
            # TODO: encode_string =
        output_strings.append(encode_string)
    return output_strings

def generate_text(model, tokenizer, prompts, verbose=True, device='cpu'):
    """(3 pts) Complete the prompt using the LLM.
    1. Tokenize the prompts: https://huggingface.co/docs/transformers/preprocessing
        Put data and model on device to speed up computations
        (Note that in real life, you'd use a dataloader to do this efficiently in the background during training.)

    2. Generate text using the model.
        Turn off gradient tracking to save memory.
        Determine the sampling hyper-parameters.
        You may need to do it in batches, depending on memory constraints

    3. Use the tokenizer to decode the output.
    You will need to optionally print out the tokenization of the input and output strings for use in the write-up.
    """
    t0 = time.time()
    # TODO: tokenize
    # TODO: generate text, turn off gradient tracking
    # TODO: decode output, output_strings = ...

    if verbose:
        # TODO: print example tokenization for write-up
    print("Time to generate text: ", time.time() - t0)  # It took 4 minutes to do 25000 prompts on an NVIDIA 1080Ti.
    return output_strings

def decode_output(output_strings, strategy='baseline', verbose=True):
    """(1 pt) Decode the output strings into a list of integers. Use "t.nan" for failed responses.
    One suggestion is to split on non-numeric characters, then convert to int. And use try/except to catch errors.
    """
    y_hat = []
    for s in output_strings:
        # TODO: y = f(s)
        y_hat.append(y)
    return y_hat

def analyze_results(X, y, y_hats, strategies):
    """(3 pts) Analyze the results.
    Output the accuracy of each strategy.
    Plot a scatter plot of the problems “x1+x2” with x1,x2 on each axis,
    and different plot markers to indicate whether the answer from your LLM was correct.
    (See write-up instructions for requirements on plots)
    
    Train a classifier to predict whether the LLM gave the correct response (using scikit-learn, for example)
    and plot the classifier boundary over the scatter plot with “contour”. (Use whatever classifier looks appropriate)"""
    # TODO


if __name__ == "__main__":
    device = t.device("cuda" if t.cuda.is_available() else "cpu")  # Use GPU if available
    device = t.device('mps') if t.backends.mps.is_available() else device  # Use Apple's Metal backend if available

    X, y = create_dataset(0, 50)
    model, tokenizer = load_LLM(device=device)

    y_hats = []  # list of lists of predicted answers, y_hat, for each strategy
    strategies = ['baseline', 'new']
    for strategy in strategies:
        input_strings = encode_problems(X, strategy=strategy)
        output_strings = generate_text(model, tokenizer, input_strings, device=device)
        output_strings = [out_s[len(in_s):] for in_s, out_s in zip(input_strings, output_strings)]  # Remove the input string from generated answer
        y_hats.append(decode_output(output_strings, strategy=strategy))

    analyze_results(X, y, y_hats, strategies)
