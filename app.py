import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import time
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True
import urllib.request
import re
import os
os.environ["TORCHDYNAMO_DISABLE_TRITON"] = "1"

device = torch.device("cpu")


st.write("# Text Generator")

st.sidebar.title("Model Information")

st.sidebar.write("Made using a simple single hidden layered Neural Network, this text generator can predict next word of the input text provided.")

st.sidebar.write("Here, the intension is not to generate meaningful sentences, we require a lot of compute for that. This app aims at showing how a vanilla neural network is also capable of capturing the format of English language, and generate words that are (close to) valid words. Notice that the model uses capital letters (including capital I), punctuation marks and fullstops.")

st.sidebar.write("This model was trained on a simple ebook 'The Adventures of Sherlock Holmes' available at https://www.gutenberg.org/files/1661/1661-0.txt")

no_of_chars = st.slider("Number of words to be generated", 10, 2000, 200)
man_seed = st.slider("Select the seed", 10, 2000, 42)


g = torch.Generator()
g.manual_seed(man_seed) 

url = "https://www.gutenberg.org/files/1661/1661-0.txt"
response = urllib.request.urlopen(url)
sherlock_text = response.read().decode("utf-8")
sherlock_text = sherlock_text[1504:]


def generate_stoi(text):
    """
    Generates a dataset for word-based prediction from input text.
    
    Args:
        text (str): Input text to process.
        block_size (int): Number of words used as context to predict the next word.
        print_limit (int): Number of (context, target) pairs to print for visualization.

    Returns:
        X (torch.Tensor): Input tensor containing contexts.
        Y (torch.Tensor): Output tensor containing target word indices.
        stoi (dict): String-to-index mapping of words.
        itos (dict): Index-to-string mapping of words.
    """
    # Step 1: Split the text into sentences using regex
    sentences = re.split(r'\.\s+|\r\n\r\n', text)

    # Step 2: Clean each sentence and tokenize into words
    cleaned_sentences = [
        re.sub(r'[^a-zA-Z0-9 ]', ' ', sentence).strip()
        for sentence in sentences
    ]

    # **Filter out sentences with fewer than two words**
    cleaned_sentences = [s for s in cleaned_sentences if len(s.split()) >= 2]

    words = [word for sentence in cleaned_sentences for word in sentence.split()]

    # Step 3: Create vocabulary and mappings
    vocabulary = set(words)
    
    stoi = {word: i + 1 for i, word in enumerate(vocabulary)}
    stoi["."] = 0  # Sentence-end marker
    itos = {i: word for word, i in stoi.items()}
    itos[0] = "."  # Ensure "." is included in `itos`
    return stoi, itos

stoi, itos = generate_stoi(sherlock_text)
vocab_size = len(stoi)
class NextWord(nn.Module):
    
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation_fn='ReLU'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        if activation_fn == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_fn == 'Tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function. Use 'relu' or 'tanh'.")

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))  # Apply activation function
        x = self.lin2(x)
        return x



def generate_text(model, itos, stoi, block_size, input_sentence, max_len=100):
    # Convert the input sentence to a list of word indices
    input_indices = [stoi.get(word, 0) for word in input_sentence.split()]  # 0 for unknown words
    
    # Initialize context with the last `block_size` indices of the input sentence
    context = [0] * max(0, block_size - len(input_indices)) + input_indices[-block_size:]
    generated_text = input_sentence.strip() + ' '
    
    for _ in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        
        # Sample the next word
        y_pred = F.softmax(y_pred, dim=-1)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        next_word = itos[ix]
        
        # Append the generated word to the text
        generated_text += next_word + ' '
        
        # Update the context
        context = context[1:] + [ix]
    
    # Remove spaces before periods
    generated_text = generated_text.replace(' .', '.')
    
    return generated_text.strip()


# Function to simulate typing effect
def type_text(text):
    # Create an empty text element
    text_element = st.empty()
    s = ""
    for char in text:
        # Update the text element with the next character
        s += char
        text_element.write(s+'$ꕯ$')
        time.sleep(0.004)  # Adjust the sleep duration for the typing speed

    text_element.write(s)
    
# Embedding layer for the context

emb_dim = st.selectbox(
  'Select embedding size',
  (64, 128), index=0)

# block_size = 15
block_size = st.selectbox(
  'Select context length',
  (5, 10, 15), index=0)

activation_fn = st.selectbox(
  'Select activation function',
  ("ReLU", "Tanh"), index=0)

model =  NextWord(vocab_size = vocab_size, emb_dim = emb_dim, hidden_size = 1024, block_size = block_size, activation_fn = activation_fn).to(device)
model.eval()

inp = st.text_input("Enter text", placeholder="Enter valid English text. You can also leave this blank.")

btn = st.button("Generate")
if btn:
    st.subheader("Seed Text")
    type_text(inp)
    
    
    state_dict = torch.load(
        f"trained_models/model_emb{emb_dim}_ctx{block_size}_act{activation_fn}.pth", 
        map_location=device, weights_only=True
    )
    
    # # Check if the model is compiled, and load the state_dict accordingly
    # if hasattr(model, "_orig_mod"):  # Handle compiled model case
    #     model._orig_mod.load_state_dict(state_dict)
    # else:  # For standard, non-compiled models
    model.load_state_dict(state_dict)
    
    gen_txt = generate_text(model, itos, stoi, block_size,inp, no_of_chars)
    st.subheader("Generated Text")
    type_text(gen_txt)
