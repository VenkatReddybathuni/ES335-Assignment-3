import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import time
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


st.write("# Text Generator")

st.sidebar.title("Model Information")

st.sidebar.write("Made using a simple 2 hidden layered Neural Network, this text generator can predict next characters of the input text provided.")

st.sidebar.write("Here, the intension is not to generate meaningful sentences, we require a lot of compute for that. This app aims at showing how a vanilla neural network is also capable of capturing the format of English language, and generate words that are (very close to) valid words. Notice that the model uses capital letters (including capital I), punctuation marks and fullstops nearly correct. The text is generated paragraph wise, because the model learnt this from the text corpus.")

st.sidebar.write("This model was trained on a simple 600 KB text corpus titled: 'Gulliver's Travels'")

no_of_chars = st.slider("Number of characters to be generated", 10, 2000, 200)

# Open the file in read mode
with open('warpeace_input.txt', 'r') as file:
    # Read the entire content of the file
    data = file.read()


stoi = {}
stoi['.'] = 0
stoi['\n'] = 1
i = 2
words = sorted(set(data.split()))  # Split content into unique words, sorted lexicographically

for word in words:
    if word not in stoi:
        stoi[word] = i
        i += 1
        
        
itos = {value: key for key, value in stoi.items()}
vocab_size = len(stoi)      
# block_size = 15

man_seed = st.slider("Select the seed", 10, 2000, 42)


g = torch.Generator()
g.manual_seed(man_seed)

# def generate_text(model, inp, itos, stoi, block_size, max_len=no_of_chars):
#     """
#     Generate text at the word level with line breaks after periods.

#     Parameters:
#     - model: Trained PyTorch model
#     - inp: Initial input (seed) words as a string
#     - itos: Dictionary mapping integer indices to words (int-to-string)
#     - stoi: Dictionary mapping words to integer indices (string-to-int)
#     - block_size: Number of words in the context window
#     - max_len: Maximum number of words to generate
#     """
#     # Initialize the context with block_size padding (index 0)
#     context = [0] * block_size

#     # Split input into words
#     inp_words = inp.split()

#     # Fill the context with the input words
#     if len(inp_words) <= block_size:
#         for i in range(len(inp_words)):
#             context[i] = stoi.get(inp_words[i], 0)  # Default to 0 if word not found
#     else:
#         # Use only the last 'block_size' words if input is longer
#         for j, word in enumerate(inp_words[-block_size:]):
#             context[j] = stoi.get(word, 0)

#     # Generate words iteratively
#     generated_text = []
#     for _ in range(max_len):
#         # Convert context to a tensor and pass it through the model
#         x = torch.tensor(context).view(1, -1).to(device)
#         y_pred = model(x)

#         # Sample the next word's index from the model's predictions
#         ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()

#         # Map the index back to a word
#         if ix in itos:
#             word = itos[ix]
#             generated_text.append(word)

#             # Update context with the new word
#             context = context[1:] + [ix]

#             # Add a newline whenever a period is predicted
#             if word == ".":
#                 generated_text.append("\n")

#     # Join the generated words into a single string and clean up whitespace
#     generated_output = ' '.join(generated_text).replace(" \n", "\n").replace("\n ", "\n").strip()

#     return generated_output

def generate_text(model, inp, itos, stoi, block_size, max_len=no_of_chars):

    context = [0] * block_size
    # inp = inp.lower()
    if len(inp) <= block_size:
      for i in range(len(inp)):
        context[i] = stoi[inp[i]]
    else:
      j = 0
      for i in range(len(inp)-block_size,len(inp)):
        context[j] = stoi[inp[i]]
        j+=1

    name = ''
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        if ix in itos:
          ch = itos[ix]
          name += ch
          context = context[1:] + [ix]
    return name

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
    
class NextWordMLP(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_size=1024, block_size=5, activation_fn=nn.ReLU):
        super(NextWordMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(emb_dim * block_size, hidden_size)  # Correct input size for fc1
        self.activation = activation_fn()  # User-specified activation function
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, block_size, emb_dim)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, block_size * emb_dim)
        x = self.activation(self.fc1(x))  # Apply the activation function
        x = self.fc2(x)  # Final output layer
        return x

# Embedding layer for the context

# emb_dim = 10
emb_dim = st.selectbox(
  'Select embedding size',
  (64, 128), index=0)
emb = torch.nn.Embedding(vocab_size, emb_dim)

# block_size = 15
block_size = st.selectbox(
  'Select context length',
  (5, 10, 15), index=0)

activation_fn = st.selectbox(
  'Select activation function',
  (nn.ReLU, nn.Tanh), index=0)

emb = torch.nn.Embedding(vocab_size, emb_dim)
model =  NextWordMLP(vocab_size, emb_dim = emb_dim, hidden_size = 1024, block_size = block_size, activation_fn = activation_fn).to(device)
model = torch.compile(model)

inp = st.text_input("Enter text", placeholder="Enter valid English text. You can also leave this blank.")

btn = st.button("Generate")
if btn:
    st.subheader("Seed Text")
    type_text(inp)
    
    # model._orig_mod.load_state_dict(torch.load("trained_models/model_emb"+str(emb_dim)+"_ctx"+str(block_size)+"_act"+str(activation_fn.__name__)+".pth", map_location = device))
    
    state_dict = torch.load(
        f"trained_models/model_emb{emb_dim}_ctx{block_size}_act{activation_fn.__name__}.pth", 
        map_location=device
    )

    # Check if the model is compiled, and load the state_dict accordingly
    if hasattr(model, "_orig_mod"):  # Handle compiled model case
        model._orig_mod.load_state_dict(state_dict)
    else:  # For standard, non-compiled models
        model.load_state_dict(state_dict)
    
    
    gen_txt = generate_text(model, inp, itos, stoi, block_size, no_of_chars)
    st.subheader("Generated Text")
    print(inp+gen_txt)
    type_text(inp+gen_txt)