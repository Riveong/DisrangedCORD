import nltk
import pickle
import markovify
from nltk.tokenize import word_tokenize
import os

nltk.download('punkt-tab') # or just download your own idk

with open("model\\markov_chain_model.pkl", 'rb') as f:
    markov_model = pickle.load(f)

def generate_joke_from_prompt(prompt, model, length=100):
    prompt_tokens = word_tokenize(prompt.lower())
    start_word = prompt_tokens[0]
    
    try:
        joke = model.make_sentence_with_start(start_word, max_overlap_ratio=0.75, tries=100)
    except KeyError:
        joke = None
    
    if not joke:
        joke = model.make_sentence(max_overlap_ratio=0.75, tries=100)
    
    joke = joke.capitalize() if joke else "ARGGGHHH THE VOICES INSIDE MY HEAD ARE TOO STRONG, SHUT UPPP!"
    joke_words = joke.split()
    
    if len(joke_words) < length:
        while len(joke_words) < length:
            next_word = model.make_sentence(max_overlap_ratio=0.75, tries=100).split()[0]
            joke_words.append(next_word)
    elif len(joke_words) > length:
        joke_words = joke_words[:length]
    
    joke = ' '.join(joke_words)
    return f"Did somebody say {prompt}? That reminds me of: {joke}"

# Main program
if __name__ == "__main__":
    user_prompt = input("Enter a prompt for the joke: ")
    generated_joke = generate_joke_from_prompt(user_prompt, markov_model)
    print(generated_joke)
