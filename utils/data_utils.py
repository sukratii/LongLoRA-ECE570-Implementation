import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset

from datasets import load_dataset

def load_water_policy_dataset(split):
    dataset = load_dataset('csv', data_files={split: "/PhD/Courses/ECE570/Water_Kansas.csv"}, split=split)
    return dataset

def preprocess_policy_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove short words (length < 3)
    tokens = [token for token in tokens if len(token) > 2]
    
    # Join tokens back into a string
    processed_text = ' '.join(tokens)
    
    # Remove extra whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text
