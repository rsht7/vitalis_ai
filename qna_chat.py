


# #lasttt

# import pandas as pd
# import numpy as np
# import re
# import nltk
# # nltk.download('punkt_tab')
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import pipeline
# from nltk.tokenize import sent_tokenize



# # Load structured Q&A dataset
# qa_data = pd.read_csv("datasets/qa_data_combined.csv")  # Ensure correct path
# questions = qa_data["Question"].astype(str).values
# answers = qa_data["Answer"].astype(str).values

# # Load NLTK stopwords and lemmatizer
# stop_words = set(stopwords.words("english"))
# lemmatizer = nltk.WordNetLemmatizer()

# def preprocess_text(text):
#     """Preprocess text: lowercase, remove punctuation, remove stopwords, and lemmatize."""
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
#     tokens = word_tokenize(text)  # Tokenize text
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords & lemmatize
#     return " ".join(tokens)  # Reconstruct text

# # Preprocess all questions before vectorization
# preprocessed_questions = [preprocess_text(q) for q in questions]

# # TF-IDF Vectorization on Preprocessed Questions
# vectorizer = TfidfVectorizer()
# question_vectors = vectorizer.fit_transform(preprocessed_questions)

# def get_tfidf_answer(user_query):
#     """Find the best matching answer using TF-IDF similarity."""
#     query = preprocess_text(user_query)  # Preprocess user query
#     query_vector = vectorizer.transform([query])  # Convert to TF-IDF vector
#     similarity_scores = cosine_similarity(query_vector, question_vectors).flatten()  # Compute cosine similarity
#     best_match_idx = np.argmax(similarity_scores)  # Get index of best match

#     # Set a similarity threshold to filter out weak matches
#     if similarity_scores[best_match_idx] > 0.3:  
#         return answers[best_match_idx]  # Return the best-matching answer
    
#     return None  # Return None if no strong match is found



# #BERT part
# # Load pretrained BERT-based QA model
# qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

# # Load unstructured text data
# with open("data/raw_info.txt", "r", encoding="utf-8") as file:
#     context_text = file.read()

# # Tokenize text into sentences (useful for precise answers)
# sentences = sent_tokenize(context_text)

# def get_bert_answer(user_query):
#     """
#     Extract answer from unstructured fitness & diet information using BERT.
#     """
#     best_answer = None
#     best_score = 0

#     for sentence in sentences:
#         result = qa_pipeline(question=user_query, context=sentence)

#         if result["score"] > best_score:
#             best_answer = result["answer"]
#             best_score = result["score"]

#     return best_answer if best_score > 0.5 else "Sorry, I couldn't find an answer."

# if __name__ == "__main__":
#     print("Fitness & Diet Chatbot (Structured Q&A) - Ask your question!")
#     while True:
#         user_query = input("You: ")
#         if user_query.lower() in ["exit", "quit"]:
#             print("Chatbot: Goodbye!")
#             break
        
#         response = get_tfidf_answer(user_query)
#         if response:
#             print(f"Chatbot: {response}")
#         else:
#             print("Chatbot: Sorry, I couldn't find an answer in the structured dataset.")






import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import contractions

import random
from intent_classification.infer_content import predict_intent

# 10 different greeting styles
# greeting_responses = [
#     "Hey there! 👋",
#     "Hello! How can I help you today?",
#     "Hi! Need some fitness advice?",
#     "Hey! Let’s talk health and gains 💪",
#     "Yo! What's up?",
#     "Hi there! Ready to crush your fitness goals?",
#     "Good to see you! 😊",
#     "Hello friend! How can I assist?",
#     "Wassup! Ready to get fit?",
#     "Hey champ! Let's get started!"
# ]

# Ensure required NLTK data is downloaded
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")


greet_data = pd.read_csv("datasets/vitalis_greetings_dataset3.csv")  # Ensure correct path
greetings = greet_data["user_greeting"].astype(str).values
greet_reply = greet_data["bot_reply"].astype(str).values

# Load structured Q&A dataset
qa_data = pd.read_csv("datasets/qa_data_combined.csv")  # Ensure correct path
questions = qa_data["Question"].astype(str).values
answers = qa_data["Answer"].astype(str).values

# Load NLTK stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = nltk.WordNetLemmatizer()

# def preprocess_text(text):
#     """Preprocess text: lowercase, remove punctuation, remove stopwords, and lemmatize."""
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
#     tokens = word_tokenize(text)  # Tokenize text
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords & lemmatize
#     return " ".join(tokens)  # Reconstruct text



# # greetings data preprocessing
# preprocessed_user_greetings = [preprocess_text(greet) for greet in greetings]

# vectorizer_greet_data = TfidfVectorizer()
# greetings_vectors = vectorizer_greet_data.fit_transform(preprocessed_user_greetings)

# def get_greeting(user_greeting):
#     greet = preprocess_text(user_greeting)  # Preprocess user greeting
#     greet_vector = vectorizer_greet_data.transform([greet])  # Convert to TF-IDF vector
#     similarity_greet_scores = cosine_similarity(greet_vector, greetings_vectors).flatten()  # Compute cosine similarity
#     best_match_idx = np.argmax(similarity_greet_scores)  # Get index of best match

#     return greet_reply[best_match_idx]  # Return the best-matching reply

def preprocess_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Preprocess greetings
preprocessed_user_greetings = [preprocess_text(greet) for greet in greetings]

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can switch to another model if needed

# Encode all greetings to embeddings
greetings_vectors = model.encode(preprocessed_user_greetings, convert_to_tensor=True)



# Function to get the best matching reply
def get_greeting(user_greeting):
    preprocessed = preprocess_text(user_greeting)

    if preprocessed in preprocessed_user_greetings:
        idx = preprocessed_user_greetings.index(preprocessed)
        return greet_reply[idx]
    query_vector = model.encode(preprocessed, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarity_scores = util.cos_sim(query_vector, greetings_vectors)[0]
    best_score = float(similarity_scores.max())
    best_match_idx = int(np.argmax(similarity_scores))

    # if best_score < threshold:
    #     return "I'm not sure how to respond to that, but I'm here to help!"
    
    return greet_reply[best_match_idx]



# Queries data preprocessing
# Preprocess all questions before vectorization
preprocessed_questions = [preprocess_text(q) for q in questions]

# TF-IDF Vectorization on Preprocessed Questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(preprocessed_questions)

def get_tfidf_answer(user_query):
    """Find the best matching answer using TF-IDF similarity."""
    query = preprocess_text(user_query)  # Preprocess user query
    query_vector = vectorizer.transform([query])  # Convert to TF-IDF vector
    similarity_scores = cosine_similarity(query_vector, question_vectors).flatten()  # Compute cosine similarity
    best_match_idx = np.argmax(similarity_scores)  # Get index of best match

    # Set a similarity threshold to filter out weak matches
    if similarity_scores[best_match_idx] > 0.6:  
        return answers[best_match_idx]  # Return the best-matching answer
    
    return None  # Return None if no strong match is found

# Load pretrained BERT-based QA model
# qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")



# Load unstructured text data
with open("datasets/wikipedia_article.txt", "r", encoding="utf-8") as file:
    context_text = file.read()

# # Tried - Tokenize text into sentences for better precision
# sentences = sent_tokenize(context_text)

# def get_bert_answer(user_query):
#     """Extract answer from unstructured fitness & diet information using BERT."""
#     best_answer = None
#     best_score = 0

#     for sentence in sentences:
#         result = qa_pipeline(question=user_query, context=sentence)

#         if result["score"] > best_score:
#             best_answer = result["answer"]
#             best_score = result["score"]

#     return best_answer if best_score > 0.1 else None  # Return None if BERT fails



# Tried - Chunk the context text into groups of ~250 words for better context
# def chunk_text(text, chunk_size=250):
#     words = text.split()
#     return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# text_chunks = chunk_text(context_text, chunk_size=250)

# def get_bert_answer(user_query):
#     """Extract answer from unstructured fitness & diet information using BERT over larger chunks."""
#     best_answer = None
#     best_score = 0

#     for chunk in text_chunks:
#         result = qa_pipeline(question=user_query, context=chunk)
#         if result["score"] > best_score:
#             best_answer = result["answer"]
#             best_score = result["score"]

#     return best_answer if best_score > 0.4 else None  # Lower threshold if needed



# Split text into paragraphs using double newlines
paragraphs = [p.strip() for p in context_text.split('\n\n') if p.strip()]

def get_bert_answer(user_query):
    """Extract answer from unstructured fitness & diet information using BERT."""
    best_answer = None
    best_score = 0

    for para in paragraphs:
        result = qa_pipeline(question=user_query, context=para)

        if result["score"] > best_score:
            best_answer = result["answer"]
            best_score = result["score"]

    return best_answer if best_score > 0.1 else None  # Tweak this threshold








# if __name__ == "__main__":
#     print("Fitness & Diet Chatbot - Ask your question! (Type 'exit' to quit)")
    
#     while True:
#         user_query = input("You: ").strip()
        
#         if user_query.lower() in ["exit", "quit"]:
#             print("Chatbot: Goodbye! Stay healthy! 💪")
#             break

#         # Step 1: Classify Intent
#         intent = predict_intent(user_query)

#         # Step 2: Respond based on intent
#         if intent == "greeting":
#             print("Chatbot:", random.choice(greeting_responses))

#         elif intent == "query":
#             # Try structured Q&A first (TF-IDF)
#             tfidf_response = get_tfidf_answer(user_query)
#             if tfidf_response and tfidf_response.strip():
#                 print(f"Chatbot: {tfidf_response}")
#             else:
#                 # Fall back to BERT
#                 bert_response = get_bert_answer(user_query)
#                 if bert_response and bert_response.strip():
#                     print(f"Chatbot: {bert_response}")
#                 else:
#                     print("Chatbot: Sorry, I couldn't find an answer to your question.")

#         else:
#             print(f"Chatbot: Hmm, I detected intent '{intent}', but I’m not sure how to handle it yet.")