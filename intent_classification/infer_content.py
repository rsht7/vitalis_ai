# import torch
# from transformers import RobertaTokenizer
# from sklearn.preprocessing import LabelEncoder
# import joblib
# import os
# from train_intent_classifier import RoBERTaIntentClassifier  # Reuse same model structure

# # Load tokenizer
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# # Load model
# model_path = "intent_classification/model/intent_bert.pth"
# model = RoBERTaIntentClassifier()
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()

# # Load label encoder
# label_encoder = joblib.load("intent_classification/model/label_encoder.pt")

# def predict_intent(text):
#     # Tokenize input
#     encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
#     with torch.no_grad():
#         outputs = model(**encoding)
#         logits = outputs
#         predicted_class_id = torch.argmax(logits, dim=1).item()
#         intent = label_encoder.inverse_transform([predicted_class_id])[0]
    
#     return intent


# import torch
# from transformers import RobertaTokenizer, RobertaModel
# from sklearn.preprocessing import LabelEncoder
# import joblib
# from train_intent_classifier import RoBERTaIntentClassifier  # Reuse same model structure

# # Load tokenizer
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# # Load label encoder
# # label_encoder = joblib.load("intent_classification/model/label_encoder.pt")  # or .pkl
# # Allow PyTorch to load LabelEncoder safely
# torch.serialization.add_safe_globals([LabelEncoder])

# # Now load the label encoder
# label_encoder = torch.load("intent_classification/model/label_encoder.pt", weights_only=False)



# # Load pretrained RoBERTa base model
# roberta = RobertaModel.from_pretrained("roberta-base")

# # Set same values used during training
# hidden_size = 768
# num_classes = len(label_encoder.classes_)

# # Load trained intent classifier
# model = RoBERTaIntentClassifier(roberta, hidden_size, num_classes)
# model.load_state_dict(torch.load("intent_classification/model/intent_bert.pth", map_location=torch.device('cpu')))
# model.eval()

# def predict_intent(text):
#     # Tokenize input
#     encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

#     # Make prediction
#     with torch.no_grad():
#         outputs = model(**encoding)
#         logits = outputs
#         predicted_class_id = torch.argmax(logits, dim=1).item()
#         intent = label_encoder.inverse_transform([predicted_class_id])[0]
    
#     return intent


# if __name__ == "__main__":
#     # Example input
#     user_input = "What should I eat to build muscle?"
#     predicted_intent = predict_intent(user_input)
#     print(f"Predicted Intent: {predicted_intent}")


# import torch
# from transformers import RobertaTokenizer
# from sklearn.preprocessing import LabelEncoder
# # from train_intent_classifier import RoBERTaIntentClassifier  # Reuse same model structure
# from .train_intent_classifier import RoBERTaIntentClassifier


# # Load tokenizer
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# # Safely load the label encoder (PyTorch 2.6+ fix)
# torch.serialization.add_safe_globals([LabelEncoder])
# label_encoder = torch.load("intent_classification/model/label_encoder.pt", weights_only=False)

# # Number of output classes
# num_classes = len(label_encoder.classes_)

# # Initialize model with correct number of classes
# model = RoBERTaIntentClassifier(num_labels=num_classes)
# model.load_state_dict(torch.load("intent_classification/model/intent_bert.pth", map_location=torch.device('cpu')))
# model.eval()

# def predict_intent(text):
#     # Tokenize input
#     encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

#     # Predict
#     with torch.no_grad():
#         outputs = model(**encoding)
#         logits = outputs
#         predicted_class_id = torch.argmax(logits, dim=1).item()
#         intent = label_encoder.inverse_transform([predicted_class_id])[0]
    
#     return intent

# if __name__ == "__main__":
#     user_input = "hello brother?"
#     predicted_intent = predict_intent(user_input)
#     print(f"Predicted Intent: {predicted_intent}")



import os
import torch
from transformers import RobertaTokenizer
from sklearn.preprocessing import LabelEncoder
from .train_intent_classifier import RoBERTaIntentClassifier

# Resolve absolute path to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to model files
LABEL_ENCODER_PATH = os.path.join(BASE_DIR,"intent_classification", "model", "label_encoder.pt")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR,"intent_classification", "model", "intent_bert.pth")

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Safely load the label encoder
torch.serialization.add_safe_globals([LabelEncoder])
label_encoder = torch.load(LABEL_ENCODER_PATH, weights_only=False)

# Number of output classes
num_classes = len(label_encoder.classes_)

# Initialize and load the trained model
model = RoBERTaIntentClassifier(num_labels=num_classes)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
model.eval()

def predict_intent(text):
    # Tokenize input
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs
        predicted_class_id = torch.argmax(logits, dim=1).item()
        intent = label_encoder.inverse_transform([predicted_class_id])[0]
    
    return intent
