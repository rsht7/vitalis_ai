# import spacy

# nlp = spacy.load("en_core_web_sm")

# def extract_user_info(text):
#     doc = nlp(text.lower())
#     user_info = {
#         "age": None,
#         "height": None,
#         "weight": None,
#         "gender": None,
#         "goal": None
#     }

#     for ent in doc.ents:
#         if ent.label_ == "CARDINAL":
#             if "year" in text or "age" in text:
#                 user_info["age"] = ent.text
#             elif "cm" in ent.text or "height" in text:
#                 user_info["height"] = ent.text
#             elif "kg" in ent.text or "weight" in text:
#                 user_info["weight"] = ent.text

#     for token in doc:
#         if token.text in ["male", "female", "other"]:
#             user_info["gender"] = token.text

#     goal_keywords = {
#         "lose": "weight_loss",
#         "gain": "weight_gain",
#         "build": "muscle_gain",
#         "bulk": "muscle_gain",
#         "cut": "weight_loss",
#         "maintain": "maintain_weight"
#     }
#     for token in doc:
#         if token.lemma_ in goal_keywords:
#             user_info["goal"] = goal_keywords[token.lemma_]

#     return user_info


import spacy
import re

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Keywords for gender and fitness goals
GENDER_KEYWORDS = {
    "male": ["male", "man", "boy", "guy"],
    "female": ["female", "woman", "girl", "lady"]
}

STANDARDIZED_GOALS = {
    "weight loss": ["lose weight", "weight loss", "cut", "fat loss", "cutting", "burn fat"],
    "weight gain": ["gain weight", "bulking", "bulk", "weight gain", "gain muscle", "get stronger", "build muscle", "muscle gain"],
    # "build muscle": ["build muscle", "muscle gain", "get stronger"],
    # "maintain weight": ["maintain weight", "stay the same"],
    # "stay fit": ["stay fit", "stay healthy", "fitness", "get fit"]
}


def extract_gender(text):
    for gender, keywords in GENDER_KEYWORDS.items():
        for word in keywords:
            if word in text.lower():
                return gender
    return None


def extract_goal(text):
    text_lower = text.lower()
    for standard_goal, synonyms in STANDARDIZED_GOALS.items():
        for phrase in synonyms:
            if phrase in text_lower:
                return standard_goal
    return None


def check_condition(text, condition_keywords):
    text = text.lower()
    for keyword in condition_keywords:
        if keyword in text:
            # Check for negation near the keyword
            negation_patterns = [
                r"no\s+" + keyword,
                r"not\s+" + keyword,
                r"don't\s+have\s+" + keyword,
                r"dont\s+have\s+" + keyword,
                r"do\s+not\s+have\s+" + keyword,
                r"haven't\s+had\s+" + keyword,
                r"free\s+of\s+" + keyword,
                r"without\s+" + keyword,
                r"never\s+had\s+" + keyword
            ]
            for pattern in negation_patterns:
                if re.search(pattern, text):
                    return "no"
            return "yes"
    return None

def extract_user_info(text):
    doc = nlp(text)

    user_data = {
        "age": None,
        "height_cm": None,
        "weight_kg": None,
        "gender": extract_gender(text),
        "fitness_goal": extract_goal(text),
        "hypertension": check_condition(text, ["hypertension", "high blood pressure"]),
        "diabetes": check_condition(text, ["diabetes"])
    }

    # age_match = re.search(r"(?:i am|i'm|age|aged)?\s*(\d{1,2})\s*(?:years? old)?", text.lower())
    # if age_match:
    #     user_data["age"] = int(age_match.group(1))

    # 1. Age: better regex to avoid confusion with "5 feet"
    age_match = re.search(r"(?:age|aged)\s*(\d{1,2})", text.lower()) or \
            re.search(r"(\d{1,2})\s*(?:years?\s*old)", text.lower()) or \
            re.search(r"(\d{1,2})\s*years?", text.lower())

    if age_match:
        user_data["age"] = int(age_match.group(1))



    # 2. Height: in feet and inches (e.g., "5ft 8in", "5'8\"", "5 feet 7 inches")
    feet_inch_match = re.search(r"(\d)'(\d{1,2})", text.lower()) or \
                      re.search(r"(\d)\s*(?:ft|feet)\s*(\d{1,2})\s*(?:in|inches)?", text.lower())

    if feet_inch_match:
        feet = int(feet_inch_match.group(1))
        inches = int(feet_inch_match.group(2))
        user_data["height_cm"] = round((feet * 12 + inches) * 2.54, 1)

    for ent in doc.ents:
        
        if ent.label_ == "QUANTITY":
            value = re.search(r"\d+\.?\d*", ent.text)
            unit = ent.text.lower()

            if value:
                val = float(value.group())

                if "kg" in unit or "kilogram" in unit:
                    user_data["weight_kg"] = val
                elif "cm" in unit or "centimeter" in unit:
                    user_data["height_cm"] = val
                # elif "feet" in unit or "ft" in unit:
                #     # Convert feet to cm
                #     user_data["height_cm"] = val * 30.48
                # elif "inch" in unit or "inches" in unit:
                #     # Convert inches to cm
                #     user_data["height_cm"] = val * 2.54

        elif ent.label_ == "CARDINAL":
            # Fallback: Guess weight/height if clearly not age
            num = int(ent.text)
            if 30 < num < 200 and user_data["weight_kg"] is None:
                user_data["weight_kg"] = num
            elif 100 < num < 250 and user_data["height_cm"] is None:
                user_data["height_cm"] = num

    return user_data


