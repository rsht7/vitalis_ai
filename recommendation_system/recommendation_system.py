import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
# Load preprocessed dataset and scaler

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# encoded_data_path = os.path.join(BASE_DIR, "encoded_dataset.csv")
encoded_data_path = os.path.join(BASE_DIR, "encoded_dataset.csv")

# scaler_path = os.path.join(BASE_DIR, "trained_scaler.pkl")
scaler_path = os.path.join(BASE_DIR, "trained_scaler.pkl")

encoded_data = pd.read_csv(encoded_data_path)
# scaler = joblib.load(scaler_path)
scaler = joblib.load(scaler_path)


def get_bmi_category(bmi):
    if bmi < 18.5:
        return 3  # Underweight
    elif 18.5 <= bmi < 25:
        return 0  # Normal
    elif 25 <= bmi < 30:
        return 2  # Overweight
    else:
        return 1  # Obese

def get_goal_code(goal):
    goal = goal.lower()
    if goal == "weight gain":
        return 0
    elif goal == "weight loss":
        return 1
    else:
        return 0  # Defaulting to weight gain if unknown

# def get_fitness_type_code(fitness_type):
#     fitness_type = fitness_type.lower()
#     if "muscular" in fitness_type:
#         return 1
#     return 0  # Cardio

def recommend_diet_exercise(user_info, data=encoded_data, scaler=scaler, top_n=3):
    """
    Takes in user_info dict with keys:
    - 'gender', 'age', 'height_cm', 'weight_kg', 'fitness_goal', 'fitness_type'
    Optional keys: 'hypertension', 'diabetes'
    Returns top 3 recommendations.
    """
    height_m = user_info["height_cm"] / 100
    weight = user_info["weight_kg"]
    age = user_info['age']
    bmi = weight / (height_m ** 2)
    level = get_bmi_category(bmi)
    hypertension = user_info['hypertension']
    diabetes = user_info['diabetes']

    input_data = {
        "Sex": 1 if user_info["gender"].lower() == "male" else 0,
        "Age": float(age),
        "Height": float(height_m),
        "Weight": float(weight),
        # "Hypertension": 1 if user_info.get("hypertension", "no") == "yes" else 0,
        # "Diabetes": 1 if user_info.get("diabetes", "no") == "yes" else 0,
        "Hypertension": 1 if hypertension == "yes" else 0,
        "Diabetes": 1 if diabetes == "yes" else 0,
        "BMI": float(bmi),
        "Level": int(level),
        "Fitness Goal": int(get_goal_code(user_info["fitness_goal"])),
        # "Fitness Type": get_fitness_type_code(user_info["fitness_type"]),
    }

    # num_features = ["Age", "Height", "Weight", "BMI"]
    # user_df = pd.DataFrame([input_data])
    # user_df[num_features] = scaler.transform(user_df[num_features])
    # Normalize numerical features
    num_features = ['Age', 'Height', 'Weight', 'BMI']
    user_df = pd.DataFrame([input_data], columns=num_features)
    user_df[num_features] = scaler.transform(user_df[num_features])
    input_data.update(user_df.iloc[0].to_dict())
    user_df = pd.DataFrame([input_data])

    user_features = data[["Sex", "Age", "Height", "Weight", "Hypertension", "Diabetes", "BMI", "Level", "Fitness Goal"]]
    similarity_scores = cosine_similarity(user_features, user_df).flatten()
    similar_indices = similarity_scores.argsort()[-5:][::-1]
    similar_users = data.iloc[similar_indices]
    # Filter data to only those with the same fitness goal
    # filtered_data = data[data["Fitness Goal"] == input_data["Fitness Goal"]].copy()

    # if filtered_data.empty:
    #     return [{"Exercises": "No match", "Diet": "No match", "Equipment": "No match", "Recommendation": "No match"}]

    # user_features = filtered_data[["Sex", "Age", "Height", "Weight", "Hypertension", "Diabetes", "BMI", "Level", "Fitness Goal"]]
    # similarity_scores = cosine_similarity(user_features, user_df).flatten()
    # similar_indices = similarity_scores.argsort()[-5:][::-1]
    # similar_users = filtered_data.iloc[similar_indices]

    recommendation_1 = similar_users[["Exercises", "Diet", "Equipment", "Recommendation"]].mode().iloc[0]

    # simulated_recommendations = []
    # for _ in range(2):
    #     modified_input = input_data.copy()
    #     modified_input["Age"] += random.randint(-5, 5)
    #     modified_input["Weight"] += random.uniform(-5, 5)
    #     modified_input["BMI"] += random.uniform(-1, 1)

    #     mod_df = pd.DataFrame([modified_input])
    #     mod_df[num_features] = scaler.transform(mod_df[num_features])

    #     mod_scores = cosine_similarity(user_features, mod_df).flatten()
    #     mod_indices = mod_scores.argsort()[-5:][::-1]
    #     mod_sim_users = data.iloc[mod_indices]
    #     recommendation = mod_sim_users[["Exercises", "Diet", "Equipment","Recommendation"]].mode().iloc[0]

    #     if not any(
    #         rec["Exercises"] == recommendation["Exercises"]
    #         and rec["Diet"] == recommendation["Diet"]
    #         and rec["Equipment"] == recommendation["Equipment"]
    #         and rec["Recommendation"] == recommendation["Recommendation"]
    #         for rec in simulated_recommendations
    #     ):
    #         simulated_recommendations.append(recommendation)

    # return [recommendation_1] + simulated_recommendations
    return [recommendation_1] 

