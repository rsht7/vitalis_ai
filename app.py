from flask import Flask, render_template, request, jsonify
from qna_chat import get_tfidf_answer, get_bert_answer, get_greeting
from intent_classification import predict_intent
from user_info_extraction import extract_user_info
from recommendation_system import recommend_diet_exercise
import random

app = Flask(__name__)

user_data = {
    "age": None,
    "height_cm": None,
    "weight_kg": None,
    "gender": None,
    "fitness_goal": None,
    "hypertension": None,
    "diabetes": None
}

chat_history = []

def is_user_info_complete(data):
    return all(value is not None for value in data.values())

def merge_user_data(existing, new_data):
    for key in existing:
        if existing[key] is None and new_data.get(key) is not None:
            existing[key] = new_data[key]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("message")

    if not user_query:
        return jsonify({"response": "Please enter a message."})

    intent = predict_intent(user_query)

    if intent == "greeting":
        greeting_response = get_greeting(user_query)
        response = greeting_response if greeting_response else "Hello! How can I assist you today?"


    elif intent == "query":
        tfidf_response = get_tfidf_answer(user_query)
        if tfidf_response and tfidf_response.strip():
            response = tfidf_response
        else:
            bert_response = get_bert_answer(user_query)
            if bert_response and bert_response.strip():
                response = bert_response
            else:
                response = "Sorry, I couldn't find an answer to your question."

    # elif intent == "providing_info":
    #     extracted_info = extract_user_info(user_query)
    #     merge_user_data(user_data, extracted_info)

    #     if is_user_info_complete(user_data):
    #         try:
    #             recommendations = recommend_diet_exercise(user_data)
    #             response = "Here's your personalized plan:\n"
    #             for idx, rec in enumerate(recommendations, 1):
    #                 response += f"\nRecommendation {idx}:\n"
    #                 response += f"- Exercises: {rec['Exercises']}\n"
    #                 response += f"- Diet: {rec['Diet']}\n"
    #                 response += f"- Equipment: {rec['Equipment']}\n"
    #                 response += f"- Advice: {rec['Recommendation']}\n"
    #         except Exception as e:
    #             response = f"Error generating recommendations: {e}"
    #     else:
    #         response = "Got it! Tell me more details so I can give a full recommendation."
    elif intent == "providing_info":
        extracted_info = extract_user_info(user_query)
        merge_user_data(user_data, extracted_info)

        if is_user_info_complete(user_data):
            try:
                recommendations = recommend_diet_exercise(user_data)
                response = "Here's your personalized plan:\n"
                for idx, rec in enumerate(recommendations, 1):
                    response += f"\nRecommendation {idx}:\n\n"
                    response += f"- Exercises: {rec['Exercises']}\n\n"
                    response += f"- Diet: {rec['Diet']}\n\n"
                    response += f"- Equipment: {rec['Equipment']}\n\n"
                    response += f"- Advice: {rec['Recommendation']}\n\n"
            except Exception as e:
                response = f"Error generating recommendations: {e}"
        else:
            missing_fields = []
            required_fields = ['age', 'weight_kg', 'height_cm', 'gender', 'fitness_goal', 'hypertension', 'diabetes']

            for field in required_fields:
                if not user_data.get(field):
                    missing_fields.append(field.replace('_', ' ').capitalize())

            response = "Got it! But I still need a few more details to give a full recommendation.\n"
            response += "Missing info: " + ", ".join(missing_fields)


    else:
        response = f"Hmm, I’m not sure how to handle this yet."

    # return jsonify({"response": response})
    # return jsonify({"response": response.replace('\n', '<br>')})
    return jsonify({
    "response": response.replace('\n', '<br>'),
    "user_data": user_data
    })



if __name__ == "__main__":
    app.run(debug=True)
