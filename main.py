
# from qna_chat import get_tfidf_answer, get_bert_answer, greeting_responses
# from intent_classification import predict_intent
# from user_info_extraction import extract_user_info  # <- NEW
# from recommendation_system import recommend_diet_exercise


# import random

# # Store user details across multiple inputs
# user_data = {
#     "age": None,
#     "height_cm": None,
#     "weight_kg": None,
#     "gender": None,
#     "fitness_goal": None,
#     "hypertension": None,
#     "diabetes": None
# }


# def is_user_info_complete(data):
#     return all(value is not None for value in data.values())

# def merge_user_data(existing, new_data):
#     for key in existing:
#         if existing[key] is None and new_data.get(key) is not None:
#             existing[key] = new_data[key]

# def run_chatbot():
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
#             tfidf_response = get_tfidf_answer(user_query)
#             if tfidf_response and tfidf_response.strip():
#                 print(f"Chatbot: {tfidf_response}")
#             else:
#                 bert_response = get_bert_answer(user_query)
#                 if bert_response and bert_response.strip():
#                     print(f"Chatbot: {bert_response}")
#                 else:
#                     print("Chatbot: Sorry, I couldn't find an answer to your question.")

#         elif intent == "providing_info":
#             extracted_info = extract_user_info(user_query)
#             merge_user_data(user_data, extracted_info)

#             print("Chatbot: Got it! Here's what I have so far:")
#             for key, value in user_data.items():
#                 print(f"  - {key}: {value}")

#             if is_user_info_complete(user_data):
#                 print("\n✅ All information received!")
#                 print("📊 Ready to calculate your calorie needs and build your custom diet plan.")
#                 # 👉 You can call a function here to calculate BMR and recommend a diet
#                 # e.g. generate_diet_plan(user_data)

#         else:
#             print(f"Chatbot: Hmm, I detected intent '{intent}', but I’m not sure how to handle it yet.")

# if __name__ == "__main__":
#     run_chatbot()





# CODE WHEN IT WAS MAIN.PY

# from qna_chat import get_tfidf_answer, get_bert_answer, greeting_responses
# from intent_classification import predict_intent
# from user_info_extraction import extract_user_info
# from recommendation_system import recommend_diet_exercise

# import random

# # Store user details across multiple inputs
# user_data = {
#     "age": None,
#     "height_cm": None,
#     "weight_kg": None,
#     "gender": None,
#     "fitness_goal": None,
#     "hypertension": None,
#     "diabetes": None
# }


# def is_user_info_complete(data):
#     return all(value is not None for value in data.values())

# def merge_user_data(existing, new_data):
#     for key in existing:
#         if existing[key] is None and new_data.get(key) is not None:
#             existing[key] = new_data[key]

# def run_chatbot():
#     print("🏋️‍♂️ Fitness & Diet Chatbot - Ask your question! (Type 'exit' to quit)")

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
#             tfidf_response = get_tfidf_answer(user_query)
#             if tfidf_response and tfidf_response.strip():
#                 print(f"Chatbot: {tfidf_response}")
#             else:
#                 bert_response = get_bert_answer(user_query)
#                 if bert_response and bert_response.strip():
#                     print(f"Chatbot: {bert_response}")
#                 else:
#                     print("Chatbot: Sorry, I couldn't find an answer to your question.")

#         elif intent == "providing_info":
#             extracted_info = extract_user_info(user_query)
#             merge_user_data(user_data, extracted_info)

#             print("Chatbot: Got it! Here's what I have so far:")
#             for key, value in user_data.items():
#                 print(f"  - {key}: {value}")

#             if is_user_info_complete(user_data):
#                 print("\n✅ All information received!")
#                 print("📊 Ready to calculate your calorie needs and build your custom diet plan.")

#                 try:
#                     recommendations = recommend_diet_exercise(user_data)

#                     print("\n🍽️ Personalized Diet & Exercise Recommendations:")

#                     for idx, rec in enumerate(recommendations, 1):
#                         print(f"\n🔹 Recommendation {idx}:")
#                         print(f"   • Exercises: {rec['Exercises']}")
#                         print(f"   • Diet: {rec['Diet']}")
#                         print(f"   • Equipment: {rec['Equipment']}")
#                         print(f"   • Extra Advice: {rec['Recommendation']}")

#                 except Exception as e:
#                     print("❌ Error generating recommendations:", e)

#         else:
#             print(f"Chatbot: Hmm, I detected intent '{intent}', but I’m not sure how to handle it yet.")

# if __name__ == "__main__":
#     run_chatbot()


