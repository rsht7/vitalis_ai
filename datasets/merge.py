import pandas as pd

gq_data = pd.read_csv("ordered_dataset.csv", index_col=0)
nutri_data = pd.read_csv("nutrition_qna_data.csv")

qa_data = pd.concat([gq_data, nutri_data], ignore_index=True)
qa_data.to_csv("qa_data_combined.csv", index=False)