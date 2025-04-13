from datasets import load_dataset

dataset = load_dataset("its-myrto/fitness-question-answers")

dataset['train'].to_csv('datasets/ordered_dataset.csv')