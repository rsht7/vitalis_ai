from docx import Document

# Load the .docx file
doc = Document("wikipedia_fitness_article.docx")

# Extract text while ensuring empty lines between paragraphs
text = "\n\n".join([para.text for para in doc.paragraphs])

# Save to .txt
with open("wikipedia_article.txt", "w", encoding="utf-8") as txt_file:
    txt_file.write(text)

print("Conversion complete! Saved as 'wikipedia_article.txt' with paragraph spacing preserved.")
