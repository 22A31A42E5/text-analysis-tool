import os
from flask import Flask, request, render_template
import PyPDF2
import docx
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Initialize Flask app
app = Flask(__name__)

# Load the spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text
    return text

# Function for sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Polarity: -1 (negative) to 1 (positive)
    return sentiment

# Function to extract keywords using TF-IDF
def extract_keywords(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    keyword_scores = zip(feature_names, scores)
    sorted_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
    return [keyword for keyword, score in sorted_keywords[:5]]  # Top 5 keywords

# Function to extract named entities using spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file uploads
@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    # Save the uploaded file to the uploads folder
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Extract text from the uploaded file
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file.filename.endswith('.docx'):
        text = extract_text_from_docx(file_path)
    else:
        return "Unsupported file format. Please upload a PDF or DOCX file."

    # Perform analysis
    word_count = len(text.split())
    sentiment = analyze_sentiment(text)
    keywords = extract_keywords(text)
    entities = extract_entities(text)

    # Generate a report
    report = {
        "word_count": word_count,
        "sentiment": "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral",
        "keywords": keywords,
        "text_summary": text[:500],  # Display the first 500 characters as summary
        "entities": entities  # Add named entities to the report
    }

    # Render the results in result.html
    return render_template('result.html', report=report, text=text)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
