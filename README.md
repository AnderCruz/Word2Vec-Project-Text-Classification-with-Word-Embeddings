# Word2Vec Project: Text Classification with Word Embeddings

## Project Overview

This project demonstrates the application of Word2Vec embeddings for text classification tasks. Using Brazilian Portuguese news articles from Folha de São Paulo, we implement and compare two Word2Vec architectures (CBOW and Skip-gram) to classify articles into categories such as sports, politics, entertainment, and more.

## Features

- **Word Embedding Models**: Pre-trained CBOW and Skip-gram models with 300-dimensional vectors
- **Text Classification**: Logistic Regression classifier for categorizing news articles
- **Data Preprocessing**: Tokenization and text cleaning using spaCy and NLTK
- **Model Comparison**: Performance evaluation of CBOW vs Skip-gram architectures

## Dataset

The project uses two datasets:
- `treino.csv`: Training data with 90,000 articles
- `teste.csv`: Test data with 20,513 articles

Each article contains:
- Title
- Text content
- Publication date
- Category (target variable)
- Subcategory
- Source link

## Models

### Word2Vec Models
- **CBOW (Continuous Bag of Words)**: `cbow_s300.txt`
- **Skip-gram**: `skip_s300.txt`

Both models provide 300-dimensional word vectors trained on Brazilian Portuguese text.

### Classification Approach
1. **Text Vectorization**: Convert article titles to vectors by summing word embeddings
2. **Classification**: Use Logistic Regression to predict article categories
3. **Evaluation**: Compare model performance using precision, recall, and F1-score

## Results

### CBOW Model Performance
- Overall Accuracy: ~79%
- Best performance on Sports category (F1-score: 0.89)
- Categories: colunas, cotidiano, esporte, ilustrada, mercado, mundo

### Skip-gram Model Performance  
- Overall Accuracy: ~80%
- Slightly better performance across most categories
- Improved F1-scores for sports and market categories

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn gensim nltk spacy
```

### Download Portuguese Language Model for spaCy
```bash
python -m spacy download pt_core_news_sm
```

### Basic Usage
```python
# Load pre-trained models
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format("cbow_s300/cbow_s300.txt")

# Get similar words
similar_words = model.most_similar("brasil")
print(similar_words)

# Word analogies
analogy_result = model.most_similar(positive=["mulher", "rei"], negative=["homem"])
print(analogy_result)  # Should return "rainha" (queen)
```

## Project Structure

```
Word2Vec_Project/
├── data/
│   ├── treino.csv
│   ├── teste.csv
│   ├── modelo_cbow.txt
│   └── modelo_skipgram.txt
├── cbow_s300/
│   └── cbow_s300.txt
├── skip_s300/
│   └── skip_s300.txt
├── Word2Vec_Project_Final.ipynb
└── README.md
```

## Key Findings

1. **Skip-gram generally outperforms CBOW** for this classification task
2. **Sports and market categories** are easiest to classify with high F1-scores
3. **Entertainment category (ilustrada)** shows lower performance, possibly due to diverse content
4. **Word embeddings effectively capture semantic relationships** in Portuguese

## Applications

This approach can be extended to:
- News categorization systems
- Content recommendation engines
- Semantic search applications
- Other Portuguese NLP tasks

## Future Work

- Fine-tune Word2Vec models on domain-specific data
- Experiment with other classification algorithms (Neural Networks, SVM)
- Incorporate article text content in addition to titles
- Explore transformer-based models (BERT) for comparison

## Contributors

This project demonstrates practical applications of word embeddings for Portuguese text classification, suitable for educational purposes and as a foundation for more advanced NLP applications.
