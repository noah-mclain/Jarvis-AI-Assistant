import spacy
print(f'spaCy version: {spacy.__version__}')
try:
    nlp = spacy.load('en_core_web_sm')
    print('en_core_web_sm model loaded successfully')
    
    # Test basic NLP functionality
    doc = nlp("This is a test sentence for spaCy.")
    print("Tokenization test:")
    print([token.text for token in doc])
    
    print("\nPart-of-speech tagging test:")
    print([(token.text, token.pos_) for token in doc])
    
except Exception as e:
    print(f'Error loading model: {e}') 