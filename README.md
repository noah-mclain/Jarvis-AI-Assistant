# Intent Classification with BERT

## Overview
This project focuses on intent classification using the BERT (Bidirectional Encoder Representations from Transformers) model. The goal is to classify user queries into predefined intents (e.g., "share location," "set alarm," "play music"). The model is trained on a dataset of 21,769 labeled examples across 148 unique intents.

---

## Key Components

### 1. **Dataset**
- **Source**: Custom dataset (`pre_intent.csv`) with 2 columns:
  - `text`: User query (e.g., "send my location to my husband").
  - `label`: Intent label (e.g., `sharecurrentlocation`).
- **Size**: 21,769 examples.
- **Unique Intents**: 148 (e.g., `gettime`, `playmusic`, `managegroupchat`).

### 2. **Preprocessing**
- **Tokenization**: Split text into tokens using `wordsegment` to handle concatenated words.
- **Stopword Removal**: Eliminate common English stopwords via `nltk.corpus.stopwords`.
- **Lemmatization**: Reduce words to base forms using `WordNetLemmatizer`.
- **Label Encoding**: Convert string labels to numerical values with `LabelEncoder`.

### 3. **Model Architecture**
- **Base Model**: `bert-base-uncased` (12-layer, 768-hidden, 12-heads, 110M parameters).
- **Fine-Tuning**:
  - Added a classification head for 148 intents.
  - Trained for **100 epochs** with `AdamW` optimizer (learning rate = 3e-5).
  - Batch size = 16, max sequence length = 32.

### 4. **Training**
- **Train/Test Split**: 80/20 split (`train_test_split`).
- **Loss Function**: Cross-entropy loss.
- **Hardware**: GPU acceleration (Google Colab L4 GPU).

### 5. **Evaluation**
- **Metrics**: Precision, recall, F1-score (micro/macro averages).
- **Challenges**:
  - Class imbalance (some intents had limited examples).
  - Warnings due to missing labels in test/train splits.

### 6. **Inference**
- **Gradio Interface**: Deployed a web app for real-time predictions.
- **Example Prediction**:
  ```python
  Input: "What's the weather today?"
  Output: "weatherquery"
  ```

---

## Code Workflow

### 1. **Data Loading & Inspection**
- Load `pre_intent.csv` and display unique labels.

### 2. **Text Preprocessing**
- Tokenize, remove stopwords, and lemmatize text/labels.

### 3. **Train/Test Split**
- Split data into 80% training and 20% testing.

### 4. **Model Setup**
- Load pre-trained BERT, add classification layer, and move to GPU.

### 5. **Training Loop**
- Train for 100 epochs with progress tracking.

### 6. **Evaluation**
- Generate predictions and calculate classification metrics.

### 7. **Saving the Model**
- Save weights and full checkpoint for deployment.

### 8. **Deployment**
- Create a Gradio interface for interactive predictions.

---

## Results
- Achieved **~46% weighted F1-score** on the test set.
- High performance on common intents (e.g., `bookrestaurant`: 96% F1).
- Struggled with rare intents due to limited examples.

---

## How to Use

### 1. **Install Dependencies**
```bash
pip install torch transformers pandas nltk wordsegment gradio
```

### 2. **Run Inference**
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=148)
model.load_state_dict(torch.load("model_weights.pth"))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Predict
text = "set an alarm for 6 AM"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()
```

### 3. **Launch Gradio App**
```python
import gradio as gr

iface = gr.Interface(
    fn=predict_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter your message..."),
    outputs="text",
    title="Intent Classifier"
)
iface.launch()
```

---

## Future Improvements
1. **Data Augmentation**: Generate synthetic examples for rare intents.
2. **Hyperparameter Tuning**: Optimize learning rate, batch size, and epochs.
3. **Multilingual Support**: Add support for non-English queries.
4. **Error Analysis**: Investigate misclassified examples to refine the model.

---

## Files
- `intent_classification.ipynb`: Jupyter notebook with full code.
- `pre_intent.csv`: Dataset (not included in repo).
- `model_weights.pth`: Saved model weights.
