# Text Prediction with LSTM Model

This project implements a text prediction model using a Long Short-Term Memory (LSTM) neural network to generate text based on input sequences. Below is a detailed explanation of the workflow, model architecture, and datasets used.

---

## 📁 Project Structure
```
text_prediction/
├── text_prediction.ipynb       # Jupyter Notebook with full code
├── df_text_final.csv          # Dataset (custom text corpus)
├── joined_text.txt            # Combined text from the dataset
├── text_gen_model2.h5         # Pretrained LSTM model
└── history2.p                 # Training history (metrics)
```

---

## 🛠️ Dependencies
- Python 3
- Libraries: 
  ```bash
  pandas, numpy, matplotlib, tensorflow, nltk, pickle, heapq
  ```

---

## 📚 Dataset
- **Source**: Custom dataset (`df_text_final.csv`) containing text data.
- **Preprocessing**:
  - Combined all text entries into a single string (`joined_text`).
  - Tokenized using `nltk.tokenize.RegexpTokenizer` to split words.
  - Sequences of **10 words** are used as input, and the 11th word is the prediction target.
  - Limited to the first **1,000,000 characters** for computational efficiency.

---

## 🧠 Model Architecture
A **2-layer LSTM** model built with TensorFlow/Keras:
```python
model = Sequential()
model.add(LSTM(128, input_shape=(10, vocab_size), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation="softmax"))
```
- **Optimizer**: RMSprop (learning rate = 0.01)
- **Loss**: Categorical cross-entropy
- Trained for **10 epochs** with a batch size of 64.

---

## 🚀 Workflow

### 1. Data Preparation
- Loaded text data from `df_text_final.csv`.
- Tokenized and converted words to lowercase.
- Created input sequences (`X`) and target labels (`y`) using one-hot encoding.

### 2. Training
- Model saved to `text_gen_model2.h5` after training.
- Training history (accuracy/loss) saved to `history2.p`.

### 3. Text Generation
- **`predict_next_word`**: Predicts the top `n` likely next words given an input sequence.
- **`generate_text`**: Generates text iteratively using the model's predictions.

---

## 💡 Example Usage
### Predict Next Word
```python
indices = predict_next_word("I will have to look into this thing because I", 5)
print([unique_tokens[idx] for idx in indices])
# Output: ['would', 'aes', 'ace', 'take', 'wrong']
```

### Generate Text
```python
generated_text = generate_text("The president will", n_words=50, creativity=5)
print(generated_text)
# Output: "The president will way give even get deal head school ae know lazarus use stake..."
```

---

## 📊 Performance
- The model achieves moderate accuracy but may produce repetitive or nonsensical text due to:
  - Limited training data (1M characters).
  - Simplicity of the architecture.
- Training metrics (loss/accuracy) can be visualized from `history2.p`.

---

## 🔧 Improvements
- Use a larger dataset (e.g., Wikipedia, books).
- Increase model depth/width.
- Implement beam search for better text generation.
- Add dropout layers to reduce overfitting.

---

## ⚠️ Limitations
- Struggles with long-term coherence.
- Vocabulary limited to words in the training data.
- Requires GPU for efficient training (T4 GPU used in Colab).

---

## 📝 License
- Code is MIT licensed.
- Dataset may have custom licensing (not specified).
