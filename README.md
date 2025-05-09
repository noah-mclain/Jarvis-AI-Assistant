Certainly! Here's a comprehensive README for the `intent_classification.ipynb` notebook from the [Jarvis-AI-Assistant](https://github.com/noah-mclain/Jarvis-AI-Assistant) repository. This document explains the code's workflow, the model architecture, datasets used, and how to run the notebook from start to finish.

---

# Jarvis AI Assistant – Intent Classification Module

## Overview

This notebook implements an intent classification system, a crucial component of the Jarvis AI Assistant. The goal is to classify user input into predefined intent categories, enabling the assistant to understand and respond appropriately.

## Table of Contents

* [Dataset](#dataset)
* [Data Preprocessing](#data-preprocessing)
* [Model Architecture](#model-architecture)
* [Training](#training)
* [Evaluation](#evaluation)
* [Usage](#usage)
* [Contributors](#contributors)
* [References](#references)

## Dataset

The notebook utilizes a custom dataset comprising user input phrases labeled with corresponding intents. Each entry in the dataset includes:

* **Text**: A user input sentence (e.g., "What's the weather like today?")
* **Intent**: The category of the user's intent (e.g., `weather_query`)

The dataset is structured in a JSON format with the following schema:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "How are you?"],
      "responses": ["Hello!", "Hi there!", "Greetings!"]
    },
    ...
  ]
}
```

## Data Preprocessing

The preprocessing steps include:

1. **Tokenization**: Breaking down sentences into individual words.
2. **Stemming**: Reducing words to their root form using the PorterStemmer from the NLTK library.
3. **Vocabulary Creation**: Building a sorted list of unique stemmed words from the patterns.
4. **Bag-of-Words Encoding**: Converting each pattern into a binary vector indicating the presence of vocabulary words.

These steps transform textual data into numerical form suitable for model training.

## Model Architecture

The model is a simple feedforward neural network implemented using PyTorch. The architecture consists of:

* **Input Layer**: Size equal to the length of the vocabulary.
* **Hidden Layer**: A fully connected layer with ReLU activation.
* **Output Layer**: Size equal to the number of unique intents, with a softmax activation to produce probability distributions over intents.

The model class is defined as follows:

```python
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = F.relu(self.l1(x))
        out = self.l2(out)
        return out
```

## Training

Training involves the following steps:

1. **Loss Function**: CrossEntropyLoss is used to measure the discrepancy between predicted and actual intents.
2. **Optimizer**: Adam optimizer is employed for efficient gradient descent.
3. **Epochs**: The model is trained over multiple epochs (e.g., 1000) to minimize the loss function.
4. **Batch Processing**: Data is loaded in batches using PyTorch's DataLoader for efficient computation.

The training loop updates model weights to minimize the loss on the training data.

## Evaluation

After training, the model's performance is evaluated by:

* **Accuracy**: Measuring the proportion of correctly predicted intents on a validation set.
* **Testing**: Running the model on unseen inputs to assess generalization.

The model's predictions are compared against true labels to compute accuracy metrics.

## Usage

To use the trained model for intent classification:

1. **Load the Model**: Deserialize the trained model parameters.
2. **Preprocess Input**: Tokenize and stem the user input, then convert it into a bag-of-words vector.
3. **Predict Intent**: Pass the vector through the model to obtain intent probabilities.
4. **Select Response**: Choose an appropriate response based on the predicted intent.

Example usage:

```python
sentence = "Hello, how can you assist me?"
tokens = tokenize(sentence)
X = bag_of_words(tokens, all_words)
X = torch.from_numpy(X).float()
output = model(X)
_, predicted = torch.max(output, dim=0)
intent = tags[predicted.item()]
```

## Contributors

* **Ahmed**: Developed the NLP component using bag-of-words and neural networks.
* **Hamza**: Built the speech recognition module with GRU/CTC and MFCC features.
* **Nada**: Implemented the generative text module using character-level LSTM.
* **Amr**: Created the generative image module with a simple GAN.

## References

* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [NLTK Documentation](https://www.nltk.org/)
* [Original Repository](https://github.com/noah-mclain/Jarvis-AI-Assistant)

---

This README provides a detailed explanation of the intent classification module, covering all aspects from data preprocessing to model deployment.
