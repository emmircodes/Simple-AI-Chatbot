# Simple AI Chatbot

README file is partly AI generated

A Python-based conversational AI chatbot built with TensorFlow/Keras that responds to user inputs using natural language processing (NLP) and a neural network trained on intent patterns. This is a tutorial from [https://youtu.be/1lwddP0KUEg?si=tsOpUXgr5cZ6fdCQ](https://youtu.be/1lwddP0KUEg?si=tsOpUXgr5cZ6fdCQ)

## Features

- **Intent-based conversations**: Recognizes user intents (greeting, goodbye, name queries, etc.)
- **NLP preprocessing**: Uses NLTK for tokenization and lemmatization
- **Neural network classification**: Deep learning model with dropout layers for accurate intent prediction
- **Easy to extend**: Simply add new intents and patterns to `intents.json` to teach the chatbot new responses
- **Lightweight model**: Pre-trained Keras model stored as `.keras` file

## 📋 Project Structure

├── chatbot.py # Main chatbot script (run this to chat)

├── trainingdata.py # Training script to build/update the model

├── intents.json # Intent patterns and responses (customize here)

├── chatbot_model.keras # Pre-trained neural network model

├── words.pkl # Serialized vocabulary from training data

├── classes.pkl # Serialized intent classes from training data

├── requirements.txt # Python dependencies

└── README.md # This file

## 🚀 Quick Start

### Prerequisites
- Python 3.7+

### Installation 

1. Clone the repository:
git clone <repository-url>
cd Simple-AI-Chatbot

2. Install dependencies:
pip install -r requirements.txt

3. Download NLTK data (required for tokenization and lemmatization):
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

To run the Chatbot, type:

python chatbot.py

You'll see the prompt:

Bot is running
you: 

Type your message and press Enter to get a response from the bot(example):

you: hello
Hi! How are you?

🧠 How It Works
1. Data Preparation (trainingdata.py)
-Loads intents from intents.json
-Tokenizes patterns into individual words
-Applies lemmatization to normalize words (e.g., "working" → "work")
-Removes punctuation

2. Training Data Encoding
-Creates a "bag of words" representation where each word from the -vocabulary is either 1 (present) or 0 (absent)
-One-hot encodes the intent labels
-Shuffles the training data

3. Neural Network Architecture

Input Layer (vocabulary size)
    ↓
Dense Layer (128 neurons, ReLU activation)
    ↓
Dropout (50% drop rate)
    ↓
Dense Layer (64 neurons, ReLU activation)
    ↓
Dropout (50% drop rate)
    ↓
Output Layer (number of intents, Softmax activation)

## Training Configuration:

-Optimizer: SGD (Stochastic Gradient Descent) with momentum and Nesterov acceleration
-Loss function: Categorical crossentropy
-Epochs: 200
-Batch size: 5

4. Inference (chatbot.py)
-Preprocesses user input (tokenization + lemmatization)
-Converts to bag of words
-Feeds to the neural network
-Returns intent predictions with confidence scores
-Selects the highest confidence intent (above 0.25 threshold)
-Returns a random response from the matching intent

📝 Customizing Intents
Edit intents.json to add or modify intents:

{
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["hi", "hello", "hey", "what's up"],
            "responses": ["Hi there!", "Hello!", "Hey, how can I help?"]
        },
        {
            "tag": "your_new_intent",
            "patterns": ["pattern1", "pattern2", "pattern3"],
            "responses": ["response1", "response2"]
        }
    ]
}

After modifying intents.json, retrain the model:

python trainingdata.py

Then run the chatbot again:

python chatbot.py

📦 Dependencies

-TensorFlow/Keras: Deep learning framework for the neural network
-NLTK: Natural language processing for tokenization and lemmatization
-NumPy: Numerical computing
-Pickle: Serialization for saving/loading vocabulary and intents
See requirements.txt for exact versions.

Note: This is a basic chatbot demonstrating NLP and neural network fundamentals. For production use, consider using more advanced frameworks.