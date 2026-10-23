# 🎬 IMDB Movie Review Sentiment Analysis using Simple RNN

An end-to-end Deep Learning project that uses a **Simple Recurrent Neural Network (Simple RNN)** to classify movie reviews as positive or negative. Built using TensorFlow and Keras, the project demonstrates text preprocessing, word embeddings, recurrent neural networks, model training, and sentiment prediction through an interactive Streamlit application.

## 📌 Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task that identifies the emotional tone or opinion expressed in a piece of text.

In this project, we build a binary sentiment classification model using the IMDB movie reviews dataset. The model processes numerical representations of movie reviews and predicts whether each review expresses positive or negative sentiment.

The project covers the complete workflow, from loading and preprocessing textual data to training a neural network and deploying a simple interactive application for predictions.

### 🎯 Objectives

* Understand text preprocessing for Natural Language Processing.
* Explore word indexing and embedding representations.
* Build a Simple RNN model using TensorFlow and Keras.
* Train a neural network for binary sentiment classification.
* Use binary cross-entropy as the loss function and Adam as the optimizer.
* Apply early stopping to monitor validation loss during training.
* Save and load a trained model for inference.
* Build an interactive web interface using Streamlit.

## 🧠 Model Architecture

The project uses a neural network consisting of three main layers:

1. **Embedding Layer:** Converts integer-encoded tokens into dense vectors of 128 dimensions.
2. **Simple RNN Layer:** Processes the sequence of embeddings and produces a final hidden representation.
3. **Dense Output Layer:** Uses a sigmoid activation function to generate a sentiment score between 0 and 1.

### Architecture Configuration

| Component               | Configuration                   |
| ----------------------- | ------------------------------- |
| Dataset                 | IMDB Movie Reviews              |
| Task                    | Binary Sentiment Classification |
| Vocabulary Size         | 10,000                          |
| Maximum Sequence Length | 500                             |
| Embedding Dimension     | 128                             |
| Recurrent Layer         | SimpleRNN                       |
| Recurrent Units         | 128                             |
| RNN Activation          | ReLU                            |
| Output Layer            | Dense with 1 neuron             |
| Output Activation       | Sigmoid                         |
| Optimizer               | Adam                            |
| Loss Function           | Binary Cross-Entropy            |
| Maximum Training Epochs | 10                              |
| Batch Size              | 32                              |
| Validation Split        | 20%                             |
| Early Stopping Patience | 5 epochs                        |

The model is trained using the training dataset, while validation data is used to monitor performance during training.

## 📊 Dataset

This project uses the IMDB movie reviews dataset available through `tensorflow.keras.datasets.imdb`.

The dataset contains 50,000 movie reviews divided into training and testing sets.

| Dataset Split | Number of Reviews | Purpose                       |
| ------------- | ----------------: | ----------------------------- |
| Training Set  |            25,000 | Model training and validation |
| Testing Set   |            25,000 | Held-out evaluation           |
| **Total**     |        **50,000** | **Sentiment classification**  |

The dataset uses binary labels:

* `0` — Negative sentiment
* `1` — Positive sentiment

The reviews are represented as integer sequences corresponding to tokens in the dataset vocabulary.

The project restricts the vocabulary to 10,000 indices and pads or truncates reviews to a maximum sequence length of 500.

## ⚙️ Project Workflow

The project follows these steps:

1. **Data Loading:** Load the IMDB dataset using TensorFlow/Keras.
2. **Data Inspection:** Examine encoded reviews and their sentiment labels.
3. **Vocabulary Mapping:** Use the dataset's word index to decode integer sequences into readable text for inspection.
4. **Sequence Preprocessing:** Pad or truncate reviews to a fixed length of 500 tokens.
5. **Embedding Representation:** Convert token IDs into trainable 128-dimensional vectors.
6. **RNN Model Development:** Build a Simple RNN architecture for sequence classification.
7. **Model Compilation:** Configure the Adam optimizer, binary cross-entropy loss, and accuracy metric.
8. **Model Training:** Train the model with a batch size of 32 and a maximum of 10 epochs.
9. **Early Stopping:** Monitor validation loss and restore the best observed weights.
10. **Model Saving:** Save the trained network as an HDF5 model file.
11. **Sentiment Prediction:** Load the saved model and predict sentiment for input reviews.
12. **Web Application:** Provide an interactive interface using Streamlit.

## 🛠️ Technologies Used

* **Python** — Programming language
* **TensorFlow** — Deep Learning framework
* **Keras** — Neural network development and training
* **NumPy** — Numerical computing
* **Pandas** — Data manipulation library
* **Matplotlib** — Data visualization
* **Scikit-learn** — Machine Learning utilities
* **TensorBoard** — Training visualization and monitoring
* **Streamlit** — Interactive web application
* **Jupyter Notebook** — Experimentation and development

## 📁 Repository Structure

```text
IMDB-Movie-Review-Sentiment-Analysis/
│
├── simplernn.ipynb
│   └── Dataset preprocessing, Simple RNN architecture,
│       training, and model saving
│
├── embedding.ipynb
│   └── Demonstration of integer encoding,
│       padding, and embedding representations
│
├── prediction.ipynb
│   └── Loading the trained model and predicting
│       sentiment for example reviews
│
├── main.py
│   └── Streamlit application for interactive
│       movie review sentiment prediction
│
├── simple_rnn_imdb.h5
│   └── Saved trained model
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation
```

## 🚀 Getting Started

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/IMDB-Movie-Review-Sentiment-Analysis.git
```

Navigate to the project directory:

```bash
cd IMDB-Movie-Review-Sentiment-Analysis
```

Replace `YOUR-USERNAME` with your GitHub username and use your actual repository URL.

### 2. Create a Virtual Environment

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

The project specifies TensorFlow 2.15.0 in its requirements file.

### 4. Run the Streamlit Application

Ensure that `main.py` and `simple_rnn_imdb.h5` are present in the project directory.

Run:

```bash
streamlit run main.py
```

Streamlit will start a local web server and provide a URL in the terminal. Open that URL in your browser to access the application.

### 5. Perform Sentiment Analysis

1. Open the Streamlit application.
2. Enter a movie review in the text input area.
3. Click the **Classify** button.
4. View the predicted sentiment.
5. Inspect the prediction score returned by the model.

The application classifies a review as positive when the predicted score exceeds 0.5 and negative otherwise.

## 💻 Example Prediction

Consider the following review:

**Input:**

```text
This movie was fantastic! The acting was great and the plot was thrilling.
```

The application returns a sentiment label and a numerical prediction score.

Example output format:

```text
Sentiment: Positive
Prediction Score: <model-generated score>
```

The actual score depends on the trained model. The example illustrates the output format and does not represent a verified prediction from a particular run.

## 📈 Model Training and Evaluation

The model is trained using the `model.fit()` method.

Training configuration:

* Maximum epochs: 10
* Batch size: 32
* Validation split: 20%
* Optimizer: Adam
* Loss: Binary cross-entropy
* Metric: Accuracy
* Early stopping: Monitors validation loss with a patience of 5 epochs.

Early stopping helps avoid unnecessary training and restores the model weights from the epoch with the best observed validation loss.

**Note:** The current training notebook saves the trained model but does not report a final evaluation on the held-out test dataset. Test accuracy and other evaluation metrics should be calculated separately before reporting model performance.

## 🔍 Understanding Word Embeddings

The `embedding.ipynb` notebook demonstrates how text can be represented numerically for neural networks.

It explores:

* Converting words into integer representations.
* One-hot-style integer encoding using a vocabulary size.
* Padding sequences to a fixed length.
* Creating an embedding layer using Keras.
* Generating dense vector representations for tokens.

In the main sentiment analysis model, the embedding layer learns token representations during supervised training.

These representations are passed to the Simple RNN, which processes the sequence to generate a sentiment prediction.

## 🔮 Future Improvements

Potential extensions to this project include:

* Evaluate the model on the held-out test dataset.
* Calculate precision, recall, F1-score, and a confusion matrix.
* Improve text preprocessing and unknown-word handling.
* Add explicit masking for padded tokens.
* Compare Simple RNN performance with LSTM and GRU architectures.
* Experiment with different embedding dimensions and recurrent units.
* Visualize training and validation accuracy and loss.
* Improve the Streamlit interface with clearer feedback and probability visualization.
* Explore pretrained word embeddings or Transformer-based sentiment classifiers.
* Deploy the application to a public hosting platform.

## 📚 Learning Outcomes

Through this project, I explored:

* Fundamental Natural Language Processing concepts.
* Integer encoding and word embedding representations.
* Recurrent neural networks and hidden states.
* Binary classification using sigmoid activation.
* Binary cross-entropy loss.
* Optimization using Adam.
* Backpropagation through time.
* Training and validation workflows.
* Early stopping and model persistence.
* Building a simple interactive application with Streamlit.

## 🤝 Contributing

Suggestions, bug reports, and improvements are welcome.

To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Submit a pull request.

## 📄 License

This project is available for educational and learning purposes.

If you intend to distribute or reuse the project, consider adding an appropriate open-source license.

## 👨‍💻 Author

**Manas Mishra**

Artificial Intelligence and Data Science Student
Interested in Deep Learning, Natural Language Processing, and Generative AI.

---

If you find this project useful for learning about recurrent neural networks and sentiment analysis, consider giving the repository a ⭐.
