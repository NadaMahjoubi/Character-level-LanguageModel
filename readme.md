# GPT from Scratch: Character-Level Language Model

A small GPT-style language model implemented in PyTorch and trained from scratch on Shakespeare's text. The project explores the fundamentals of Transformer architectures, causal self-attention, next-character prediction, and text generation.

The model is trained using Google Colab's GPU runtime, then exported and loaded locally through a Streamlit web interface.

## Project Overview

This project implements a decoder-only Transformer inspired by the GPT architecture. It learns statistical patterns from a text corpus and generates new text one character at a time.

The goal is to understand the internal mechanisms of GPT models, from data preprocessing and model training to checkpoint management and inference.

### Key Features

* Character-level tokenization
* Token and positional embeddings
* Causal multi-head self-attention
* Transformer blocks with residual connections and layer normalization
* Next-character prediction using cross-entropy loss
* Model optimization with AdamW
* GPU training with Google Colab
* Model checkpoint saving and local loading
* Interactive text generation with Streamlit
* Adjustable generation length and temperature

## Model Architecture

The model follows a GPT-style autoregressive Transformer architecture.

1. **Token embedding:** Converts each character ID into a dense vector.
2. **Positional embedding:** Adds information about character positions.
3. **Transformer blocks:** Process the sequence using causal self-attention and feed-forward networks.
4. **Layer normalization:** Stabilizes the hidden representations.
5. **Language modeling head:** Produces logits for the next character.
6. **Autoregressive generation:** Samples a character and appends it to the context, repeating until the requested length is reached.

### Training Configuration

| Parameter            |                       Value |
| -------------------- | --------------------------: |
| Training corpus      |            Shakespeare text |
| Dataset size         |        1,115,394 characters |
| Vocabulary size      |                          65 |
| Trainable parameters | Approximately 10.79 million |
| Transformer layers   |                           6 |
| Attention heads      |                           6 |
| Embedding dimension  |                         384 |
| Context length       |              256 characters |
| Batch size           |                          64 |
| Training iterations  |                       5,000 |
| Learning rate        |                      0.0003 |
| Dropout              |                         0.2 |
| Optimizer            |                       AdamW |
| Training environment |      Google Colab with CUDA |

*These values describe the reported training run. Ensure they match the configuration stored in the checkpoint before reproducing the experiment.*

## Training Results

The model was trained for 5,000 iterations. The recorded losses were:

|  Step | Training Loss | Validation Loss |
| ----: | ------------: | --------------: |
|     0 |        4.2216 |          4.2311 |
|   500 |        1.7533 |          1.9085 |
| 1,000 |        1.3917 |          1.6176 |
| 2,000 |        1.1883 |          1.5037 |
| 3,000 |        1.0680 |          1.4969 |
| 4,000 |        0.9609 |          1.5151 |
| 4,999 |        0.8563 |          1.5731 |

The loss decreased substantially during training, indicating that the model learned patterns in the training corpus. However, the increase in validation loss during later iterations suggests overfitting.

The generated output demonstrates learned Shakespeare-like vocabulary, dialogue formatting, punctuation, and sentence patterns, although coherence is not guaranteed.

## Project Structure

```text
gpt-dev/
├── input.txt                 # Training corpus
├── model.py                  # GPT architecture
├── tokenizer.py              # Character-level tokenizer
├── train.py                  # Training script
├── app.py                    # Streamlit application
├── checkpoints/
│   └── model.pth             # Trained model checkpoint
└── README.md
```

The model checkpoint is generated during training and contains the learned parameters and the configuration and vocabulary required for inference.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd gpt-dev
```

Replace `<YOUR_REPOSITORY_URL>` with your GitHub repository URL.

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install torch streamlit numpy
```

For GPU training, use a PyTorch installation compatible with your CUDA environment. The provided installation is sufficient for CPU inference.

## Training

The model was trained on Google Colab to take advantage of GPU acceleration.

To reproduce training locally, ensure that `input.txt` is available and that the training configuration in `train.py` matches the intended experiment.

```bash
python train.py
```

The script trains the model and saves its checkpoint to:

```text
checkpoints/model.pth
```

For Colab training, upload the required source files and dataset, execute the training script, and download the generated checkpoint.

## Running the Web Interface

Place your trained checkpoint at `checkpoints/model.pth`, then run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

Enter a prompt such as:

```text
ROMEO:
```

Choose the number of characters to generate and adjust the temperature to experiment with the sampling process.

### Generation Parameters

* **Maximum new tokens:** Controls how many characters are generated after the prompt.
* **Temperature:** Controls the sharpness of the next-character probability distribution. Lower values generally produce more predictable output; higher values increase variation.

## Technologies Used

* Python
* PyTorch
* Transformer architecture
* Google Colab
* CUDA
* Streamlit
* NumPy

## Current Limitations

* Character-level tokenization is less efficient than subword tokenization.
* The model is trained on a limited text domain and is not a general-purpose conversational assistant.
* Generated passages may contain grammatical inconsistencies, repetition, or incoherent transitions.
* Validation loss suggests that additional experimentation with checkpoint selection and regularization may improve generalization.
* Local inference speed depends on the available hardware.

## Future Improvements

* Save the best checkpoint based on validation loss.
* Add automatic evaluation and training-loss visualizations.
* Experiment with temperature, context length, and model size.
* Implement a Byte Pair Encoding (BPE) tokenizer.
* Add text sampling strategies such as top-k sampling.
* Explore fine-tuning a pretrained language model.
* Develop a conversational interface using an instruction-tuned model.

## Learning Objectives

This project provides hands-on experience with:

* Understanding Transformer and GPT internals
* Implementing causal self-attention
* Training neural language models
* Using GPU acceleration for deep learning
* Saving and loading PyTorch checkpoints
* Deploying a machine learning model through a web interface
