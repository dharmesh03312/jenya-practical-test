# Generative AI Engineer Task - LSTM Text Generation

## 1. Problem Statement

Build a text generator using an LSTM (Long Short-Term Memory) model. The system learns from a large text corpus and predicts the next word from a sequence of previous words.

This implementation follows the supplied task requirements:

- Load a public-domain text dataset.
- Convert text to lowercase.
- Remove punctuation.
- Tokenize the text into word sequences.
- Create input/target pairs where the target is the next token.
- Build an Embedding + LSTM + Dense model.
- Train with a training/validation split.
- Use Adam optimization and cross-entropy loss.
- Use checkpointing and early stopping.
- Generate text from multiple seed inputs.
- Save generated outputs and training information.
- Support experiments with sequence length and number of LSTM layers.

## 2. Dataset

Dataset used:

**The Complete Works of William Shakespeare - Project Gutenberg, eBook #100**

Official page:
https://www.gutenberg.org/ebooks/100

Plain-text download used by the script:
https://www.gutenberg.org/files/100/100-0.txt

The script automatically downloads the dataset into:

`data/shakespeare.txt`

If automatic download fails, manually download the plain-text file and place it at the same location.

## 3. Why Word-Level Tokenization?

The task allows either word-level or character-level tokenization. This solution uses word-level tokenization because the goal is next-token prediction and the generated output is easier to interpret.

Example:

```text
Input:
to be or not to

Target:
be
```

For a sequence length of 5:

```text
Input:  to be or not to
Target: be
```

The sliding window then moves by one word.

## 4. Model Architecture

```text
Input word IDs
      |
      v
Embedding Layer
      |
      v
LSTM Layer 1
      |
      v
LSTM Layer 2
      |
      v
Dropout
      |
      v
Dense Output Layer
      |
      v
Next-word logits
      |
      v
CrossEntropyLoss / Softmax probabilities
```

Default configuration:

- Vocabulary: up to 20,000 words
- Embedding dimension: 128
- Hidden dimension: 256
- LSTM layers: 2
- Dropout: 0.3
- Sequence length: 20
- Batch size: 128
- Optimizer: Adam
- Learning rate: 0.001
- Early stopping patience: 3 epochs

### Important implementation detail

PyTorch's `CrossEntropyLoss` expects raw logits and internally applies the log-softmax operation. Therefore, an explicit `Softmax` layer is not placed in the training model.

For inference, `torch.softmax()` is used to convert logits into probabilities before sampling the next word.

## 5. Installation

Create a virtual environment if desired:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 6. Run the Project

### Quick test

Use a smaller corpus and fewer sequences:

```bash
python lstm_text_generator.py --epochs 2 --max_tokens 15000 --num_sequences 5000
```

### Better demo run

```bash
python lstm_text_generator.py --epochs 15 --max_tokens 100000 --num_sequences 50000 --sequence_length 30
```

### Full run

```bash
python lstm_text_generator.py --epochs 15
```

Training time depends heavily on CPU/GPU, corpus size, vocabulary size and hardware.

## 7. Generated Text

The program automatically tests these seed inputs:

1. `to be or not to be`
2. `shall i compare thee`
3. `the lady doth protest`
4. `all the worlds a stage`

The generated results are saved to:

`outputs/sample_generations.txt`

Do not manually invent the output for the report. Run the model and paste the actual generated text from this file.

## 8. Bonus Experiment

The task asks for experimentation with architecture or sequence length.

### Experiment A - 1 LSTM layer

```bash
python lstm_text_generator.py --num_layers 1 --sequence_length 20 --epochs 10
```

### Experiment B - 2 LSTM layers

```bash
python lstm_text_generator.py --num_layers 2 --sequence_length 20 --epochs 10
```

### Experiment C - 3 LSTM layers

```bash
python lstm_text_generator.py --num_layers 3 --sequence_length 20 --epochs 10
```

### Experiment D - Short sequence

```bash
python lstm_text_generator.py --num_layers 2 --sequence_length 10 --epochs 10
```

### Experiment E - Longer sequence

```bash
python lstm_text_generator.py --num_layers 2 --sequence_length 30 --epochs 10
```

### Report these values for every experiment

| Experiment | Sequence Length | LSTM Layers | Train Loss | Validation Loss | Perplexity | Training Time | Observation |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 20 | 1 | Run | Run | Run | Run | Record actual result |
| B | 20 | 2 | Run | Run | Run | Run | Record actual result |
| C | 20 | 3 | Run | Run | Run | Run | Record actual result |
| D | 10 | 2 | Run | Run | Run | Run | Record actual result |
| E | 30 | 2 | Run | Run | Run | Run | Record actual result |

Use actual measurements rather than claiming that one architecture is better without running the experiments.

## 9. Important Concepts for the Interview

### What is an LSTM?

LSTM is a recurrent neural network architecture designed to model sequential data while reducing the vanishing-gradient problem associated with simple RNNs.

### Why LSTM for text generation?

Text is sequential. The meaning of the next word depends on previous words. LSTM maintains hidden/cell states that can carry information across the sequence.

### What does the Embedding layer do?

It maps each integer word ID to a dense numerical vector. Instead of representing a word as a very large one-hot vector, the model learns a compact representation.

### What is the target?

For each input sequence, the target is the next word.

Example:

```text
Input:  i love machine
Target: learning
```

### Why CrossEntropyLoss?

The model predicts one word from the vocabulary. Cross-entropy measures the difference between the predicted probability distribution and the correct target class.

### Why Adam?

Adam adapts the learning rate for individual parameters and is a common optimizer for neural-network training.

### What is temperature?

Temperature controls randomness during generation.

- Lower temperature: more conservative/repetitive.
- Around 1.0: normal sampling.
- Higher temperature: more diverse/random.

### What is top-k sampling?

Instead of sampling from the entire vocabulary, top-k keeps only the k most likely next words and samples from them.

### What is early stopping?

Training stops when validation loss fails to improve for a configured number of epochs. This helps avoid unnecessary training and can reduce overfitting.

### What is perplexity?

Perplexity is derived from the loss and is commonly used for language-model evaluation. Lower values generally indicate that the model assigns higher probability to the observed validation sequence.

## 10. Expected Project Structure

```text
lstm_text_generation_solution/
│
├── lstm_text_generator.py
├── requirements.txt
├── README.md
│
├── data/
│   └── shakespeare.txt
│
└── outputs/
    ├── best_model.pt
    ├── vocab.json
    ├── config.json
    ├── training_history.json
    └── sample_generations.txt
```

The `data/` and `outputs/` files are created automatically after running the script.

## 11. Limitations

This is a word-level next-token generator, not a modern Transformer-based LLM. With limited training and a modest LSTM, generated text may become repetitive or grammatically inconsistent.

The purpose of this task is to demonstrate the complete Generative AI pipeline:

```text
Dataset
   ↓
Preprocessing
   ↓
Tokenization
   ↓
Vocabulary
   ↓
Sequence creation
   ↓
Embedding
   ↓
LSTM
   ↓
Next-token prediction
   ↓
Validation / checkpointing
   ↓
Sampling
   ↓
Generated text
```

## 12. Submission Checklist

Before submitting:

- [ ] Python script included
- [ ] Dataset link included
- [ ] Lowercase preprocessing included
- [ ] Punctuation removal included
- [ ] Tokenization included
- [ ] Input-output sequence generation included
- [ ] Embedding layer included
- [ ] LSTM layer(s) included
- [ ] Dense output layer included
- [ ] Adam optimizer included
- [ ] Cross-entropy loss included
- [ ] Training/validation split included
- [ ] Early stopping/checkpointing included
- [ ] Multiple seed inputs tested
- [ ] Generated output saved
- [ ] Bonus architecture/sequence experiment performed
- [ ] Actual experiment results included in the report
