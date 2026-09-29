# Generative AI Engineer Task
## LSTM-Based Text Generation

### 1. Objective

The objective of this project is to develop a text generation system using a Long Short-Term Memory (LSTM) neural network. The model learns word sequences from a large text corpus and generates new text from a supplied seed phrase.

### 2. Dataset

The project uses **The Complete Works of William Shakespeare** from Project Gutenberg.

Dataset page:
https://www.gutenberg.org/ebooks/100

Plain text:
https://www.gutenberg.org/files/100/100-0.txt

The source is public-domain material in the United States according to the Project Gutenberg listing.

### 3. Preprocessing

The following preprocessing pipeline is used:

1. Load the text file.
2. Convert all text to lowercase.
3. Remove punctuation and non-alphanumeric symbols.
4. Normalize whitespace.
5. Split the text into word tokens.
6. Build a vocabulary.
7. Convert words into integer IDs.

Special tokens:

- `<PAD>` - padding token
- `<UNK>` - unknown-word token

### 4. Sequence Preparation

A fixed sequence length is used.

For example, with sequence length 5:

```text
Input:
the king is in the

Target:
house
```

The window then shifts one word forward to create another training example.

### 5. Model Architecture

The model consists of:

- Embedding layer
- Two LSTM layers by default
- Dropout
- Dense output layer

The dense layer produces one logit for every vocabulary word.

During training, PyTorch `CrossEntropyLoss` is used with raw logits. During text generation, softmax converts logits into probabilities.

### 6. Training

The dataset is split chronologically into:

- 90% training data
- 10% validation data

The optimizer is Adam.

Early stopping is implemented using validation loss. Whenever validation loss improves, the model checkpoint is saved.

### 7. Text Generation

The generation process is iterative:

```text
Seed text
   ↓
Convert seed words to IDs
   ↓
Take latest sequence_length words
   ↓
Predict next-word probabilities
   ↓
Sample next word
   ↓
Append word to sequence
   ↓
Repeat
```

Generation uses temperature and top-k sampling.

### 8. Sample Seeds

The implementation tests:

```text
to be or not to be
shall i compare thee
the lady doth protest
all the worlds a stage
```

Actual generated text should be copied from:

```text
outputs/sample_generations.txt
```

### 9. Experimental Results

Run the experiments and fill this table with measured results.

| Experiment | Seq. Length | LSTM Layers | Train Loss | Val Loss | Perplexity | Time | Observation |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 20 | 1 | | | | | |
| B | 20 | 2 | | | | | |
| C | 20 | 3 | | | | | |
| D | 10 | 2 | | | | | |
| E | 30 | 2 | | | | | |

### 10. Discussion

The experiments should be compared using actual validation loss, perplexity, training time and generated-text samples.

A deeper LSTM can increase model capacity but also increases computational cost and may be harder to train. Sequence length controls how much previous context is available for predicting the next word. Shorter sequences reduce computation but provide less context, while longer sequences provide more context at increased computational cost.

The final conclusion should be based on the measured experiment results rather than assumptions.

### 11. Challenges and Solutions

**Challenge 1: Large vocabulary**

A very large vocabulary increases the size of the final dense layer.

**Solution:** Limit the vocabulary size and map infrequent words to `<UNK>`.

**Challenge 2: Overfitting**

A language model can memorize frequent patterns from its training corpus.

**Solution:** Use validation data, dropout, checkpointing and early stopping.

**Challenge 3: Repetitive generation**

Sampling only the highest-probability word can make generated text repetitive.

**Solution:** Use temperature and top-k sampling.

**Challenge 4: Long training time**

LSTM training can be computationally expensive.

**Solution:** Start with a reduced corpus for development and then run the larger configuration on a GPU if available.

### 12. Conclusion

This project implements an end-to-end LSTM text generation pipeline. It covers dataset acquisition, text preprocessing, word-level tokenization, vocabulary creation, sequence generation, neural-network training, validation, checkpointing and probabilistic text generation.

The project demonstrates the fundamental mechanics of neural language modeling while keeping the implementation understandable and reproducible.
