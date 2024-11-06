# English to Hindi Machine Translation using LSTM

This project demonstrates English-to-Hindi machine translation using Long Short-Term Memory (LSTM) neural networks. Implemented entirely in a Jupyter Notebook, it leverages deep learning techniques to translate sentences from English to Hindi.

## Project Overview

Machine translation is a key application of NLP, allowing automatic translation between languages. This project uses an LSTM-based sequence-to-sequence model to learn translations from English to Hindi, with data sourced from Kaggle.

## Dataset

- **Source**: The dataset is sourced from Kaggle and contains parallel English-Hindi sentences.
- **Dataset Link**: [Kaggle Dataset for English-Hindi Translation](https://www.kaggle.com/datasets) _(Please replace this link with the actual URL of the dataset you used)_.
- **Data Format**: Each entry in the dataset includes an English sentence and its corresponding Hindi translation.

To access the dataset:

1. Download it from Kaggle.
2. Place the dataset file in the same directory as the notebook.

## Requirements

- **Jupyter Notebook**: Run the project in a Jupyter environment.
- **Python Libraries**:
  - `tensorflow` for deep learning model implementation.
  - `numpy` for numerical operations.
  - `pandas` for data handling.
  - `matplotlib` for visualizations.

You can install the required libraries with:

```bash
pip install tensorflow numpy pandas matplotlib

### How to Run

1. **Open the Notebook**: Launch Jupyter Notebook and open `English_Hindi_Translation.ipynb`.
2. **Run Each Cell**: Execute each cell in the notebook sequentially. The notebook handles data loading, preprocessing, model training, and evaluation.
3. **Translate Sentences**: The notebook includes a section to test the trained model by translating sample English sentences into Hindi.

### Project Structure

- **`English_Hindi_Translation.ipynb`**: The Jupyter Notebook containing the full implementation:
  - **Data Preprocessing**: Tokenizes English and Hindi sentences and pads sequences.
  - **Model Definition**: Builds an encoder-decoder LSTM model.
  - **Training**: Trains the model using the prepared dataset.
  - **Evaluation**: Tests the model’s translation accuracy and demonstrates sample translations.

### Example Output

Example translation from the model:

- **Input (English)**: "Good morning."
- **Output (Hindi)**: "सुप्रभात।"

### Future Improvements

- **Attention Mechanism**: Enhance the model with attention to improve translation accuracy.
- **Bidirectional LSTMs**: Use bidirectional layers for richer context understanding.
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and sequence lengths for better performance.

### References

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [TensorFlow Neural Machine Translation Tutorial](https://www.tensorflow.org/text/tutorials/nmt_with_attention)
- [Kaggle Dataset](https://www.kaggle.com/datasets) *(Replace with the specific dataset link)*
```
