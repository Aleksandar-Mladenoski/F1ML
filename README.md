# F1ML: Formula 1 Machine Learning Project

Welcome to the F1ML project, an innovative approach to predicting Formula 1 race outcomes using advanced machine learning techniques. This project leverages transformer architectures to process sequential racing data, generating feature embeddings that inform a feedforward neural network (FNN) for ranking predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training Procedure](#training-procedure)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The F1ML project aims to predict the finishing positions of drivers in Formula 1 races by analyzing data from practice sessions (FP1, FP2, FP3) and qualifying sessions. The model employs a transformer-based architecture to capture the sequential nature of lap times and other relevant metrics, producing feature embeddings that are subsequently processed by a feedforward neural network to generate ranking predictions.

## Data Preparation

The dataset comprises lap times and final race results from multiple seasons. The data is preprocessed to handle missing values, normalize features, and split into training and testing sets. This is done in the `preprocess.py` file, in which categorical features are one hot encoded and parsed as float, alongside all of the other neccesary preprocessing steps. The `dataset.py` script defines the `FSDataset` class, which loads the data, and the `collate_fn` function, which prepares batches for training. It also incudes a splitting function for splitting the data into training and test data. The splitting is done chronologically as the data is time based.

## Model Architecture

The model consists of two primary components:

1. **Transformer Encoder**: Processes sequential lap data to generate feature embeddings for each driver.

2. **Feedforward Neural Network (FNN)**: Takes the transformer-generated embeddings as input and outputs a single scalar value per driver, representing the predicted finishing position.

The `F1MLTransformer` object in the `train.py` script defines this architecture, with already configured hyperparameters such as the number of transformer blocks, attention heads, and FNN layer sizes.
All of the architecture classes are saved in the `transformer.py` file. The transformer is made such that any block can be configured individualy alongside how many features it handles. Multi heat attention is present to parallelize the training process.
## Training Procedure

The training process involves:

- **Loss Function**: Utilizing the `LambdaLoss` function from the `loss.py` script, designed for ranking problems to penalize incorrect orderings in the predictions.

- **Optimizer**: Employing the Adam optimizer.

- **Batch Size**: Batch size.

- **Epochs**: Training iterations.

The `train.py` script orchestrates the training loop, including data loading, forward propagation, loss computation, and backpropagation.

## Evaluation Metrics

The model's performance is evaluated using the Normalized Discounted Cumulative Gain (NDCG) at rank 10, which measures the quality of the top-10 ranking predictions compared to the actual results.

## Usage

To run the project:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Aleksandar-Mladenoski/F1ML.git
   cd F1ML
   ```

2. **Install Dependencies**:

   Ensure you have Python 3.8+ and install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data**:

   Place your preprocessed CSV files in the `F1Data` directory. Use the `f1data_train_test_split` function in `dataset.py` to split the data into training and testing sets.

4. **Train the Model**:

   Run the training script:

   ```bash
   python train.py
   ```

5. **Evaluate the Model**:

   During the training, at each epoch the test evaluation score is printed to the console. 

## Results

The model achieved the following NDCG@10 scores during training:

- **Epoch 13**:
  - Train NDCG@10: 0.8106
  - Eval NDCG@10: 0.7824

- **Epoch 14**:
  - Train NDCG@10: 0.8113
  - Eval NDCG@10: 0.7880

These results indicate the model's effectiveness in ranking drivers' finishing positions based on pre-race session data.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your proposed changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more information and detailed documentation, please refer to the individual scripts and their docstrings within the repository. 
