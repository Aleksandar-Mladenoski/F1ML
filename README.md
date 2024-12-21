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

Oddly enough, for this current prototype model I only use the FP1 FP2 FP3 Qualifying or corresponding Sprint sessions. I obstain from using the Race data for obvious reasons. The plan for the future is to also include previous 5 race data, and perhaps even last years race at that GP.
## Model Architecture

The model consists of two primary components:

1. **Transformer Encoder**: Processes sequential lap data to generate feature embeddings for each driver.

2. **Feedforward Neural Network (FNN)**: Takes the transformer-generated embeddings as input and outputs a single scalar value per driver, representing the predicted finishing position.

The `F1MLTransformer` object in the `train.py` script defines this architecture, with already configured hyperparameters such as the number of transformer blocks, attention heads, and FNN layer sizes.
All of the architecture classes are saved in the `transformer.py` file. The transformer is made such that any block can be configured individualy alongside how many features it handles. Multi heat attention is present to parallelize the training process. This all amounts to outputing a single value, that being the 'Lambda' that at the end is sorted and represents the prediction of the model's ranking. In reality the lambda has no real meaning, however once sorted against other lambdas, it represents the ranking of the model.
## Training Procedure

The training process involves:

- **Loss Function**: Utilizing the `LambdaLoss` function from the `loss.py` script, designed for ranking problems to penalize incorrect orderings in the predictions.

- **Optimizer**: Employing the Adam optimizer.

- **Batch Size**: Batch size.

- **Epochs**: Training iterations.

The `train.py` script orchestrates the training loop, including data loading, forward propagation, loss computation, and backpropagation.

## Evaluation Metrics

The model's performance is evaluated using the Normalized Discounted Cumulative Gain (NDCG) at rank 10/20, which measures the quality of the top-10/20 ranking predictions compared to the actual results.

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

   Data is prepared and downloaded through the `data.py` file within this project. It is taken directly from fastf1 and essentially mashed into a massive csv file for easier use. I have done some behind the scenes imputing and nan removal, however I lost the file somewhere and have no idea where it is. Oops. Once I find it I will attach is once again.

4. **Train the Model**:

   Run the training script:

   ```bash
   python train.py
   ```

5. **Evaluate the Model**:

   During the training, at each epoch the test evaluation score is printed to the console. At the last epoch, the evaluation batch race predictions and race results get printed, alongside their corresponding year and grand prix number.

## Results

The model achieves NDCG@10 score of around 80% for the evaluation set with the current setup. If we switch to NDCG@20 we get around 90-91% for the evaluation set.   

These results indicate the model's effectiveness in ranking drivers' finishing positions based on pre-race session data. If desired there is a possibility to print the predicted race result through the function I provided within `train.py`.

## Contributing

A massive thanks to the developer of FASTF1! All of the data has been used from the fastf1 python module! If you love data and AI please consider buying them a coffee! Otherwise, I can thank Formula one for making me a fan and giving me this learning opportunity. 
Contributions are welcome! Please fork the repository and create a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more information and detailed documentation, please refer to the individual scripts and their docstrings within the repository. 
