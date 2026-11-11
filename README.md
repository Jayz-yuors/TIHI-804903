# TIHI-804903

**Mentor:** Dr Sushopti Gawade\
**College:** Vidyalankar Institute of Technology, Mumbai

**Students:** 1. Jay Keluskar 2. Kedar Sawant

## PHASES

### 1. Phase -1: ML Fundamentals (Days 1 to 4)

**Phase objective:** Build a foundation in machine learning concepts,
mathematical operations, data preparation, regression models, and
optimization.

#### Day 1 -- Foundations of ML & Math Essentials

**Task:** Fetch a crop-yield-versus-rainfall dataset, perform an 80:20
train-test split using pure NumPy, and calculate mean, variance, and
standard deviation.

**Methodology and work completed:** 1. Selected a crop-yield dataset
containing annual rainfall and crop-yield information. 2. Loaded the CSV
dataset into a Pandas DataFrame and inspected its dimensions, columns,
and sample records. 3. Selected `Annual_Rainfall` as the input feature
(X) and `Yield` as the target variable (y). 4. Checked for invalid
numeric values and removed observations containing non-finite values. 5.
Used NumPy to shuffle sample indices and divide the data into 80%
training data and 20% testing data. 6. Calculated the mean, variance,
and standard deviation for the training-set rainfall and yield values
using NumPy. 7. Created a scatter plot to examine the relationship
between annual rainfall and crop yield.

**Google Colab notebook:**\
[Open Day 1 -- Foundations of ML & Math
Essentials](https://colab.research.google.com/drive/1KHwMREO-WmNnfcgA9A5nQyO7FYCA_NHe?usp=sharing)

#### Day 2 -- Regression Models & Optimization

**Task:** Build Linear Regression from scratch using NumPy, implement a
manual Gradient Descent loop, track Mean Squared Error (MSE), and plot
the loss curve using Matplotlib.

**Methodology and work completed:** 1. Continued with the same dataset
and the 80:20 training/testing split prepared on Day 1. 2. Standardized
the rainfall input using the training-set mean and standard deviation.
3. Defined the Linear Regression equation as `y_pred = w*x + b`,
initializing the weight and bias. 4. Implemented Gradient Descent
manually using NumPy. In each epoch, the model generated predictions,
calculated the error and MSE, computed gradients for the weight and
bias, and updated the parameters. 5. Recorded the MSE at every epoch and
plotted the training loss curve using Matplotlib to observe convergence.
6. Generated predictions for the test set and evaluated the model using
test MSE, RMSE, and MAE. 7. Plotted actual versus predicted crop-yield
values and compared the model against a mean-prediction baseline.

**Observation:** The Gradient Descent implementation converged. The
single-feature rainfall model achieved an improvement of approximately
0.0836% over the mean-prediction baseline on the test set. This
indicates that rainfall alone has limited predictive power for this
dataset, even though the regression and optimization implementation ran
successfully.

**Google Colab notebook:**\
[Open Day 2 -- Regression Models &
Optimization](https://colab.research.google.com/drive/1HGVWZYVEHVGtoGw50TWS0g_vF2Lm18u5?usp=sharing)

### Day 3 -- Classification Fundamentals
Task: Train an sklearn LogisticRegression model to perform binary classification and evaluate the predictions using a confusion matrix and ROC-AUC score.

**Methodology and work completed**: 1. Loaded the Crop_recommendation.csv dataset containing soil metrics (Nitrogen, Phosphorus, Potassium, pH) and weather conditions (temperature, humidity, rainfall). 2. Transformed the dataset's multi-class label into a binary target variable (is_rice), assigning 1 for optimal rice conditions and 0 for all other crops. 3. Separated the input features (X) from the binary target variable (y). 4. Divided the dataset into an 80% training set and a 20% testing set using a fixed random state for reproducibility. 5. Applied StandardScaler to standardize the feature values, ensuring metrics with larger ranges (like rainfall) did not disproportionately influence the model over metrics with smaller ranges (like pH). 6. Initialized and trained a Logistic Regression model on the scaled training data. 7. Evaluated the model's performance on the unseen test set by generating probability predictions and calculating the Confusion Matrix and ROC-AUC score.

**Observation**: The model achieved an overall accuracy of 97.73% and an ROC-AUC score of 0.9959. The Confusion Matrix (418 True Negatives, 12 True Positives, 7 False Negatives, 3 False Positives) indicates the model is highly effective at distinguishing suitable rice environments from unsuitable ones, maintaining a conservative threshold that minimizes costly false-positive recommendations.

Google Colab notebook:
[Open Day 3 -- Classification Fundamentals](https://colab.research.google.com/drive/1DeV-kwnKFGoRU7QBBlnjT11cfnNgIHRi?usp=sharing)


### Day 4 -- Model Evaluation & Regularization
Task: Add L2 regularization to the regression model and use GridSearchCV to find the best alpha penalty on tabular crop data to prevent overfitting.   

**Methodology and work completed**:Loaded the master yield_df.csv dataset containing aggregated crop yield, weather, and pesticide data.Isolated the target variable (yield) and removed non-predictive tracking columns like the index and year.Applied One-Hot Encoding to convert categorical text variables (like crop names and regions) into numerical features suitable for regression.Divided the dataset into an 80% training set and a 20% testing set.Standardized the feature values using StandardScaler to ensure the L2 penalty would be applied fairly across metrics with wildly different scales.Initialized a Ridge regression model and defined a parameter grid of candidate alpha values ([0.01, 0.1, 1, 10, 100]).   Executed GridSearchCV with 5-fold cross-validation (cv=5) exclusively on the training data to isolate the optimal hyperparameter without exposing the final test set.   

**Observation**: The GridSearch cross-validation successfully identified 10 as the optimal alpha penalty level. When evaluated on the unseen test data, the finalized model achieved an R-squared score of 0.7493, indicating that the model explains approximately 75% of the variance in crop yield. The Mean Squared Error (MSE) stabilized at 1,818,133,553.04, demonstrating that the L2 regularization effectively balanced the feature weights and mitigated overfitting.

Google Colab notebook:[Open Day 4 -- Model Evaluation & Regularization](https://colab.research.google.com/drive/1IceI8NOJ2CMIAH19hnUNVgVd8F1bU3GX?usp=sharing)


### Day 5 -- Introduction to PyTorch & Tensors
**Task**: Port the linear regression model to PyTorch by converting data into tensors, moving them to the GPU (cuda), and utilizing Autograd (loss.backward()) to handle gradient calculations.

**Methodology and work completed**: Extracted the single-variable rainfall and crop yield columns from the master dataset and applied StandardScaler to normalize the values for neural network optimization.
Initialized a PyTorch environment and verified the active assignment of the hardware accelerator (T4 GPU).
Converted the scaled NumPy arrays into PyTorch Tensors and transferred them to the cuda device to enable parallel processing.
Initialized weight (w) and bias (b) tensors, explicitly setting requires_grad=True to command PyTorch to actively track all mathematical operations applied to the parameters.
Constructed a training loop replacing manual gradient calculus with PyTorch's Autograd engine (loss.backward()).
Updated the model parameters within a torch.no_grad() block to prevent the engine from tracking the updates, and cleared the gradients (zero_()) at the conclusion of each epoch.

**Observation**: The PyTorch Autograd implementation successfully executed on the GPU. The Mean Squared Error (Loss) initialized at 1.0000 and the model's weight parameter converged almost immediately at 0.0010. This mathematically confirms the Day 2 observation: because the standardized data has a variance of 1.0, and rainfall alone lacks a strong linear correlation with yield in this dataset, the Autograd engine correctly determined that a near-zero weight minimizes the error, proving the automated gradient system functions correctly on real-world inputs.

Google Colab notebook:
[Open Day 5 -- Introduction to PyTorch & Tensors](https://colab.research.google.com/drive/1isI-z3ZOhCCJ6AvF5w0_3WljOA4BVhYM?usp=sharing)


### Day 6 -- Multi-Layer Perceptrons (MLPs)
**Task**: Build a PyTorch nn. Module with two hidden layers to classify soil types and track accuracy.   

**Methodology and work completed**: Simulated 1,000 samples of 5-feature tabular soil data and 4-class target labels using NumPy, as per the bootcamp instructions. Converted the synthetic arrays to PyTorch Tensors and pushed them to the T4 GPU (cuda) for accelerated processing.
Constructed a PyTorch nn.
Sequential neural network featuring two hidden layers (32 nodes and 16 nodes) connected by non-linear ReLU activations to learn curved data boundaries.  
Implemented nn.CrossEntropyLoss, passing raw network scores directly to the function so it could internally apply Softmax and compute the multi-class categorization error.   
Optimized the network using the Adam optimizer (lr=0.01) over 100 epochs, actively tracking loss reduction and prediction accuracy.

**Observation**: The Multi-Layer Perceptron architecture executed flawlessly on the GPU. Because the dataset was comprised of purely random simulated numbers, the theoretical baseline accuracy for 4 classes was 25%. The model initialized at 25.70% and successfully optimized its weights to reach 38.90% by Epoch 100. This confirms that the backpropagation loop, hidden layers, and loss functions are mathematically sound, while highlighting that an MLP cannot extract meaningful predictive patterns from purely random noise.

Google Colab notebook:
[Open Day 6 -- Multi-Layer Perceptrons (MLPs)](https://colab.research.google.com/drive/1tCk4xWpmWDzsJL1O07-Dryqf2KdPefNc?usp=sharing)


### Day 7 -- Advanced Training Techniques
**Task**: Enhance the MLP architecture with Batch Normalization and Dropout, and evaluate model generalization by explicitly toggling between training and evaluation modes.   

**Methodology and work completed**: Simulated tabular soil datasets utilizing NumPy, explicitly partitioning the data into a 1000-sample training set and a 200-sample validation set to evaluate out-of-sample generalization.
Constructed an advanced PyTorch nn.Sequential architecture by integrating nn.BatchNorm1d to normalize layer inputs and stabilize the learning process.   Applied nn.
Dropout(0.3) after each non-linear activation to randomly deactivate 30% of neurons during each forward pass, actively mitigating feature co-adaptation and overfitting.   
Configured the Adam optimizer with a learning rate of 1e-3 (0.001).   
Implemented a bifurcated training loop, explicitly activating net.train() for gradient updates and switching to net.eval() paired with torch.no_grad() to compute unbiased validation metrics without Dropout interference.   

**Observation**: The regularization techniques successfully prevented the neural network from memorizing random statistical noise. While the unregularized Day 6 model overfit to reach 38.9% accuracy, the Day 7 architecture correctly identified the lack of correlative patterns in the synthetic data, maintaining a stable validation accuracy near the 25% baseline for 4-class classification. The train and validation loss curves remained tightly coupled (1.4136 vs 1.4092 at Epoch 50), mathematically proving that the Dropout and BatchNorm implementations successfully enforce generalization over memorization.

Google Colab notebook:
[Open Day 7 -- Advanced Training Techniques](https://colab.research.google.com/drive/1GCuV0BxwQMNOozk5tFuBcspfs2oP2uqS?usp=sharing)
