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

------------------------------------------------------------------------
