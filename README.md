# Car Price Prediction

This repository contains a machine learning project for predicting car prices based on various vehicle attributes. The project uses a Linear Regression model trained on a car price dataset and includes exploratory data analysis (EDA) performed in a Jupyter notebook. The goal is to provide an accurate price prediction model for automotive applications.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The Car Price Prediction project aims to predict the price of cars based on their technical and categorical attributes, such as engine size, horsepower, fuel type, and car body type. The project involves data preprocessing, feature engineering, and training a Linear Regression model to achieve accurate predictions. The analysis and model training are documented in a Jupyter notebook.

## Features
- **Predictive Model**: Utilizes a Linear Regression model to predict car prices.
- **Exploratory Data Analysis**: Includes EDA to understand feature distributions and relationships.
- **Data Preprocessing**: Handles categorical encoding and numerical feature scaling.
- **Evaluation**: Reports model performance using the R² score.

## Dataset
The dataset used is the [Car Price Assignment Dataset](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction), containing 205 records with 26 features, including:
- **CarName**: Name of the car (e.g., alfa-romero giulia).
- **fueltype**: Gas or diesel.
- **aspiration**: Standard (std) or turbo.
- **doornumber**: Two or four doors.
- **carbody**: Body type (e.g., convertible, sedan, hatchback).
- **drivewheel**: Drive type (fwd, rwd, 4wd).
- **enginelocation**: Front or rear.
- **wheelbase**: Distance between axles (inches).
- **carlength**, **carwidth**, **carheight**: Car dimensions (inches).
- **curbweight**: Vehicle weight (pounds).
- **enginetype**: Engine configuration (e.g., dohc, ohc, ohcv).
- **cylindernumber**: Number of cylinders (e.g., four, six).
- **enginesize**: Engine displacement (cubic inches).
- **fuelsystem**: Fuel system type (e.g., mpfi, 2bbl).
- **boreratio**, **stroke**, **compressionratio**: Engine parameters.
- **horsepower**: Engine power (hp).
- **peakrpm**: Maximum revolutions per minute.
- **citympg**, **highwaympg**: Fuel efficiency (miles per gallon).
- **price**: Target variable, car price (USD).

The dataset has no missing values, as confirmed during EDA. Preprocessing includes one-hot encoding for categorical variables and standardization of numerical features using `StandardScaler`.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/car-price-prediction.git
   cd car-price-prediction
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   Place the `CarPrice_Assignment.csv` file in the `Data/` directory. You can download it from [Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction).

4. **Run the Jupyter notebook**:
   Start Jupyter Notebook and open `Car Price Prediction.ipynb`:
   ```bash
   jupyter notebook
   ```

## Usage
1. Open `Car Price Prediction.ipynb` in Jupyter Notebook.
2. Execute the cells sequentially to:
   - Load and explore the dataset.
   - Perform preprocessing (one-hot encoding and scaling).
   - Split the data into training and testing sets.
   - Train the Linear Regression model.
   - Evaluate the model using the R² score.
3. Review the R² score (approximately 0.8035) to assess model performance on the test set.

To extend the project, you can:
- Experiment with other regression models (e.g., Random Forest, XGBoost).
- Save the trained model using `joblib` for deployment.
- Create a Streamlit app for interactive predictions (see [Future Improvements](#future-improvements)).

## Model Details
- **Algorithm**: Linear Regression.
- **Performance**:
  - R² Score: ~0.8035 (indicating ~80.35% of the variance in car prices is explained by the model).
- **Preprocessing**:
  - Categorical features (e.g., `fueltype`, `carbody`, `drivewheel`) encoded using one-hot encoding.
  - Numerical features (e.g., `wheelbase`, `enginesize`, `horsepower`) scaled using `StandardScaler`.
- **Training**:
  - Split: 80% training, 20% testing.
  - Random State: 42 for reproducibility.

## File Structure
```
car-price-prediction/
├── Data/
│   └── CarPrice_Assignment.csv  # Dataset file
├── Car Price Prediction.ipynb   # Jupyter notebook for EDA and model training
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
```

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- jupyter
- seaborn (for EDA visualizations)
- matplotlib (for EDA visualizations)

Install all dependencies using:
```bash
pip install pandas numpy scikit-learn jupyter seaborn matplotlib
```