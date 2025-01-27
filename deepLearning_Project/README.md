# RSS Value Prediction Using Log Distance and Neural Network

This repository contains a Python implementation for predicting the Received Signal Strength (RSS) values based on log distance using both Linear Regression and a Neural Network model.

### Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Data Description](#data-description)
4. [Methodology](#methodology)
5. [Usage](#usage)
6. [Results](#results)

## Project Overview

The goal of this project is to predict the RSS values using the concept of log distance. The model uses two different approaches:
1. **Linear Regression**: A basic linear model to predict the relationship between log distance and RSS values.
2. **Neural Network**: A deeper neural network model to predict the same relationship with more complexity and flexibility.

The dataset is based on the locations of transmitters and receivers, with corresponding RSS values.

## Prerequisites

To run this project, you need the following libraries:

- `numpy` >= 1.21.0
- `pandas` >= 1.3.0
- `matplotlib` >= 3.4.0
- `tensorflow` >= 2.0.0
- `scikit-learn` >= 0.24.0

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn

