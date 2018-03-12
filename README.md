# Data Science Portfolio by Stephane Gouloumes

This repository contains all the personal projects in Data Science I've been working on.

## 1. Deep Learning Library

I decided to create my own Deep Learning library to have an in-depth understanding of ANN, more specifically the backpropagation process.
The goal of this project was to create a simple DNN. But it evolved, and then, I wanted to implement the mains steps of a CNN.

It is structured into several classes :
* Model : the main component of the library, in charge of creating the NN, handling forward, backpropagation and making predictions
* Dense : a simple dense layer
* Conv2D : a 2D convulational layer
* Pool : a pooling layer
* Activation : this class computes the activation function of a layer (three functions are available : Sigmoid, ReLu and Softmax)

[Link](https://github.com/stephanegouloumes/data-science-portfolio/blob/master/DL_Library/main.py)

## 2. Prediction of the Success of Kickstart Projects (Classification)

Exploratory data analysis to detect the key success factors of a Kickstarter project.

Then, I used the most important features to predict the success of a project. It uses classification techniques such as Logistic Regression and Random Forest.

[Link](https://github.com/stephanegouloumes/data-science-portfolio/blob/master/01_Kickstarter_Projects_Analysis/main.ipynb)

## 3. Boston Housing Prices (Regression)

[Link](https://github.com/stephanegouloumes/data-science-portfolio/blob/master/02_Boston_Housing_Prices/main.ipynb)

## 4. Titanic (Classification)

[Link](https://github.com/stephanegouloumes/data-science-portfolio/blob/master/03_Titanic_Classification/main.ipynb)
