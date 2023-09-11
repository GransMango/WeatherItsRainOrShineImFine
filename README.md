# Weather Prediction Project

## Overview

This project focuses on weather prediction using a neural network model. It includes data processing, model training, and prediction based on Arduino sensor data.
The project is not to be seen as a reliable tool to plan what you should wear today.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/weather-prediction-project.git
   cd weather-prediction-project
   ```
2. **Install dependencies**
      ```bash
   pip install -r requirements.txt
   ```
3. **Execute main program**
   ```bash
   python main.py
   ```

## Usage

### Data Processing:

- Process and normalize sensor data.
- Prepare data for training.

### Model Training:

- Build and train a neural network model.
- Save the trained model.

### Weather Prediction:

- Load the trained model.
- Make weather predictions based on new sensor data.

## Project Structure

- `Data/`: Data to use to train model.
- `Model/`: Saved trained models.
- `main.py`: Main script for weather prediction.
