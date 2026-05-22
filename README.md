# House Price Predictor

A simple Streamlit app that predicts house prices using linear regression.

## Features

- Enter house details: area, bedrooms, and age
- Predict estimated price using a trained linear regression model
- Preview the sample training dataset
- View model information

## Requirements

- Python 3.13
- Streamlit
- pandas
- scikit-learn

## Setup

```bash
cd d:/houseprice
.venv\Scripts\activate
python -m pip install streamlit pandas scikit-learn
python -m streamlit run house.py
```

## Usage

Open the browser at `http://localhost:8501` and fill in the input fields. Click **Predict House Price** to see the estimated price.

## Files

- `house.py` — Streamlit application
- `.gitignore` — ignored files and folders
