# 🐐 Goat Weight Predictor

A Streamlit web app that trains on your own goat herd data. It estimates current weight from body measurements, forecasts future weight, flags underweight and overweight goats early, and suggests a daily feed plan.

## Features

- **Data tab**: upload CSV or Excel, or use a generated sample dataset. Shows a 3D herd plot (age, body measure, weight) and an age vs weight chart by breed.
- **Train tab**: fits three things in one click:
  - Breed growth curve (Gompertz)
  - Weight estimator from body measures (gradient boosting)
  - Future-weight model (gradient boosting)
- **Predict one goat**: enter breed, sex, age, and either current weight or body measures. Get status now, status in N months (1 to 12), a gauge chart, a growth trajectory, and a feed suggestion.
- **Predict many goats**: upload a herd file, forecast all goats at once, and download the predictions as CSV.
- **Model export**: download the trained model as a `.joblib` file.

## How it works

**1. Reference growth curve.** Each breed and sex gets a Gompertz curve. Built-in starting values exist for Boer, Sirohi, Shannan, Khari, and Black Bengal. When a breed and sex group has 20+ rows and 6+ distinct ages, the curve is refitted on your own data.

**2. Herd status.** Each goat is compared with the reference weight for its breed, sex, and age:

| Weight / reference | Status |
| --- | --- |
| below 0.85 | Underweight |
| 0.85 to 1.15 | Normal |
| above 1.15 | Overweight |

**3. Weight estimator.** Predicts weight from height, chest girth, body length, breed, sex, and optionally age. Needs at least 30 rows with measurements. Metrics (MAE and R²) come from a holdout split, with whole goats held out when `goat_id` exists.

**4. Future weight.** If your data has repeat measurements per goat (60+ before/after pairs from 15+ goats), a gradient boosting model learns the growth ratio. Otherwise the app falls back to the breed reference curve scaled to each goat's current condition. The learned model is compared against the reference-curve baseline.

**5. Feeding suggestion.** Rule-of-thumb daily amounts of concentrate, dry roughage, green fodder, and mineral mix, based on weight, age, and status. Kids under 3 months get no plan (milk and creep feed, ask a vet).

> The feed plan is a general guide only. Confirm with a vet or animal nutritionist.

## Project structure

```
.
├── app.py                # Streamlit UI (tabs, charts, 3D hero)
├── goat_ml.py            # Cleaning, growth curves, models, feeding plan
├── generate_dataset.py   # Sample dataset generator
├── sample_goat_data.csv  # Example data
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/subedikaruna/weight-prediction-.git
cd weight-prediction-
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Data format

| Column | Required | Notes |
| --- | --- | --- |
| `breed` | Yes | e.g. Boer, Sirohi, Shannan, Khari, Black Bengal |
| `sex` | Yes | `male`/`female` (also accepts `m`, `f`, `buck`, `doe`) |
| `age_months` | Yes | Number |
| `weight_kg` | Yes | Must be above 0 |
| `goat_id` | No | Repeat across dates to enable the learned future-weight model |
| `height_cm` | No | Enables the weight estimator |
| `chest_girth_cm` | No | Enables the weight estimator |
| `body_length_cm` | No | Enables the weight estimator |

Rows with missing required values, unknown sex, or non-positive weight are dropped. A blank template can be downloaded from the app sidebar. At least 30 valid rows are needed to train.

Example:

```csv
goat_id,breed,sex,age_months,height_cm,chest_girth_cm,body_length_cm,weight_kg
G001,Sirohi,female,6,52.0,58.0,50.0,16.5
G001,Sirohi,female,9,57.5,64.0,55.0,21.0
G002,Boer,male,8,60.0,66.0,58.0,27.0
```

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, SciPy, Plotly, joblib, openpyxl, three.js (hero animation).

## Limitations

- Built-in breed curves are approximate starting values. Accuracy improves with your own data.
- The future-weight model needs repeat measurements per goat. Without them, forecasts follow the breed curve.
- Feed amounts are general guidance, not a formulated ration.

## License

Not specified yet. Add a license file if you plan to share or reuse this project.
