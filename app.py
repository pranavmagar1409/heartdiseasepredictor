import os
import pickle
import warnings

import pandas as pd
from flask import Flask, render_template_string, request

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "knnmodel.pkl")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Column order must match the order used while training the model.
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]

# Label-encoded categories (alphabetical order, as produced by LabelEncoder).
SEX = {"F": 0, "M": 1}
CHEST_PAIN = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
RESTING_ECG = {"LVH": 0, "Normal": 1, "ST": 2}
EXERCISE_ANGINA = {"N": 0, "Y": 1}
ST_SLOPE = {"Down": 0, "Flat": 1, "Up": 2}

# Allowed ranges for the numeric inputs: (min, max, label)
RANGES = {
    "age": (1, 120, "Age"),
    "resting_bp": (50, 250, "Resting blood pressure"),
    "cholesterol": (50, 700, "Cholesterol"),
    "max_hr": (50, 250, "Maximum heart rate"),
    "oldpeak": (-3, 7, "ST depression"),
}


def parse_form(form):
    """Validate the submitted form and return (row_dict, errors)."""
    errors, values = [], {}

    for key, (low, high, label) in RANGES.items():
        raw = form.get(key, "").strip()
        try:
            number = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if not low <= number <= high:
            errors.append(f"{label} must be between {low} and {high}.")
            continue
        values[key] = number

    try:
        values["sex"] = SEX[form.get("sex", "")]
        values["chest_pain"] = CHEST_PAIN[form.get("chest_pain", "")]
        values["fasting_bs"] = 1 if form.get("fasting_bs") == "1" else 0
        values["resting_ecg"] = RESTING_ECG[form.get("resting_ecg", "")]
        values["exercise_angina"] = EXERCISE_ANGINA[form.get("exercise_angina", "")]
        values["st_slope"] = ST_SLOPE[form.get("st_slope", "")]
    except KeyError:
        errors.append("Please choose a value for every option.")

    if errors:
        return None, errors

    row = {
        "Age": values["age"],
        "Sex": values["sex"],
        "ChestPainType": values["chest_pain"],
        "RestingBP": values["resting_bp"],
        "Cholesterol": values["cholesterol"],
        "FastingBS": values["fasting_bs"],
        "RestingECG": values["resting_ecg"],
        "MaxHR": values["max_hr"],
        "ExerciseAngina": values["exercise_angina"],
        "Oldpeak": values["oldpeak"],
        "ST_Slope": values["st_slope"],
    }
    return row, []


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Heart Disease Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #eef3f6;
    --panel: #12303a;
    --panel-ink: #e8f3f1;
    --panel-muted: #9bb9b8;
    --surface: #ffffff;
    --ink: #17252b;
    --muted: #5c707a;
    --line: #d5e0e5;
    --teal: #0f766e;
    --teal-dark: #0b5a54;
    --teal-soft: #dff1ee;
    --high: #b42343;
    --high-soft: #fbe6eb;
    --low: #1f7a4d;
    --low-soft: #e0f3e8;
    --radius: 10px;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Instrument Sans", system-ui, sans-serif;
    background: var(--bg);
    color: var(--ink);
    line-height: 1.5;
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(280px, 380px) 1fr;
    min-height: 100vh;
  }
  /* ---------- left panel ---------- */
  .side {
    background: var(--panel);
    color: var(--panel-ink);
    padding: 48px 36px;
    position: sticky;
    top: 0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    gap: 28px;
    overflow-y: auto;
  }
  .side h1 {
    font-family: "Fraunces", Georgia, serif;
    font-size: 2.3rem;
    line-height: 1.1;
    margin: 0;
    font-weight: 700;
  }
  .side p { margin: 0; color: var(--panel-muted); max-width: 34ch; }
  .pulse { width: 100%; max-width: 280px; height: 56px; }
  .pulse path { stroke: #4fd1c0; stroke-width: 2.5; fill: none; stroke-linecap: round; stroke-linejoin: round; }
  .note { font-size: 0.82rem; margin-top: auto; border-top: 1px solid #254852; padding-top: 16px; color: var(--panel-muted); }

  .result { border-radius: var(--radius); padding: 22px; }
  .result.high { background: var(--high-soft); color: #5e1022; }
  .result.low { background: var(--low-soft); color: #11432b; }
  .result h2 { font-family: "Fraunces", Georgia, serif; font-size: 1.5rem; margin: 0 0 6px; }
  .result .big { font-family: "Fraunces", Georgia, serif; font-size: 3rem; font-weight: 700; line-height: 1; }
  .result p { color: inherit; max-width: none; margin-top: 8px; font-size: 0.92rem; }
  .bar { height: 8px; border-radius: 99px; background: rgba(0,0,0,.12); margin-top: 12px; overflow: hidden; }
  .bar span { display: block; height: 100%; border-radius: 99px; }
  .high .bar span { background: var(--high); }
  .low .bar span { background: var(--low); }

  /* ---------- form ---------- */
  .main { padding: 48px clamp(20px, 5vw, 64px); max-width: 860px; }
  .errors { background: var(--high-soft); color: #5e1022; border-left: 4px solid var(--high); border-radius: 6px; padding: 14px 18px; margin-bottom: 24px; }
  .errors ul { margin: 0; padding-left: 18px; }
  fieldset {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 24px 26px 26px;
    margin: 0 0 22px;
  }
  legend {
    font-family: "Fraunces", Georgia, serif;
    font-weight: 600;
    font-size: 1.2rem;
    padding: 0 10px;
    margin-left: -10px;
    color: var(--teal-dark);
  }
  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px 22px; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  label { font-weight: 600; font-size: 0.92rem; }
  .hint { font-size: 0.8rem; color: var(--muted); font-weight: 400; }
  input, select {
    font: inherit;
    padding: 11px 12px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #f9fbfc;
    color: var(--ink);
    width: 100%;
  }
  input:focus, select:focus { outline: 3px solid var(--teal-soft); border-color: var(--teal); background: #fff; }
  button {
    font: inherit; font-weight: 600; font-size: 1.02rem;
    background: var(--teal); color: #fff;
    border: 0; border-radius: 8px;
    padding: 14px 30px; cursor: pointer;
    transition: background .15s;
  }
  button:hover { background: var(--teal-dark); }
  button:focus-visible { outline: 3px solid var(--teal-soft); outline-offset: 2px; }
  .actions { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  .actions a { color: var(--muted); font-size: 0.92rem; }

  @media (max-width: 860px) {
    .layout { grid-template-columns: 1fr; }
    .side { position: static; height: auto; padding: 32px 22px; }
    .note { margin-top: 0; }
    .main { padding: 28px 18px 48px; }
  }
  @media (max-width: 520px) { .grid { grid-template-columns: 1fr; } }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
<div class="layout">

  <aside class="side">
    <svg class="pulse" viewBox="0 0 280 56" aria-hidden="true">
      <path d="M0 30 H70 L82 8 L98 50 L110 30 H165 L176 16 L188 38 L198 30 H280"/>
    </svg>
    <h1>Heart Disease Predictor</h1>
    <p>Enter a patient's details and a K-nearest-neighbours model compares them with 734 past cases.</p>

    {% if result %}
    <div class="result {{ 'high' if result.high_risk else 'low' }}" role="status">
      <h2>{{ 'Higher risk of heart disease' if result.high_risk else 'Lower risk of heart disease' }}</h2>
      <div class="big">{{ result.probability }}%</div>
      <div class="bar"><span style="width: {{ result.probability }}%"></span></div>
      <p>{{ result.agree }} of the {{ result.neighbors }} most similar cases had heart disease.</p>
    </div>
    {% endif %}

    <p class="note">This is a screening aid, not a diagnosis. Please see a doctor for medical advice.</p>
  </aside>

  <main class="main">
    {% if errors %}
    <div class="errors" role="alert">
      <ul>{% for e in errors %}<li>{{ e }}</li>{% endfor %}</ul>
    </div>
    {% endif %}

    <form method="post" action="/">
      <fieldset>
        <legend>Patient profile</legend>
        <div class="grid">
          <div class="field">
            <label for="age">Age <span class="hint">(years)</span></label>
            <input type="number" id="age" name="age" min="1" max="120" step="1" required value="{{ f.get('age', '') }}">
          </div>
          <div class="field">
            <label for="sex">Sex</label>
            <select id="sex" name="sex" required>
              <option value="" disabled {{ 'selected' if not f.get('sex') }}>Select</option>
              <option value="M" {{ 'selected' if f.get('sex') == 'M' }}>Male</option>
              <option value="F" {{ 'selected' if f.get('sex') == 'F' }}>Female</option>
            </select>
          </div>
          <div class="field" style="grid-column: 1 / -1;">
            <label for="chest_pain">Chest pain type</label>
            <select id="chest_pain" name="chest_pain" required>
              <option value="" disabled {{ 'selected' if not f.get('chest_pain') }}>Select</option>
              <option value="ASY" {{ 'selected' if f.get('chest_pain') == 'ASY' }}>Asymptomatic</option>
              <option value="ATA" {{ 'selected' if f.get('chest_pain') == 'ATA' }}>Atypical angina</option>
              <option value="NAP" {{ 'selected' if f.get('chest_pain') == 'NAP' }}>Non-anginal pain</option>
              <option value="TA" {{ 'selected' if f.get('chest_pain') == 'TA' }}>Typical angina</option>
            </select>
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>Blood tests and vitals</legend>
        <div class="grid">
          <div class="field">
            <label for="resting_bp">Resting blood pressure <span class="hint">(mm Hg)</span></label>
            <input type="number" id="resting_bp" name="resting_bp" min="50" max="250" required value="{{ f.get('resting_bp', '') }}">
          </div>
          <div class="field">
            <label for="cholesterol">Cholesterol <span class="hint">(mg/dl)</span></label>
            <input type="number" id="cholesterol" name="cholesterol" min="50" max="700" required value="{{ f.get('cholesterol', '') }}">
          </div>
          <div class="field">
            <label for="fasting_bs">Fasting blood sugar</label>
            <select id="fasting_bs" name="fasting_bs" required>
              <option value="0" {{ 'selected' if f.get('fasting_bs', '0') == '0' }}>120 mg/dl or below</option>
              <option value="1" {{ 'selected' if f.get('fasting_bs') == '1' }}>Above 120 mg/dl</option>
            </select>
          </div>
          <div class="field">
            <label for="resting_ecg">Resting ECG</label>
            <select id="resting_ecg" name="resting_ecg" required>
              <option value="" disabled {{ 'selected' if not f.get('resting_ecg') }}>Select</option>
              <option value="Normal" {{ 'selected' if f.get('resting_ecg') == 'Normal' }}>Normal</option>
              <option value="ST" {{ 'selected' if f.get('resting_ecg') == 'ST' }}>ST-T wave abnormality</option>
              <option value="LVH" {{ 'selected' if f.get('resting_ecg') == 'LVH' }}>Left ventricular hypertrophy</option>
            </select>
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>Exercise test</legend>
        <div class="grid">
          <div class="field">
            <label for="max_hr">Maximum heart rate <span class="hint">(bpm)</span></label>
            <input type="number" id="max_hr" name="max_hr" min="50" max="250" required value="{{ f.get('max_hr', '') }}">
          </div>
          <div class="field">
            <label for="exercise_angina">Exercise-induced angina</label>
            <select id="exercise_angina" name="exercise_angina" required>
              <option value="" disabled {{ 'selected' if not f.get('exercise_angina') }}>Select</option>
              <option value="N" {{ 'selected' if f.get('exercise_angina') == 'N' }}>No</option>
              <option value="Y" {{ 'selected' if f.get('exercise_angina') == 'Y' }}>Yes</option>
            </select>
          </div>
          <div class="field">
            <label for="oldpeak">ST depression <span class="hint">(Oldpeak)</span></label>
            <input type="number" id="oldpeak" name="oldpeak" min="-3" max="7" step="0.1" required value="{{ f.get('oldpeak', '') }}">
          </div>
          <div class="field">
            <label for="st_slope">ST slope</label>
            <select id="st_slope" name="st_slope" required>
              <option value="" disabled {{ 'selected' if not f.get('st_slope') }}>Select</option>
              <option value="Up" {{ 'selected' if f.get('st_slope') == 'Up' }}>Upsloping</option>
              <option value="Flat" {{ 'selected' if f.get('st_slope') == 'Flat' }}>Flat</option>
              <option value="Down" {{ 'selected' if f.get('st_slope') == 'Down' }}>Downsloping</option>
            </select>
          </div>
        </div>
      </fieldset>

      <div class="actions">
        <button type="submit">Predict</button>
        <a href="/">Clear form</a>
      </div>
    </form>
  </main>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result, errors = None, []
    form_data = request.form if request.method == "POST" else {}

    if request.method == "POST":
        row, errors = parse_form(request.form)
        if row:
            frame = pd.DataFrame([row], columns=FEATURES)
            prediction = int(model.predict(frame)[0])
            probability = float(model.predict_proba(frame)[0][1]) * 100
            result = {
                "high_risk": prediction == 1,
                "probability": round(probability),
                "neighbors": model.n_neighbors,
                "agree": round(probability * model.n_neighbors / 100),
            }

    return render_template_string(PAGE, result=result, errors=errors, f=form_data)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
