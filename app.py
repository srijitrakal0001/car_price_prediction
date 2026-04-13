"""
Used Cars Price Prediction — Flask App
Run: python app.py
Open: http://127.0.0.1:5000
"""

import os, re, math, warnings, logging
import joblib, numpy as np, pandas as pd
from flask import Flask, request, jsonify
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "best_xgboost.pk1")
# ── FrequencyEncoder (must match training) ──────────────────────────────────
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.maps = {}
    def fit(self, X, y=None): 
        X = pd.DataFrame(X).copy()
        for col in self.cols:
            self.maps[col] = X[col].value_counts(normalize=True)
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.cols:
            X[col] = X[col].map(self.maps[col]).fillna(0)
        return X

# ── Preprocessing ────────────────────────────────────────────────────────────
def _clean_mileage(val):
    return float(str(val).replace("mi.", "").replace(",", "").strip())

def _extract_engine_capacity(s):
    if not isinstance(s, str): return None
    m = re.search(r"(\d+\.\d+)\s*L(?:iter)?", s, re.I)
    if m: return float(m.group(1))
    m = re.search(r"(\d+)\s+Liter", s, re.I)
    return float(m.group(1)) if m else None

def _extract_horsepower(s):
    if not isinstance(s, str): return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*HP", s, re.I)
    return float(m.group(1)) if m else None

def _is_electric(s):
    if not isinstance(s, str): return False
    return bool(re.search(r"Electric|Dual Motor|Battery Electric|Standard Range Battery", s, re.I))

def preprocess(raw: dict) -> pd.DataFrame:
    r = dict(raw)
    milage = _clean_mileage(r.get("milage", 0)) if isinstance(r.get("milage", 0), str) else float(r.get("milage", 0))
    engine_str = r.get("engine")
    if engine_str:
        ec = _extract_engine_capacity(engine_str)
        if ec is None and _is_electric(engine_str): ec = 0.0
        ec_missing = 1 if ec is None else 0
        ec = ec if ec is not None else 0.0
        hp = _extract_horsepower(engine_str) or 0.0
    else:
        ec         = float(r.get("Engine_Capacity", 0.0) or 0.0)
        ec_missing = int(r.get("Engine_Capacity_Missing", 0))
        hp         = float(r.get("Horse_Power", 0.0) or 0.0)
    hp = max(hp, 0.0)
    accident_occured = "No" if str(r.get("accident", "None reported")) == "None reported" else "Yes"
    clean_title = str(r.get("clean_title", "Yes"))
    if clean_title.lower() in ("nan", "none", ""): clean_title = "unknown"
    fuel_type = r.get("fuel_type") or ("Electric" if engine_str and _is_electric(engine_str) else "Gasoline")
    return pd.DataFrame([{
        "model_year": int(r.get("model_year", 2020)), "milage": milage,
        "Engine_Capacity": ec, "Horse_Power": hp, "Engine_Capacity_Missing": ec_missing,
        "brand": str(r.get("brand", "")), "model": str(r.get("model", "")),
        "transmission": str(r.get("transmission", "")),
        "ext_col": str(r.get("ext_col", "")), "int_col": str(r.get("int_col", "")),
        "fuel_type": str(fuel_type), "clean_title": clean_title, "Accident_occured": accident_occured,
    }])

# ── Model loader ─────────────────────────────────────────────────────────────
_model = None
def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'")
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── HTML UI embedded ──────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Used Car Price Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#0a0a0f;--surface:#111118;--surface2:#1a1a24;--border:#2a2a3a;--accent:#e8ff00;--text:#f0f0f8;--muted:#6b6b88;--error:#ff4d6d;--success:#00e5a0;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(232,255,0,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(232,255,0,.03) 1px,transparent 1px);background-size:48px 48px;pointer-events:none;z-index:0;}
body::after{content:'';position:fixed;top:-200px;right:-200px;width:600px;height:600px;background:radial-gradient(circle,rgba(232,255,0,.06) 0%,transparent 70%);pointer-events:none;z-index:0;}
.wrapper{position:relative;z-index:1;max-width:860px;margin:0 auto;padding:48px 24px 80px;}
header{margin-bottom:52px;}
.badge{display:inline-flex;align-items:center;gap:6px;background:rgba(232,255,0,.08);border:1px solid rgba(232,255,0,.2);color:var(--accent);font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;padding:5px 12px;border-radius:99px;margin-bottom:20px;}
.badge::before{content:'';width:6px;height:6px;background:var(--accent);border-radius:50%;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.7);}}
h1{font-family:'Bebas Neue',sans-serif;font-size:clamp(52px,8vw,96px);line-height:.92;letter-spacing:.02em;}
h1 span{color:var(--accent);}
.subtitle{margin-top:16px;color:var(--muted);font-size:15px;font-weight:300;max-width:460px;line-height:1.6;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:40px;position:relative;overflow:hidden;}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(232,255,0,.4),transparent);}
.section-label{font-size:10px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid var(--border);}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:28px;}
.grid.three{grid-template-columns:1fr 1fr 1fr;}
.grid.full{grid-template-columns:1fr;}
@media(max-width:600px){.grid,.grid.three{grid-template-columns:1fr;}.card{padding:24px;}}
.field{display:flex;flex-direction:column;gap:7px;}
label{font-size:12px;font-weight:500;color:var(--muted);letter-spacing:.04em;}
label .req{color:var(--accent);margin-left:2px;}
input,select{background:var(--surface2);border:1px solid var(--border);border-radius:10px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:14px;padding:11px 14px;outline:none;transition:border-color .2s,box-shadow .2s;width:100%;-webkit-appearance:none;}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(232,255,0,.08);}
input::placeholder{color:var(--muted);}
select option{background:#1a1a24;}
.divider{height:1px;background:var(--border);margin:28px 0;}
.btn-predict{width:100%;padding:16px;background:var(--accent);color:#0a0a0f;border:none;border-radius:12px;font-family:'Bebas Neue',sans-serif;font-size:20px;letter-spacing:.08em;cursor:pointer;transition:transform .15s,box-shadow .15s,opacity .15s;display:flex;align-items:center;justify-content:center;gap:10px;margin-top:8px;}
.btn-predict:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(232,255,0,.25);}
.btn-predict:active{transform:translateY(0);}
.btn-predict:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none;}
.spinner{width:18px;height:18px;border:2px solid rgba(0,0,0,.3);border-top-color:#0a0a0f;border-radius:50%;animation:spin .7s linear infinite;display:none;}
.loading .spinner{display:block;}.loading .btn-text{display:none;}
@keyframes spin{to{transform:rotate(360deg);}}
#result{margin-top:28px;border-radius:16px;overflow:hidden;display:none;animation:slideUp .35s cubic-bezier(.16,1,.3,1);}
@keyframes slideUp{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}
.result-success{background:linear-gradient(135deg,rgba(0,229,160,.08),rgba(0,212,255,.06));border:1px solid rgba(0,229,160,.25);padding:28px 32px;}
.result-error{background:rgba(255,77,109,.08);border:1px solid rgba(255,77,109,.25);padding:20px 24px;display:flex;align-items:center;gap:12px;}
.result-error-msg{font-size:14px;color:var(--error);}
.result-label{font-size:11px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--success);margin-bottom:10px;}
.result-price{font-family:'Bebas Neue',sans-serif;font-size:clamp(48px,8vw,72px);line-height:1;}
.result-price span{color:var(--accent);}
.result-note{margin-top:10px;font-size:13px;color:var(--muted);}
.result-details{margin-top:18px;padding-top:18px;border-top:1px solid rgba(255,255,255,.07);display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}
.detail-item label{color:var(--muted);font-size:11px;}
.detail-item p{font-size:14px;font-weight:500;margin-top:3px;}
footer{margin-top:40px;text-align:center;font-size:12px;color:var(--muted);}
</style>
</head>
<body>
<div class="wrapper">
  <header>
    <div class="badge">XGBoost Model &middot; R&sup2; 0.869</div>
    <h1>USED CAR<br><span>PRICE</span><br>PREDICTOR</h1>
    <p class="subtitle">Enter car details below — the trained XGBoost model will estimate the market price instantly.</p>
  </header>

  <div class="card">
    <div class="section-label">Car Identity</div>
    <div class="grid">
      <div class="field"><label>Brand <span class="req">*</span></label><input id="brand" type="text" placeholder="e.g. Toyota"/></div>
      <div class="field"><label>Model <span class="req">*</span></label><input id="model" type="text" placeholder="e.g. Camry XSE"/></div>
    </div>
    <div class="grid three">
      <div class="field"><label>Year <span class="req">*</span></label><input id="model_year" type="number" placeholder="2020" min="1990" max="2025"/></div>
      <div class="field"><label>Mileage <span class="req">*</span></label><input id="milage" type="text" placeholder="34,742 mi."/></div>
      <div class="field"><label>Fuel Type</label>
        <select id="fuel_type">
          <option>Gasoline</option><option>Diesel</option><option>Electric</option>
          <option>Hybrid</option><option>E85 Flex Fuel</option><option>Plug-In Hybrid</option>
        </select>
      </div>
    </div>

    <div class="divider"></div>
    <div class="section-label">Specs &amp; Appearance</div>
    <div class="grid">
      <div class="field"><label>Transmission <span class="req">*</span></label><input id="transmission" type="text" placeholder="e.g. 8-Speed Automatic"/></div>
      <div class="field"><label>Engine String <em style="font-style:normal;opacity:.5">(optional)</em></label><input id="engine" type="text" placeholder="e.g. 301.0HP 3.5L V6 Gasoline Fuel"/></div>
    </div>
    <div class="grid three">
      <div class="field"><label>Exterior Color <span class="req">*</span></label><input id="ext_col" type="text" placeholder="e.g. Black"/></div>
      <div class="field"><label>Interior Color <span class="req">*</span></label><input id="int_col" type="text" placeholder="e.g. Gray"/></div>
      <div class="field"><label>Clean Title</label>
        <select id="clean_title"><option value="Yes">Yes</option><option value="No">No</option><option value="unknown">Unknown</option></select>
      </div>
    </div>

    <div class="divider"></div>
    <div class="section-label">Condition</div>
    <div class="grid full">
      <div class="field"><label>Accident History</label>
        <select id="accident">
          <option value="None reported">None reported</option>
          <option value="At least 1 accident or damage reported">At least 1 accident reported</option>
        </select>
      </div>
    </div>

    <button class="btn-predict" id="predictBtn" onclick="predict()">
      <div class="spinner"></div>
      <span class="btn-text">&#9889; PREDICT PRICE</span>
    </button>

    <div id="result"></div>
  </div>

  <footer>Served by Flask &middot; XGBoost + FrequencyEncoder pipeline &middot; Trained on used_cars.csv</footer>
</div>

<script>
async function predict() {
  const btn = document.getElementById('predictBtn');
  const resultEl = document.getElementById('result');
  const fields = {
    brand:        document.getElementById('brand').value.trim(),
    model:        document.getElementById('model').value.trim(),
    model_year:   parseInt(document.getElementById('model_year').value),
    milage:       document.getElementById('milage').value.trim(),
    transmission: document.getElementById('transmission').value.trim(),
    ext_col:      document.getElementById('ext_col').value.trim(),
    int_col:      document.getElementById('int_col').value.trim(),
    fuel_type:    document.getElementById('fuel_type').value,
    engine:       document.getElementById('engine').value.trim() || undefined,
    accident:     document.getElementById('accident').value,
    clean_title:  document.getElementById('clean_title').value,
  };
  const required = ['brand','model','model_year','milage','transmission','ext_col','int_col'];
  const missing  = required.filter(k => !fields[k] || fields[k]==='NaN');
  if (missing.length) { showError('Please fill in: ' + missing.join(', ')); return; }

  btn.disabled = true; btn.classList.add('loading'); resultEl.style.display='none';
  try {
    const res  = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(fields)});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    showSuccess(data.predicted_price, fields);
  } catch(err) {
    showError(err.message);
  } finally {
    btn.disabled = false; btn.classList.remove('loading');
  }
}

function showSuccess(price, f) {
  const el  = document.getElementById('result');
  const fmt = new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0});
  el.className='result-success'; el.style.display='block';
  el.innerHTML = `
    <div class="result-label">&#10022; Estimated Market Price</div>
    <div class="result-price"><span>${fmt.format(price)}</span></div>
    <p class="result-note">XGBoost prediction &middot; log-transformed target &middot; R&sup2; 0.869 on test set</p>
    <div class="result-details">
      <div class="detail-item"><label>Vehicle</label><p>${f.brand} ${f.model}</p></div>
      <div class="detail-item"><label>Year</label><p>${f.model_year}</p></div>
      <div class="detail-item"><label>Mileage</label><p>${f.milage}</p></div>
    </div>`;
}

function showError(msg) {
  const el = document.getElementById('result');
  el.className='result-error'; el.style.display='flex';
  el.innerHTML = `<div style="font-size:20px">&#9888;</div><div class="result-error-msg">${msg}</div>`;
}

document.addEventListener('keydown', e => { if(e.key==='Enter' && e.target.tagName!=='TEXTAREA') predict(); });
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML

@app.get("/health")
def health():
    try:
        load_model()
        return jsonify({"status": "ok", "model": MODEL_PATH}), 200
    except FileNotFoundError as e:
        return jsonify({"status": "error", "detail": str(e)}), 503

@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400
    required = ["brand", "model", "model_year", "milage", "transmission", "ext_col", "int_col"]
    missing  = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 422
    try:
        model = load_model()
        price = float(model.predict(preprocess(body))[0])
        return jsonify({"predicted_price": round(price, 2), "currency": "USD"}), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.post("/predict/batch")
def predict_batch():
    body = request.get_json(silent=True)
    if not body or "cars" not in body:
        return jsonify({"error": 'Body must have a "cars" list.'}), 400
    cars = body["cars"]
    if not isinstance(cars, list) or not cars:
        return jsonify({"error": '"cars" must be a non-empty list.'}), 422
    required = ["brand","model","model_year","milage","transmission","ext_col","int_col"]
    errors, frames = [], []
    for i, car in enumerate(cars):
        miss = [f for f in required if f not in car]
        if miss: errors.append({"index": i, "error": f"Missing: {miss}"})
        else:
            try: frames.append((i, preprocess(car)))
            except Exception as e: errors.append({"index": i, "error": str(e)})
    if not frames:
        return jsonify({"error": "No valid cars.", "details": errors}), 422
    try:
        model    = load_model()
        prices   = model.predict(pd.concat([f for _, f in frames], ignore_index=True)).tolist()
        response = {"predictions": [{"index": idx, "predicted_price": round(prices[pos], 2), "currency": "USD"} for pos, (idx, _) in enumerate(frames)]}
        if errors: response["skipped"] = errors
        return jsonify(response), 200
    except Exception as e:
        logger.exception("Batch failed")
        return jsonify({"error": str(e)}), 500

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting server on http://127.0.0.1:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
