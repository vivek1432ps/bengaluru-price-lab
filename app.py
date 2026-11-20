"""Bengaluru Price Lab: local Python web GUI; no external requests."""
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import exp, isfinite
from urllib.parse import parse_qs, urlsplit

import joblib
import pandas as pd

from model import FEATURES, MODEL

CSS = """
*{box-sizing:border-box} html{color-scheme:dark}
body{margin:0;background:#0b1117;color:#e3edf4;font:15px/1.55 system-ui,sans-serif}
main{max-width:1160px;padding:0 32px 32px;margin:auto}
header{display:flex;justify-content:space-between;align-items:center;padding:25px 0;border-bottom:1px solid #2a3b48}
.brand{font-weight:820;letter-spacing:.17em;color:#6bf0c6}
.badge{border:1px solid #365b56;background:#172923;color:#a7ecd4;border-radius:99px;padding:5px 13px;font-size:12px;font-weight:650}
.hero{padding:28px 0 23px}.kicker,.label{color:#84acba;font-size:12px;font-weight:780;letter-spacing:.13em;text-transform:uppercase}
h1{font-size:clamp(32px,5.2vw,61px);letter-spacing:-.058em;line-height:1.06;margin:7px 0 9px;font-weight:800}
p{margin:0} .muted{color:#a1b6c5} .grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.24fr);gap:20px}
.card{padding:24px;border:1px solid #2e424f;background:#111c24;border-radius:19px;min-width:0}
.result{background:radial-gradient(ellipse at 92% 0%,#174139 0,transparent 57%),#11201f;border-color:#356053}
h2{margin:0 0 15px;font-size:22px;letter-spacing:-.03em}.sub{margin-bottom:12px}.fields{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.wide{grid-column:1/-1} label{display:block;font-weight:650;color:#c9dbe5;margin:0 0 4px}
select,input{width:100%;background:#0d171e;border:1px solid #36505e;color:#eef8fc;padding:10px 12px;border-radius:9px;font:inherit}
select:focus,input:focus,button:focus,summary:focus{outline:2px solid #81f4cf;outline-offset:3px}
button{cursor:pointer;width:100%;margin-top:17px;border:0;border-radius:9px;padding:12px;color:#07271f;background:#70ecc0;font:750 15px system-ui}
button:hover{background:#a2ffde}.value{font-size:clamp(32px,5vw,61px);letter-spacing:-.06em;font-weight:830;color:#9effdc;line-height:1.12;margin:7px 0}
.strip{position:relative;height:8px;background:linear-gradient(90deg,#305f56,#87ffce,#305f56);border-radius:8px;margin:16px 0 7px}
.strip:after{position:absolute;left:var(--position);transform:translateX(-50%);content:"";top:-4px;height:16px;width:4px;border-radius:2px;background:#eefcf7}
.ends,.facts{display:flex;justify-content:space-between;gap:10px}.facts{margin:18px 0 12px;padding:13px 0;border-block:1px solid #35504e}.facts b{display:block}
.error{color:#ffd5cb}.warn{margin-top:7px;color:#ffdf9b} small{font-size:12px}.foot{margin-top:17px}.score{display:flex;flex-wrap:wrap;gap:12px 29px}.score b{color:#c7f5eb}
details{margin-top:8px} summary{cursor:pointer;color:#a7e8d4} a{color:#97e9d0} 
@media(max-width:750px){main{padding-inline:16px}.grid{grid-template-columns:1fr}.card{padding:19px}.hero{padding:20px 0}.facts{font-size:13px}}
"""

def rupees(lakh):
    if lakh >= 100:
        return f"₹{lakh/100:,.2f} Cr"
    return f"₹{lakh:,.1f} L"


def validate(form, localities):
    loc = form.get("locality", "").strip()
    if loc not in localities:
        raise ValueError("Choose a supported locality.")
    try:
        sqft, bhk, bath = [float(form.get(k, "")) for k in FEATURES[1:]]
    except ValueError:
        raise ValueError("Enter numbers for area, bedrooms and bathrooms.") from None
    if not all(map(isfinite, (sqft, bhk, bath))) or not (250 <= sqft <= 15000 and
            1 <= bhk <= 10 and bhk.is_integer() and 1 <= bath <= 12 and bath.is_integer()):
        raise ValueError("Use 250–15,000 sq ft, 1–10 whole bedrooms and 1–12 whole bathrooms.")
    if sqft < bhk * 150:
        raise ValueError("Enter at least 150 sq ft per bedroom.")
    return loc, sqft, int(bhk), int(bath)


def render(bundle, query):
    model, width, localities, report = bundle
    default = "whitefield" if "whitefield" in localities else max(localities, key=lambda k: localities[k]["count"])
    form = {"locality": default, "sqft": "1250", "bhk": "2", "bath": "2"}
    form.update({k: v[0] for k,v in query.items() if k in FEATURES and v})
    options = ''.join(f'<option value="{escape(k)}" {"selected" if k == form["locality"] else ""}>{escape(k.title())}</option>' for k in sorted(localities))
    try:
        loc, sqft, bhk, bath = validate(form, localities)
        row = pd.DataFrame([[loc, sqft, bhk, bath]], columns=FEATURES)
        value = float(model.predict(row)[0])
        low, high = value*exp(-width), value*exp(width)
        observed = localities[loc]
        warning = '<p class="warn"><small>Unusual area for this locality; the estimate is less reliable.</small></p>' if not observed["lo"] <= sqft <= observed["hi"] else ""
        result = f'''<div class="kicker">Model estimate / {escape(loc.title())}</div>
        <p class="value">{rupees(value)}</p><p class="muted">Estimated listing price · {bhk} BHK · {sqft:,.0f} sq ft</p>
        <div class="strip" style="--position:{(value-low)/(high-low)*100 if high>low else 50:.1f}%"></div>
        <div class="ends"><b>{rupees(low)}</b><b>{rupees(high)}</b></div>
        <p class="muted"><small>Historical prediction interval · 80% target coverage</small></p>
        <div class="facts"><span><b>₹{value*100000/sqft:,.0f}</b><small class="muted">Estimate per sq ft</small></span>
        <span><b>{rupees(observed['median'])}</b><small class="muted">Locality median in training data</small></span>
        <span><b>{observed['count']}</b><small class="muted">Locality training rows</small></span></div>{warning}
        <small class="muted">Not a current market quote or property valuation.</small>'''
        status = 200
    except ValueError as exc:
        result = f'<h2>Check the property details</h2><p class="error" role="alert">{escape(str(exc))}</p>'
        status = 400
    model_stats = report["test_metrics"][report["model"]]
    comparisons = "".join(f"<li>{escape(name)}: ₹{stats['mae_lakh']:.2f} L MAE, R² {stats['r2']:.3f}</li>" for name,stats in report["test_metrics"].items())
    src = "https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data"
    return status, f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Bengaluru Price Lab</title>
<style>{CSS}</style><body><main>
<header><span class="brand">BLR / PRICE LAB</span><span class="badge">Historical data · ML demo</span></header>
<section class="hero"><div class="kicker">01 / Housing estimator</div><h1>What might this home cost?</h1>
<p class="muted">Explore Bengaluru listing prices by locality, size and layout.</p></section>
<div class="grid"><section class="card"><h2>Property details</h2><form method="get" action="/">
<div class="fields">
<div class="wide"><label for="locality">Locality</label><select id="locality" name="locality">{options}</select></div>
<div class="wide"><label for="sqft">Area (sq ft)</label><input id="sqft" name="sqft" type="number" step="any" min="250" max="15000" required value="{escape(form['sqft'])}"></div>
<div><label for="bhk">Bedrooms (BHK)</label><input id="bhk" name="bhk" type="number" min="1" max="10" required value="{escape(form['bhk'])}"></div>
<div><label for="bath">Bathrooms</label><input id="bath" name="bath" type="number" min="1" max="12" required value="{escape(form['bath'])}"></div>
</div><button type="submit">Estimate listing price →</button></form></section>
<section class="card result" aria-label="Estimate">{result}</section></div>
<section class="card foot"><div class="score"><span><b>{report['clean_rows']:,}</b> cleaned listings</span>
<span><b>{report['localities']}</b> supported localities</span><span><b>₹{model_stats['mae_lakh']:.2f} L</b> held-out MAE</span>
<span><b>{report['test_interval_coverage']:.0%}</b> held-out range coverage</span></div>
<details><summary>Model and data notes</summary><p>Selected model: {report['model']}. MAE is the average absolute prediction error. Held-out comparisons:</p>
<ul>{comparisons}</ul><p><a href="{src}">Dataset source (Kaggle)</a> · Historical asking prices, not sale records. Area definitions vary and listing dates are not reliable. Coverage on new market data is unknown.</p></details>
</section></main></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        if url.path != "/":
            self.send_error(404)
            return
        try:
            query = parse_qs(url.query, keep_blank_values=True, max_num_fields=16)
        except ValueError:
            self.send_error(400)
            return
        status, page = render(self.server.bundle, query)
        body = page.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not MODEL.exists():
        parser.exit(1, "No trained model. Run python model.py first.\n")
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.bundle = joblib.load(MODEL)
        print(f"Open http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()
