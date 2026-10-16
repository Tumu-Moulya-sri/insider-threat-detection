import csv
import json
from pathlib import Path
# pyrefly: ignore [missing-import]
from flask import Flask, jsonify, request

app = Flask(__name__)

# Path to the escalated alerts CSV (relative to project root)
ESCALATED_CSV = Path(__file__).resolve().parents[1] / "data" / "escalated.csv"

def _load_alerts():
    """Read the CSV and return a list of dicts (one per row)."""
    alerts = []
    if not ESCALATED_CSV.exists():
        # Return empty list if the CSV is missing – the API stays alive.
        return alerts
    with ESCALATED_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alerts.append(row)
    return alerts

@app.route("/", methods=["GET"])
def index():
    """Simple health‑check / welcome page."""
    return jsonify({"status": "ok", "message": "Insider‑Threat Alerts API – use /alerts"})

@app.route("/alerts", methods=["GET"])
def get_alerts():
    """Return alerts as JSON.
    Optional query parameter `limit` can restrict the number of rows returned.
    """
    alerts = _load_alerts()
    limit = request.args.get("limit")
    if limit and limit.isdigit():
        alerts = alerts[:int(limit)]
    return jsonify(alerts)

if __name__ == "__main__":
    # Run on localhost:5000 – suitable for a quick demo.
    app.run(host="127.0.0.1", port=5000, debug=False)
import json
from pathlib import Path
# pyrefly: ignore [missing-import]
from flask import Flask, jsonify, request

app = Flask(__name__)

# Path to the escalated alerts CSV (relative to project root)
ESCALATED_CSV = Path(__file__).resolve().parents[1] / "data" / "escalated.csv"

def _load_alerts():
    """Read the CSV and return a list of dicts (one per row)."""
    alerts = []
    if not ESCALATED_CSV.exists():
        # Return empty list if the CSV is missing – the API stays alive.
        return alerts
    with ESCALATED_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alerts.append(row)
    return alerts

@app.route("/", methods=["GET"])
def index():
    """Simple health‑check / welcome page."""
    return jsonify({"status": "ok", "message": "Insider‑Threat Alerts API – use /alerts"})

@app.route("/alerts", methods=["GET"])
def get_alerts():
    """Return alerts as JSON.
    Optional query parameter `limit` can restrict the number of rows returned.
    """
    alerts = _load_alerts()
    limit = request.args.get("limit")
    if limit and limit.isdigit():
        alerts = alerts[:int(limit)]
    return jsonify(alerts)

if __name__ == "__main__":
    # Run on localhost:5000 – suitable for a quick demo.
    app.run(host="127.0.0.1", port=5000, debug=False)
import json
from pathlib import Path

# pyrefly: ignore [missing-import]
from flask import Flask, jsonify

app = Flask(__name__)

# Path to the escalated alerts CSV (relative to project root)
ESCALATED_CSV = Path(__file__).resolve().parents[1] / "data" / "escalated.csv"

def _load_alerts():
    """Read the CSV and return a list of dicts (one per row)."""
    alerts = []
    if not ESCALATED_CSV.exists():
        return alerts
    with ESCALATED_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alerts.append(row)
    return alerts

@app.route("/alerts", methods=["GET"])
def get_alerts():
    """Return the most recent alerts as JSON.
    For simplicity we return the whole file; the caller can slice as needed.
    """
    alerts = _load_alerts()
    return jsonify(alerts)

if __name__ == "__main__":
    # Run on localhost:5000 – suitable for a quick demo.
    app.run(host="127.0.0.1", port=5000, debug=False)
