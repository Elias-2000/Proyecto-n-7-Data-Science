from flask import Flask, request, jsonify
import pandas as pd
import joblib

ARTIFACT_PATH = "model_logreg_fulloffice_remote_v1.joblib"

artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
keep_map = artifact["keep_map"]
other_label = artifact["other_label"]
threshold = float(artifact["threshold"])
model_version = artifact["model_version"]

app = Flask(__name__)

COLMAP = {
    "Your_Current_Country": "Your Current Country.",
    "Your_Gender": "Your Gender",
    "Factors_influencing_career_aspirations": "Which of the below factors influence the most about your career aspirations ?",
    "Higher_Education_outside_India_self_sponsor": "Would you definitely pursue a Higher Education / Post Graduation outside of India ? If only you have to self sponsor it.",
    "Likely_work_for_one_employer_3_years": "How likely is that you will work for one employer for 3 years or more ?",
    "Work_for_company_mission_not_defined": "Would you work for a company whose mission is not clearly defined and publicly posted.",
    "Work_for_company_mission_misaligned": "How likely would you work for a company whose mission is misaligned with their public actions or even their product ?",
    "Employers_you_would_work_with": "Which of the below Employers would you work with.",
    "Learning_environment": "Which type of learning environment that you are most likely to work in ?",
    "Manager_type": "What type of Manager would you work without looking into your watch ?",
    "Preferred_setup": "Which of the following setup you would like to work ?",
    "Likely_work_for_company_no_social_impact": "How likely would you work for a company whose mission is not bringing social impact ?"
}

REQUIRED_FIELDS = list(COLMAP.keys())

def apply_keep_map_to_other(df: pd.DataFrame, keep_map: dict, other_label: str) -> pd.DataFrame:
    df2 = df.copy()
    for col, allowed in keep_map.items():
        if col in df2.columns:
            allowed_set = set(allowed)
            df2[col] = df2[col].where(df2[col].isin(allowed_set), other_label)
    return df2

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "model_version": model_version,
        "threshold": threshold,
        "endpoints": ["/predict"]
    })

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if payload is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    missing = [k for k in REQUIRED_FIELDS if k not in payload]
    if missing:
        return jsonify({"error": "Missing required fields", "missing": missing}), 400

    data = {COLMAP[k]: payload[k] for k in REQUIRED_FIELDS}
    df_in = pd.DataFrame([data])

    df_in = apply_keep_map_to_other(df_in, keep_map, other_label)

    proba_remote = float(model.predict_proba(df_in)[:, 1][0])
    pred = "REMOTE" if proba_remote >= threshold else "FULL_OFFICE"
    confidence = proba_remote if pred == "REMOTE" else (1.0 - proba_remote)

    return jsonify({
        "prediction": pred,
        "confidence": confidence,
        "model_version": model_version,
        "threshold": threshold,
        "notes": "confidence derived from predict_proba and threshold"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
