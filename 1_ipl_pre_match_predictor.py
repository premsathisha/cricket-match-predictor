import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def preprocess_data():
    '''Load, clean, encode, and train XGBoost model using pre-match features.'''
    folder_path = os.path.join(os.path.dirname(__file__), "ipl_json")
    all_matches = []

    # Iterate through all JSON match files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file)) as f:
                data = json.load(f)
                info = data.get("info", {})
                if info.get("match_type") != "T20":
                    continue

                # Extract core match metadata
                match = {
                    "team1": info["teams"][0],
                    "team2": info["teams"][1],
                    "venue": info.get("venue"),
                    "toss_winner": info.get("toss", {}).get("winner"),
                    "toss_decision": info.get("toss", {}).get("decision"),
                    "winner": info.get("outcome", {}).get("winner"),
                    "super_over": 1 if info.get("outcome", {}).get("method") == "super over" else 0
                }

                innings = data.get("innings", [])

                # Make sure both innings are present
                if len(innings) >= 2 and "team" in innings[0] and "team" in innings[1]:
                    match["batting_first"] = innings[0]["team"]
                    match["batting_second"] = innings[1]["team"]
                else:
                    continue

                all_matches.append(match)

    # Create DataFrame from parsed matches
    df = pd.DataFrame(all_matches)
    df.dropna(subset=["winner"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Encode categorical features using LabelEncoder
    cols_to_encode = [
        "team1", "team2", "venue", "toss_winner",
        "toss_decision", "winner", "batting_first", "batting_second"
    ]
    encoders = {}
    for col in cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Define model features and target
    features = [
        "team1", "team2", "venue", "toss_winner", "toss_decision",
        "batting_first", "batting_second", "super_over"
    ]
    target = "winner"

    # Train-test split and model training
    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )
    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(x_train, y_train)

    print(f"Model accuracy (pre-match): {accuracy_score(y_test, model.predict(x_test)) * 100:.2f}%")
    return model, encoders

def predict_match(model, encoders, match_input):
    '''Predict winner from match input using the trained model.'''
    df_input = pd.DataFrame([{
        key: encoders[key].transform([val])[0] if key in encoders else val
        for key, val in match_input.items()
    }])
    pred_encoded = model.predict(df_input)[0]
    return encoders["winner"].inverse_transform([pred_encoded])[0]

def pre_match_predictor():
    '''Use predictor before toss happens.'''
    model, encoders = preprocess_data()
    match_input = {
        "team1": "Mumbai Indians",
        "team2": "Delhi Capitals",
        "venue": "Wankhede Stadium",
        "toss_winner": "Mumbai Indians",
        "toss_decision": "field",
        "batting_first": "Mumbai Indians",
        "batting_second": "Delhi Capitals",
        "super_over": 0
    }
    result = predict_match(model, encoders, match_input)
    print(f"Predicted winner: {result}")

def main():
    pre_match_predictor()

if __name__ == "__main__":
    main()
