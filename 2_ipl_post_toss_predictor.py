import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def preprocess_data(folder_path):
    '''Load and preprocess IPL match data for modeling.'''
    all_matches = []

    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file)) as f:
                data = json.load(f)
                info = data.get("info", {})

                if info.get("match_type") != "T20":
                    continue

                match = {
                    "team1": info["teams"][0],
                    "team2": info["teams"][1],
                    "venue": info.get("venue"),
                    "toss_winner": info.get("toss", {}).get("winner"),
                    "toss_decision": info.get("toss", {}).get("decision"),
                    "winner": info.get("outcome", {}).get("winner"),
                }

                all_matches.append(match)

    df = pd.DataFrame(all_matches)
    df.dropna(subset=["winner"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    cols_to_encode = ["team1", "team2", "venue", "toss_winner", "toss_decision", "winner"]
    encoders = {}
    for col in cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    features = ["team1", "team2", "venue", "toss_winner", "toss_decision"]
    target = "winner"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    print("XGBoost Model Trained for Post-Toss Prediction")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return model, encoders

def predict_match(model, encoders, team1, team2, venue, toss_winner, toss_decision):
    '''Predict winner based on team, venue, and toss outcome.'''
    try:
        input_data = pd.DataFrame([{
            "team1": encoders["team1"].transform([team1])[0],
            "team2": encoders["team2"].transform([team2])[0],
            "venue": encoders["venue"].transform([venue])[0],
            "toss_winner": encoders["toss_winner"].transform([toss_winner])[0],
            "toss_decision": encoders["toss_decision"].transform([toss_decision])[0],
        }])

        prediction = model.predict(input_data)[0]
        predicted_winner = encoders["winner"].inverse_transform([prediction])[0]
        print(f"Predicted Winner: {predicted_winner}")

    except Exception as e:
        print("Input error:", e)

def main():
    folder = "ipl_json"
    model, encoders = preprocess_data(folder)

    # Example: RCB vs CSK match after toss
    predict_match(
        model, encoders,
        team1="Royal Challengers Bangalore",
        team2="Chennai Super Kings",
        venue="M Chinnaswamy Stadium",
        toss_winner="Royal Challengers Bangalore",
        toss_decision="bat"
    )

if __name__ == "__main__":
    main()
