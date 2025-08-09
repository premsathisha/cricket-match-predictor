import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

SNAPSHOT_BALLS = 60  # first 10 overs of the chase (approx 60 legal balls)

def load_and_preprocess_data(folder_path):
    '''Load and preprocess IPL T20 match data.'''
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

                innings = data.get("innings", [])
                if len(innings) == 2:
                    match["batting_first"] = innings[0].get("team")
                    match["batting_second"] = innings[1].get("team")

                    # First innings
                    match["batting_first_runs"] = sum(
                        ball["runs"]["total"]
                        for over in innings[0].get("overs", [])
                        for ball in over.get("deliveries", [])
                    )
                    match["batting_first_wickets"] = sum(
                        1
                        for over in innings[0].get("overs", [])
                        for ball in over.get("deliveries", [])
                        if "wicket" in ball
                    )

                    # Second innings
                    runs_so_far = 0
                    wickets_so_far = 0
                    balls_seen = 0

                    for over in innings[1].get("overs", []):
                        for ball in over.get("deliveries", []):
                            # count legal deliveries
                            balls_seen += 1
                            runs_so_far += ball["runs"]["total"]
                            if "wicket" in ball:
                                wickets_so_far += 1
                            if balls_seen >= SNAPSHOT_BALLS:
                                break
                        if balls_seen >= SNAPSHOT_BALLS:
                            break

                    match["batting_second_runs"] = runs_so_far
                    match["batting_second_wickets"] = wickets_so_far
                else:
                    continue

                all_matches.append(match)

    df = pd.DataFrame(all_matches)
    df.dropna(subset=["winner"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    cols_to_encode = [
        "team1", "team2", "venue", "toss_winner",
        "toss_decision", "winner", "batting_first", "batting_second"
    ]

    encoders = {}
    for col in cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders

def train_model(df):
    '''Train XGBoost model using mid-match features.'''
    features = [
        "team1", "team2", "venue", "toss_winner", "toss_decision",
        "batting_first", "batting_second",
        "batting_first_runs", "batting_first_wickets",
        "batting_second_runs", "batting_second_wickets"
    ]
    target = "winner"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = XGBClassifier(eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return model

def predict_match(model, encoders, team1, team2, venue, toss_winner, toss_decision,
                  batting_first, batting_second, batting_first_runs, batting_first_wickets,
                  batting_second_runs, batting_second_wickets):
    '''Predict winner using mid-match data.'''
    try:
        input_df = pd.DataFrame([{
            "team1": encoders["team1"].transform([team1])[0],
            "team2": encoders["team2"].transform([team2])[0],
            "venue": encoders["venue"].transform([venue])[0],
            "toss_winner": encoders["toss_winner"].transform([toss_winner])[0],
            "toss_decision": encoders["toss_decision"].transform([toss_decision])[0],
            "batting_first": encoders["batting_first"].transform([batting_first])[0],
            "batting_second": encoders["batting_second"].transform([batting_second])[0],
            "batting_first_runs": batting_first_runs,
            "batting_first_wickets": batting_first_wickets,
            "batting_second_runs": batting_second_runs,
            "batting_second_wickets": batting_second_wickets
        }])
        result_encoded = model.predict(input_df)[0]
        winner = encoders["winner"].inverse_transform([result_encoded])[0]
        print(f"Predicted Winner: {winner}")
    except Exception as e:
        print("Input error:", e)

def main():
    folder = "ipl_json"
    df, encoders = load_and_preprocess_data(folder)
    model = train_model(df)

    # Sample mid-match input
    predict_match(
        model, encoders,
        team1="Royal Challengers Bangalore",
        team2="Chennai Super Kings",
        venue="M Chinnaswamy Stadium",
        toss_winner="Royal Challengers Bangalore",
        toss_decision="bat",
        batting_first="Royal Challengers Bangalore",
        batting_second="Chennai Super Kings",
        batting_first_runs=218,
        batting_first_wickets=5,
        batting_second_runs=100,  # Mid-match score so far
        batting_second_wickets=4
    )

if __name__ == "__main__":
    main()
