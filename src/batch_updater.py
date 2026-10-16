import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score
import joblib
import os

WINDOWS = [1, 7, 15, 30]

def simulate_batch_updates(input_path="data/features_shrinkage.csv"):
    df = pd.read_csv(input_path)
    
    features = [c for c in df.columns if c.endswith("_zscore")]
    
    # Use our pseudo-labels as ground truth target for the behavior classifier
    if "stage_label" not in df.columns:
        from escalation_engine import define_pseudo_labels
        df["stage_label"] = df.apply(define_pseudo_labels, axis=1)
        
    y = df["stage_label"]
    
    os.makedirs("models/versions", exist_ok=True)
    
    results = []
    
    # Assume first 30 days is our initial training base (T=0)
    base_train = df[df["day_index"] < 30]
    if len(base_train) == 0:
        base_train = df[df["day_index"] < 15] # fallback
        
    for w in WINDOWS:
        print(f"\n--- Simulating Update Window: {w} days ---")
        
        y_true_all = []
        y_pred_all = []
        
        # Initial Model (trained on days < 30)
        current_train_df = base_train.copy()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(current_train_df[features].fillna(0), current_train_df["stage_label"])
        
        # Save baseline model version
        joblib.dump(model, f"models/versions/rf_model_w{w}_day30.pkl")
        
        # Iterate from day 30 to end
        for start_day in range(30, df["day_index"].max() + 1, w):
            end_day = start_day + w
            
            # Predict for the upcoming window using the currently trained model
            window_test = df[(df["day_index"] >= start_day) & (df["day_index"] < end_day)]
            
            if len(window_test) > 0:
                X_test = window_test[features].fillna(0)
                y_test = window_test["stage_label"]
                
                preds = model.predict(X_test)
                
                y_true_all.extend(y_test)
                y_pred_all.extend(preds)
            
            # At the end of the window, an air-gap update occurs:
            # We append the window data to the training set and retrain
            current_train_df = pd.concat([current_train_df, window_test])
            
            # Retrain model (simulate airgap update)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(current_train_df[features].fillna(0), current_train_df["stage_label"])
            
            # Save new version
            joblib.dump(model, f"models/versions/rf_model_w{w}_day{end_day}.pkl")
            
        # Calculate metrics for this window size
        # We binarize (Normal/Drift = 0, Suspicious/Exfil = 1) to get a clear recall metric
        y_true_bin = [1 if val >= 2 else 0 for val in y_true_all]
        y_pred_bin = [1 if val >= 2 else 0 for val in y_pred_all]
        
        if len(y_true_bin) > 0:
            rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            
            results.append({
                "Window_Days": w,
                "Recall_HighRisk": rec,
                "F1_HighRisk": f1
            })
            print(f"Window {w} days -> Recall (High Risk): {rec:.3f} | F1: {f1:.3f}")

    res_df = pd.DataFrame(results)
    print("\n--- Air-Gap Batch Update Degradation ---")
    print(res_df.to_string(index=False))
    
    res_df.to_csv("data/batch_update_results.csv", index=False)
    return res_df

if __name__ == "__main__":
    simulate_batch_updates()
