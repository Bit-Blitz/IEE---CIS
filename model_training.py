import pandas as pd 
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.preprocessing import LabelEncoder 
import gc
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score 

df = pd.read_pickle('train_consolidated.pkl')


print("Training LightGBM...")
X = df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
y = df['isFraud']

# Using 3-fold TimeSeriesSplit for speed (increase to 5 for better accuracy)
tscv = TimeSeriesSplit(n_splits=3)
params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.05, 'num_leaves': 256, 'feature_fraction': 0.8, 'verbosity': -1
}

last_model = None
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    last_model = lgb.train(params, dtrain, valid_sets=[dtrain, dval], 
                           num_boost_round=500, callbacks=[lgb.early_stopping(50)])
    print(f"Fold {fold} Completed.")

# --- 4. SAVING MODEL & METRICS ---
print("Step 3: Saving results...")

# Save the model
last_model.save_model('fraud_model.txt')
print("Model saved as 'fraud_model.txt'")

# Performance Metric (Top 10 Features)
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': last_model.feature_importance(importance_type='gain')
}).sort_values(by='importance', ascending=False)

print("\n--- TOP 10 FEATURES FOR PREDICTING FRAUD ---")
print(importances.head(10))



# 1. Get predictions (probabilities)
# We use the X_val from the last fold of your training
y_probs = last_model.predict(X_val) 

# 2. Convert probabilities to binary (0 or 1) using a 0.5 threshold
y_preds = (y_probs > 0.5).astype(int)

# 3. Calculate Metrics
auc = roc_auc_score(y_val, y_probs)
acc = accuracy_score(y_val, y_preds)

print(f"--- Model Performance ---")
print(f"AUC-ROC Score: {auc:.4f}") # THIS is your most important number
print(f"Accuracy: {acc:.4f}")

# 4. Detailed Report (Precision, Recall, F1)
print("\n--- Detailed Classification Report ---")
print(classification_report(y_val, y_preds))

# 5. Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_val, y_preds)
print(cm)