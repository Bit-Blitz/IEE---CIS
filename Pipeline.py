import pandas as pd 
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.preprocessing import LabelEncoder 
import gc
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score 

# --- 1. SETTINGS & UTILS ---
def reduce_mem_usage(df):
    for col in df.columns:
        if df[col].dtype != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(df[col].dtype)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                else:
                    df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)
    return df

# --- 2. CONSOLIDATION & SAVING ---
print("Step 1: Merging and cleaning data...")
train_trans = pd.read_csv('train_transaction.csv')
train_id = pd.read_csv('train_identity.csv')
df = pd.merge(train_trans, train_id, on='TransactionID', how='left')
del train_trans, train_id; gc.collect()

# Basic Engineering
df['TransactionAmt_Log'] = np.log1p(df['TransactionAmt'])
df['hour'] = np.floor(df['TransactionDT'] / 3600) % 24

# Drop redundant V-cols (Correlation > 0.95)
v_cols = [c for c in df.columns if c.startswith('V')]
v_corr = df[v_cols].corr()
upper = v_corr.where(np.triu(np.ones(v_corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df = df.drop(columns=to_drop)

# --- 4. Handle Categorical Values (REPLACEMENT) ---
print("Encoding all categorical columns...")

# Identify all columns with 'object' or 'category' type
# This catches M1-M9, id_12-38, DeviceInfo, etc.
object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

for col in object_cols:
    le = LabelEncoder()
    # .astype(str) handles NaNs by converting them to the string "nan" before encoding
    df[col] = le.fit_transform(df[col].astype(str))
    
print(f"Successfully encoded {len(object_cols)} categorical columns.")

# Optimization: Ensure everything is now numeric
df = reduce_mem_usage(df)

# SAVE THE CONSOLIDATED DATA
print("Saving consolidated data to 'train_consolidated.pkl'...")
df.to_pickle('train_consolidated.pkl')