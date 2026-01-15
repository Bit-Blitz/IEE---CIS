import pandas as pd # type: ignore
import numpy as np
import lightgbm as lgb # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import gc

# 1. Load Data
print("Loading test data...")
test_trans = pd.read_csv('test_transaction.csv')
test_id = pd.read_csv('test_identity.csv')

# 2. FIX: Rename id-01 to id_01, id-02 to id_02, etc.
test_id.columns = [col.replace('-', '_') if 'id-' in col else col for col in test_id.columns]

# 3. Merge
test = pd.merge(test_trans, test_id, on='TransactionID', how='left')

# 4. DEFINE SUBMISSION (Fixes your NameError)
# We pull TransactionID from the test set before dropping columns 
submission = pd.DataFrame({'TransactionID': test['TransactionID']})

del test_trans, test_id; gc.collect()

# 5. Preprocessing
print("Preprocessing...")
test['TransactionAmt_Log'] = np.log1p(test['TransactionAmt'])
test['hour'] = np.floor(test['TransactionDT'] / 3600) % 24
test['day'] = np.floor(test['TransactionDT'] / (3600 * 24)) % 7

# 6. Load Model
model = lgb.Booster(model_file='fraud_model.txt')
model_features = model.feature_name()

# 7. Categorical Encoding
print("Encoding categoricals...")
object_cols = test.select_dtypes(include=['object', 'category']).columns.tolist()
for col in object_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col].astype(str))

# 8. Align Features & Predict
X_test = test[model_features]
print("Generating predictions...")
submission['isFraud'] = model.predict(X_test)

# 9. Save final file
submission.to_csv('Results.csv', index=False)
print("file created")