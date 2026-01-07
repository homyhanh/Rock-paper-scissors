import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib  # để lưu model

# 1. Đọc dữ liệu từ CSV
df = pd.read_csv("data.csv")

# 2. Chia input (X) và label (y)
X = df.drop(columns=["Label"])
y = df["Label"]

# 3. Chia train (60%), valid (20%), test (20%)
# Bước 1: Train + Temp (60/40)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Bước 2: Temp -> Valid/Test (50/50 của 40% = 20% mỗi bên)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Train size:", len(X_train))
print("Valid size:", len(X_valid))
print("Test size :", len(X_test))

# 4. Huấn luyện model (Random Forest) trên train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Đánh giá trên validation
y_valid_pred = model.predict(X_valid)
print("Validation Report:\n", classification_report(y_valid, y_valid_pred))

# 6. Đánh giá trên test (chỉ dùng khi xong model)
y_test_pred = model.predict(X_test)
print("Test Report:\n", classification_report(y_test, y_test_pred))

# 7. Lưu model để dùng sau
joblib.dump((model, X.columns.tolist()), "model.pkl")
