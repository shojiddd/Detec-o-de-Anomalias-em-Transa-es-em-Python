import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

# 1. Carregamento dos dados
# Certifique-se de que o arquivo creditcard.csv está no caminho correto
df = pd.read_csv("data/creditcard.csv")

# 2. Análise Exploratória e Verificação do Desbalanceamento
print("--- Proporção da Classe Alvo ---")
print(df["Class"].value_counts(normalize=True))

# 3. Feature Engineering & Separação dos Dados
# Exemplo de transformação de escala para a coluna Amount (Valor)
if "Amount" in df.columns:
    df["Amount_log"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Amount"])

X = df.drop(columns=["Class"])
y = df["Class"]

# Divisão estratificada entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Modelo de Baseline: Regressão Logística
print("\n--- Treinando Regressão Logística (Baseline) ---")
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
y_proba_lr = model_lr.predict_proba(X_test)[:, 1]

print("Relatório de Classificação (Regressão Logística):")
print(classification_report(y_test, y_pred_lr))
print("AUC-ROC:", roc_auc_score(y_test, y_proba_lr))

# 5. Tratamento de Dados Desbalanceados com SMOTE
print("\n--- Aplicando SMOTE para Balanceamento ---")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 6. Pipeline com Random Forest
print("\n--- Treinando Random Forest via Pipeline ---")
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42))
])

pipeline_rf.fit(X_train, y_train)

# 7. Ajuste de Threshold no Modelo Random Forest
y_probs_rf = pipeline_rf.predict_proba(X_test)[:, 1]
custom_threshold = 0.3
y_pred_custom_rf = (y_probs_rf >= custom_threshold).astype(int)

print(f"Relatório de Classificação Random Forest (Threshold = {custom_threshold}):")
print(classification_report(y_test, y_pred_custom_rf))
print("AUC-ROC (Random Forest):", roc_auc_score(y_test, y_probs_rf))

# 8. Modelo Avançado: XGBoost
print("\n--- Treinando XGBoost ---")
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

model_xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

model_xgb.fit(X_train, y_train)

y_pred_xgb = model_xgb.predict(X_test)
y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

print("Relatório de Classificação (XGBoost):")
print(classification_report(y_test, y_pred_xgb))
print("AUC-ROC (XGBoost):", roc_auc_score(y_test, y_proba_xgb))
