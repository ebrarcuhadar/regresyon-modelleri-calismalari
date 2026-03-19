import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, LassoCV, RANSACRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# ==========================================
# 1. VERİ OKUMA VE ÖN İŞLEME
# ==========================================

df = pd.read_csv("Sleep_Efficiency.csv")

# Zaman değişkenlerini düzenleme
df['Bedtime'] = pd.to_datetime(df['Bedtime'])
df['Wakeup time'] = pd.to_datetime(df['Wakeup time'])
df['Bedtime_Hour'] = df['Bedtime'].dt.hour
df['Wakeup_Hour'] = df['Wakeup time'].dt.hour
df.drop(['ID', 'Bedtime', 'Wakeup time'], axis=1, inplace=True)

# Kategorik Değişkenleri Encode Etme
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Smoking status'] = le.fit_transform(df['Smoking status'])

# Kategorik ve Nümerik Kolonları Belirleme
cat_cols = [col for col in df.columns if df[col].nunique() <= 2]
num_cols = [col for col in df.columns if col not in cat_cols and col != 'Sleep efficiency']

# ==========================================
# 2. EKSİK DEĞER ANALİZİ VE DOLDURMA (KNN IMPUTER)
# ==========================================
scaler_robust = RobustScaler()
df_scaled = pd.DataFrame(scaler_robust.fit_transform(df), columns=df.columns)

knn_imputer = KNNImputer(n_neighbors=5)
df_imputed_scaled = pd.DataFrame(knn_imputer.fit_transform(df_scaled), columns=df.columns)

# Ölçeklenmiş veriyi orijinal ölçeğine geri çevirme
df_imputed = pd.DataFrame(scaler_robust.inverse_transform(df_imputed_scaled), columns=df.columns)

# Aykırı değer analizi (Kaldıraç etkisi görseli için kopyasını tutuyoruz)
df_outlier_raw = df_imputed.copy()

# ==========================================
# 3. AYKIRI DEĞER BASKILAMA (IQR)
# ==========================================
print("\n--- IQR (%10 - %90) Aykırı Değer Baskılama İşlemi ---")
for col in num_cols:
    Q1 = df_imputed[col].quantile(0.10)
    Q3 = df_imputed[col].quantile(0.90)
    IQR = Q3 - Q1
    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR

    df_imputed[col] = np.where(df_imputed[col] < alt_sinir, alt_sinir, df_imputed[col])
    df_imputed[col] = np.where(df_imputed[col] > ust_sinir, ust_sinir, df_imputed[col])

# Yüksek korelasyonlu değişkenin silinmesi
df_final = df_imputed.drop('Light sleep percentage', axis=1)

# ==========================================
# 4. MODEL KURULUMU VE EĞİTİMİ
# ==========================================
X = df_final.drop('Sleep efficiency', axis=1)
y = df_final['Sleep efficiency']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Modeller
ols_model = LinearRegression().fit(X_train_scaled, y_train)
ols_pred = ols_model.predict(X_test_scaled)

lasso_model = LassoCV(cv=5, random_state=42).fit(X_train_scaled, y_train)
lasso_pred = lasso_model.predict(X_test_scaled)

ransac_model = RANSACRegressor(random_state=42).fit(X_train_scaled, y_train)
ransac_pred = ransac_model.predict(X_test_scaled)

# Model Değerlendirme
def evaluate_model(y_true, y_pred, model_name):
    return [model_name, mean_squared_error(y_true, y_pred), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)]

results = [
    evaluate_model(y_test, ols_pred, 'OLS (Linear Regression)'),
    evaluate_model(y_test, lasso_pred, 'Lasso Regression (CV)'),
    evaluate_model(y_test, ransac_pred, 'RANSAC Regression')
]
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'MAE', 'R2_Score']).set_index('Model')
print("\n--- Model Performans Karşılaştırması ---")
print(results_df.round(4))

# ==========================================
# 5. GÖRSELLEŞTİRME VE ANALİZLER
# ==========================================

# 5.1 OLS Üzerindeki Kaldıraç Etkisi (Leverage Effect)
feature = 'Caffeine consumption'
target = 'Sleep efficiency'

ols_kirli = LinearRegression().fit(df_outlier_raw[[feature]], df_outlier_raw[target])
ols_temiz = LinearRegression().fit(df_final[[feature]], df_final[target])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.regplot(x=df_outlier_raw[feature], y=df_outlier_raw[target], ax=axes[0], scatter_kws={'alpha':0.5, 'color':'red'}, line_kws={'color':'black', 'linewidth':3})
axes[0].set_title("Aykırı Değerler VAR (Orijinal Veri) - Doğru sağa çekiliyor", fontweight='bold')
sns.regplot(x=df_final[feature], y=df_final[target], ax=axes[1], scatter_kws={'alpha':0.5, 'color':'green'}, line_kws={'color':'black', 'linewidth':3})
axes[1].set_title("Aykırı Değerler BASKILANDI - Doğru merkeze oturdu", fontweight='bold')
plt.tight_layout()
plt.show()

# 5.2 RANSAC Inlier/Outlier Şovu
temsilci_ozellik = 'Sleep duration'
feature_idx = X.columns.get_loc(temsilci_ozellik)
x_plot = X_train_scaled.iloc[:, feature_idx]
inlier_mask = ransac_model.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

plt.figure(figsize=(10, 6))
plt.scatter(x_plot[inlier_mask], y_train[inlier_mask], color='mediumseagreen', s=70, label='Inliers (Geçerli Ana Kütle)')
plt.scatter(x_plot[outlier_mask], y_train[outlier_mask], color='crimson', marker='X', s=90, label='Outliers (RANSAC Tarafından Dışlananlar)')
plt.title(f"RANSAC Aykırı Değer Yönetimi ({temsilci_ozellik})", fontweight='bold')
plt.legend()
plt.show()

# 5.3 Lasso Geçerlilik (Diagnostic) Analizi
residuals = y_test - lasso_pred
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(x=y_test, y=lasso_pred, ax=axes[0], color='royalblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_title("Gerçek vs Tahmin Edilen")

sns.scatterplot(x=lasso_pred, y=residuals, ax=axes[1], color='darkorange')
axes[1].axhline(0, color='red', linestyle='--', lw=2)
axes[1].set_title("Hata (Residual) Dağılımı")

sns.histplot(residuals, kde=True, ax=axes[2], color='seagreen')
axes[2].set_title("Hataların Normal Dağılımı")
plt.tight_layout()
plt.show()