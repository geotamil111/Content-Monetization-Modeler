import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')


path = r"C:\Users\Wintel\Downloads\youtube_ad_revenue_dataset.csv"
df = pd.read_csv(path)
print("Dataset loaded:", df.shape)
print(df.head())


print("\nMissing values:\n", df.isnull().sum())
df.drop_duplicates(inplace=True)
print("After cleaning:", df.shape)


df['engagement_rate'] = (df['likes'] + df['comments']) / (df['views'] + 1)
df['revenue_per_view'] = df['ad_revenue_usd'] / (df['views'] + 1)
df.fillna(df.median(numeric_only=True), inplace=True)


for col in ['category', 'device', 'country']:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    joblib.dump(le, f'le_{col}.pkl')


features = ['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 
            'subscribers', 'engagement_rate', 'revenue_per_view', 
            'category_encoded', 'device_encoded', 'country_encoded']

X = df[features]
y = df['ad_revenue_usd']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features, 'features.pkl')


models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

results = {}
best_model = None
best_r2 = -float('inf')

for name, model in models.items():
    if name in ['Linear Regression', 'SVR', 'KNN']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {'R²': round(r2, 4), 'RMSE': round(rmse, 4), 'MAE': round(mae, 4)}
    
    print(f"{name}: R²={r2:.4f}, RMSE={rmse:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_scaler = scaler if name in ['Linear Regression', 'SVR', 'KNN'] else None


joblib.dump(best_model, 'best_model.pkl')
results_df = pd.DataFrame(results).T
results_df.to_pickle('results.pkl')
print("\nBest Model:", max(results, key=results.get))
print(results_df)


plt.figure(figsize=(15,10))
sns.heatmap(df[features + ['ad_revenue_usd']].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='category', y='ad_revenue_usd')
plt.xticks(rotation=45)
plt.title('Revenue by Category')
plt.savefig('revenue_by_category.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Model training completed! Files saved: best_model.pkl, results.pkl")
