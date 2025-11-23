import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


print("Carregando dados...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


y = np.log1p(train['SalePrice'])  # Log para normalizar a distribuição
train = train.drop(['SalePrice'], axis=1)


all_data = pd.concat([train, test], ignore_index=True)


print("Criando features...")
all_data['Age'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['TotalSF'] = all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalQual'] = all_data['OverallQual'] * all_data['GrLivArea']
all_data['TotalBath'] = all_data['FullBath'] + all_data['HalfBath'] * 0.5 + all_data['BsmtFullBath'] + all_data['BsmtHalfBath'] * 0.5
all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)


cols_to_drop = ['Id', 'YrSold', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'OverallQual', 
                'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF', 'GarageArea']
all_data = all_data.drop(cols_to_drop, axis=1, errors='ignore')


print("Tratando faltantes...")
numeric_cols = all_data.select_dtypes(include=[np.number]).columns
categorical_cols = all_data.select_dtypes(include=['object']).columns


imputer_num = SimpleImputer(strategy='median')
all_data[numeric_cols] = imputer_num.fit_transform(all_data[numeric_cols])


all_data[categorical_cols] = all_data[categorical_cols].fillna('Missing')


print("Codificando categóricas...")
all_data = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)


scaler = StandardScaler()
all_data[numeric_cols] = scaler.fit_transform(all_data[numeric_cols])


train_processed = all_data.iloc[:len(train)]
test_processed = all_data.iloc[len(train):]

X = train_processed
X_test = test_processed

print(f"Dimensões: Train {X.shape}, Test {X_test.shape}")


print("Treinando modelo...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)


y_pred_val = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"RMSE na validação (log scale): {rmse_val:.4f}")


print("Prevendo...")
y_pred = model.predict(X_test)
y_pred = np.expm1(y_pred)  # Deslog da predição


submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': y_pred})
submission.to_csv('submission.csv', index=False)
print("Submission salvo como 'submission.csv'!")
print(submission.head())
