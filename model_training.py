from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline

# 1. Feature Lists
num = [
    'living_space', 'bedroom_number', 'bathroom_number', 'built_year',
    'property_age', 'living_space_ratio', 'total_rooms',
    'city_price_index', 'type_space_interaction',
    'city_sqm_baseline', 'size_vs_city_avg', 'space_per_bed'
]
cat_subset = ['property_type', 'state', 'furnished', 'region', 'is_bangkok']

# 2. Preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_subset),
        ('city_enc', TargetEncoder(), ['city'])
    ]
)

# 3. Pipelines
pipeline_lr = Pipeline([('preprocess', preprocess), ('model', LinearRegression())])
pipeline_dt = Pipeline([('preprocessor', preprocess), ('regressor', DecisionTreeRegressor(max_depth=15, min_samples_leaf=50, random_state=42))])
pipeline_xgb = Pipeline([('preprocessor', preprocess), ('regressor', XGBRegressor(objective='count:poisson', n_estimators=2500, learning_rate=0.02, max_depth=10, random_state=42))])

# 4. Split & Train
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and Predict
pipeline_lr.fit(X_train, y_train)
pipeline_dt.fit(X_train, y_train)
pipeline_xgb.fit(X_train, y_train)

pred_lr = pipeline_lr.predict(X_test)
pred_dt = pipeline_dt.predict(X_test)
pred_xgb = pipeline_xgb.predict(X_test)

# 5. Results & Table
def get_metrics(y_true, y_pred, name):
    return [name, np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)]

results_df = pd.DataFrame([
    get_metrics(y_test, pred_lr, 'Linear Regression'),
    get_metrics(y_test, pred_dt, 'Decision Tree'),
    get_metrics(y_test, pred_xgb, 'XGBoost')
], columns=['Model', 'RMSE', 'MAE', 'R²'])

print(results_df.sort_values(by='R²', ascending=False).round(4))
