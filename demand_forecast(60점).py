import pandas as pd
import numpy as np
import sqlite3
import os
import random
import lightgbm as lgb
import gc
from tqdm import tqdm
import joblib

# ======================================================================================
# 0. 환경 설정
# ======================================================================================
np.random.seed(42)
random.seed(42)
DATA_DIR = "./data"
MODEL_PATH = 'demand_model_final_upgrade.pkl'
SUBMISSION_TEMPLATE_PATH = os.path.join(DATA_DIR, 'forecast_submission_template.csv')

# ======================================================================================
# 1. 보조 데이터 사전 로딩 함수
# ======================================================================================
def load_auxiliary_data():
    """성능 향상을 위해 모든 보조 데이터를 미리 로드합니다."""
    print("보조 데이터 로딩 중...")
    aux_data = {}
    files_to_load = {
        "sku_meta": "sku_meta.csv", "price_promo": "price_promo_train.csv",
        "site_candidates": "site_candidates.csv", "calendar": "calendar.csv",
        "holidays": "holiday_lookup.csv"
    }
    for key, filename in files_to_load.items():
        aux_data[key] = pd.read_csv(os.path.join(DATA_DIR, filename))

    for key in ['sku_meta', 'price_promo', 'calendar', 'holidays']:
        date_col = 'launch_date' if key == 'sku_meta' else 'date'
        aux_data[key][date_col] = pd.to_datetime(aux_data[key][date_col])
    
    aux_data['holidays']['is_holiday'] = 1
    aux_data['holidays'] = aux_data['holidays'][['date', 'country', 'is_holiday']].drop_duplicates()
    return aux_data

# ======================================================================================
# 2. 피처 엔지니어링 함수 (이벤트 정보 추가)
# ======================================================================================
def create_rich_features(df, aux_data, event_periods):
    df = df.sort_values('date').copy()
    
    # 1. 보조 데이터 병합
    df = pd.merge(df, aux_data['sku_meta'], on='sku', how='left')
    city = df['city'].iloc[0]
    country = aux_data['site_candidates'].loc[aux_data['site_candidates']['city'] == city, 'country'].iloc[0]
    df['country'] = country
    df = pd.merge(df, aux_data['calendar'], on=['date', 'country'], how='left')
    df = pd.merge(df, aux_data['holidays'], on=['date', 'country'], how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0)
    df = pd.merge(df, aux_data['price_promo'], on=['date', 'sku', 'city'], how='left')
    df['launch_date'] = pd.to_datetime(df['launch_date'])

    # 2. 날짜 및 이벤트 피처
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['days_since_launch'] = (df['date'] - df['launch_date']).dt.days
    df['is_launch_period'] = (df['days_since_launch'] <= 30).astype(int)

    # 3. ***** (핵심) '특별 이벤트' 및 '외부 효과' 피처 생성 *****
    df['is_promo_event'] = 0
    df['is_external_event'] = 0
    
    for year, (event_country, start_date, end_date) in event_periods['promo'].items():
        df.loc[(df['year'] == year) & (df['country'] == event_country) & (df['date'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))), 'is_promo_event'] = 1
        
    if 'external' in event_periods and event_periods['external']:
        ext_year, (ext_country, ext_start, ext_end) = next(iter(event_periods['external'].items()))
        # 외부 효과는 모든 국가에 영향을 미친다고 가정
        df.loc[(df['year'] == ext_year) & (df['date'].between(pd.to_datetime(ext_start), pd.to_datetime(ext_end))), 'is_external_event'] = 1

    # 4. 과거 수요 및 가격 피처
    grouped = df.groupby(['city', 'sku'])
    for lag in [7, 14, 28, 35]:
        df[f'demand_lag_{lag}'] = grouped['demand'].shift(lag)
    for window in [7, 14, 28]:
        df[f'demand_rolling_mean_{window}'] = grouped['demand'].shift(1).rolling(window, min_periods=1).mean()
        df[f'demand_rolling_std_{window}'] = grouped['demand'].shift(1).rolling(window, min_periods=1).std()
    df[f'price_lag_1'] = grouped['unit_price'].shift(1)
    df[f'price_rolling_mean_7'] = grouped['unit_price'].shift(1).rolling(7, min_periods=1).mean()
        
    return df

# ======================================================================================
# 3. 메인 실행 로직
# ======================================================================================
if __name__ == "__main__":
    
    # 분석으로 확정된 이벤트 기간
    event_periods = {
        'promo': { 2018: ('KOR', '2018-02-14', '2018-03-20'), 2019: ('JPN', '2019-01-22', '2019-02-21'),
                   2020: ('USA', '2020-02-10', '2020-03-21'), 2021: ('USA', '2021-04-04', '2021-05-21'),
                   2022: ('KOR', '2022-03-01', '2022-03-31') },
        'external': { 2020: ('DEU', '2020-08-01', '2020-08-05') }
    }
    print("적용될 이벤트 기간:", event_periods)

    aux_data = load_auxiliary_data()

    print("\n--- 1단계: 모델 학습 및 저장 시작 ---")
    conn = sqlite3.connect(os.path.join(DATA_DIR, 'demand_train.db'))
    demand_df = pd.read_sql_query("SELECT * FROM demand_train", conn)
    conn.close()
    demand_df['date'] = pd.to_datetime(demand_df['date'])

    all_groups = list(demand_df.groupby(['city', 'sku']).groups.keys())
    
    sample_fraction = 0.75
    sampled_groups_keys = random.sample(all_groups, int(len(all_groups) * sample_fraction))
    print(f"전체 {len(all_groups)}개 그룹 중 {len(sampled_groups_keys)}개 그룹({int(sample_fraction*100)}%)을 샘플링하여 학습합니다...")

    processed_groups = []
    for group_key in tqdm(sampled_groups_keys, desc="샘플 그룹 피처 생성 중"):
        group_demand = demand_df[demand_df.set_index(['city', 'sku']).index.isin([group_key])].copy()
        group_df_featured = create_rich_features(group_demand, aux_data, event_periods)
        processed_groups.append(group_df_featured)

    train_df = pd.concat(processed_groups, ignore_index=True)
    del demand_df, processed_groups; gc.collect()

    target = 'demand'
    features = [col for col in train_df.columns if col not in ['demand', 'date', 'launch_date', 'sku', 'city', 'country', 'family', 'colour', 'season']]
    categorical_features = ['month', 'dayofweek', 'is_holiday', 'is_launch_period', 'is_promo_event', 'is_external_event']
    numerical_features = [f for f in features if f not in categorical_features]
    
    train_df[numerical_features] = train_df[numerical_features].fillna(-1)
    for col in categorical_features:
        train_df[col] = train_df[col].astype('category')

    X_train = train_df[features]
    y_train = train_df[target]
    y_train_log = np.log1p(y_train)

    print("최종 모델을 학습합니다...")
    lgb_params = { 'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 31, 'subsample': 0.85, 'colsample_bytree': 0.85, 'n_jobs': -1, 'seed': 42 }
    final_model = lgb.LGBMRegressor(**lgb_params)
    final_model.fit(X_train, y_train_log, categorical_feature=categorical_features, eval_set=[(X_train, y_train_log)], callbacks=[lgb.early_stopping(50, verbose=False)])

    print(f"모델을 '{MODEL_PATH}' 파일로 저장합니다...")
    joblib.dump(final_model, MODEL_PATH)
    train_categories_map = {col: X_train[col].cat.categories for col in categorical_features if col in X_train}
    del X_train, y_train, y_train_log, train_df; gc.collect()

    print("\n--- 2단계: 그룹별 순회 예측 시작 ---")
    model = joblib.load(MODEL_PATH)
    
    conn = sqlite3.connect(os.path.join(DATA_DIR, 'demand_train.db'))
    history_df = pd.read_sql_query("SELECT * FROM demand_train", conn)
    conn.close()
    history_df['date'] = pd.to_datetime(history_df['date'])
    
    submission_df = pd.read_csv(SUBMISSION_TEMPLATE_PATH)
    submission_df['date'] = pd.to_datetime(submission_df['date'])
    
    pred_df_base = submission_df.copy()
    
    all_preds = []
    for group_key, group_df_future in tqdm(pred_df_base.groupby(['city', 'sku'])):
        history_group = history_df[history_df.set_index(['city', 'sku']).index.isin([group_key])].copy()
        
        full_series_df = pd.concat([history_group, group_df_future.drop(columns=['mean'])], ignore_index=True)
        full_series_featured = create_rich_features(full_series_df, aux_data, event_periods)
        
        X_pred = full_series_featured[full_series_featured['date'].isin(group_df_future['date'])][features]
        X_pred[numerical_features] = X_pred[numerical_features].fillna(-1)
        for col in categorical_features:
            if col in X_pred.columns:
                X_pred[col] = pd.Categorical(X_pred[col], categories=train_categories_map.get(col))

        log_preds = model.predict(X_pred)
        daily_preds = np.expm1(log_preds)
        daily_preds[daily_preds < 0] = 0
        
        group_df_future.loc[:, 'mean'] = daily_preds.round().astype(int)
        all_preds.append(group_df_future[['date', 'city', 'sku', 'mean']])
    
    final_submission_df = pd.concat(all_preds, ignore_index=True)
    
    template_df = pd.read_csv(SUBMISSION_TEMPLATE_PATH)
    template_df['date'] = pd.to_datetime(template_df['date'])
    submission_output = pd.merge(template_df[['date', 'sku', 'city']], final_submission_df, on=['date', 'sku', 'city'], how='left')
    submission_output['mean'] = submission_output['mean'].fillna(0).astype(int)
    
    output_path = 'forecast_submission_upgraded.csv'
    submission_output.to_csv(output_path, index=False)

    print(f"\n✅ 2단계 완료! 예측 결과가 '{output_path}' 파일에 저장되었습니다.")
    print(submission_output.head())
