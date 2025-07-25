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
# Random Seed 고정
np.random.seed(42)
random.seed(42)

# 경로 설정
DATA_DIR = "./data"
MODEL_PATH = 'demand_model_upgraded.pkl'
SUBMISSION_TEMPLATE_PATH = os.path.join(DATA_DIR, 'forecast_submission_template.csv')

# ======================================================================================
# 1. 보조 데이터 사전 로딩 함수 (신설)
# ======================================================================================
def load_auxiliary_data():
    """성능 향상을 위해 모든 보조 데이터를 미리 로드합니다."""
    print("보조 데이터 로딩 중...")
    # sku 메타 데이터
    sku_meta_df = pd.read_csv(os.path.join(DATA_DIR, 'sku_meta.csv'))
    sku_meta_df['launch_date'] = pd.to_datetime(sku_meta_df['launch_date'])
    # 가격/프로모션 데이터
    price_promo_df = pd.read_csv(os.path.join(DATA_DIR, 'price_promo_train.csv'))
    price_promo_df['date'] = pd.to_datetime(price_promo_df['date'])
    # 위치 데이터
    site_candidates_df = pd.read_csv(os.path.join(DATA_DIR, 'site_candidates.csv'))
    # 캘린더 데이터
    calendar_df = pd.read_csv(os.path.join(DATA_DIR, 'calendar.csv'))
    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    # 공휴일 데이터
    holiday_df = pd.read_csv(os.path.join(DATA_DIR, 'holiday_lookup.csv'))
    holiday_df['date'] = pd.to_datetime(holiday_df['date'])
    holiday_df['is_holiday'] = 1
    holiday_df = holiday_df[['date', 'country', 'is_holiday']].drop_duplicates()
    
    return {
        "sku_meta": sku_meta_df,
        "price_promo": price_promo_df,
        "site_candidates": site_candidates_df,
        "calendar": calendar_df,
        "holidays": holiday_df
    }

# ======================================================================================
# 2. 피처 엔지니어링 함수 (대폭 강화)
# ======================================================================================
def create_rich_features(df, aux_data):
    """단일 그룹 데이터프레임에 보조 데이터를 병합하고 풍부한 피처를 생성합니다."""
    df = df.sort_values('date').copy()
    
    # 보조 데이터 병합
    df = pd.merge(df, aux_data['sku_meta'], on='sku', how='left')
    city = df['city'].iloc[0]
    country = aux_data['site_candidates'].loc[aux_data['site_candidates']['city'] == city, 'country'].iloc[0]
    df['country'] = country
    df = pd.merge(df, aux_data['calendar'], on=['date', 'country'], how='left')
    df = pd.merge(df, aux_data['holidays'], on=['date', 'country'], how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0)
    df = pd.merge(df, aux_data['price_promo'], on=['date', 'sku', 'city'], how='left')

    # 날짜 및 이벤트 피처
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['days_since_launch'] = (df['date'] - df['launch_date']).dt.days
    df['is_launch_period'] = (df['days_since_launch'] <= 30).astype(int) # (핵심) 신제품 출시 기간 플래그

    # 과거 수요(Lag/Rolling) 피처
    grouped = df.groupby(['city', 'sku'])
    for lag in [7, 14, 28, 35]:
        df[f'demand_lag_{lag}'] = grouped['demand'].shift(lag)
    for window in [7, 14, 28]:
        df[f'demand_rolling_mean_{window}'] = grouped['demand'].shift(1).rolling(window, min_periods=1).mean()
        df[f'demand_rolling_std_{window}'] = grouped['demand'].shift(1).rolling(window, min_periods=1).std()
    
    # 가격 피처
    df[f'price_lag_1'] = grouped['unit_price'].shift(1)
    df[f'price_rolling_mean_7'] = grouped['unit_price'].shift(1).rolling(7, min_periods=1).mean()
        
    return df

# ======================================================================================
# 3. 메인 실행 로직 (최적화)
# ======================================================================================
if __name__ == "__main__":
    
    aux_data = load_auxiliary_data()

    print("\n--- 1단계: 모델 학습 및 저장 시작 ---")
    
    conn = sqlite3.connect(os.path.join(DATA_DIR, 'demand_train.db'))
    demand_df = pd.read_sql_query("SELECT * FROM demand_train", conn)
    conn.close()
    demand_df['date'] = pd.to_datetime(demand_df['date'])

    all_groups = list(demand_df.groupby(['city', 'sku']).groups.keys())
    
    # 포인트 1: 학습 데이터 75%로 증량
    sample_fraction = 0.75
    sampled_groups_keys = random.sample(all_groups, int(len(all_groups) * sample_fraction))
    print(f"전체 {len(all_groups)}개 그룹 중 {len(sampled_groups_keys)}개 그룹({int(sample_fraction*100)}%)을 샘플링하여 학습합니다...")

    processed_groups = []
    for group_key in tqdm(sampled_groups_keys, desc="샘플 그룹 피처 생성 중"):
        city, sku = group_key
        group_demand = demand_df[(demand_df['city'] == city) & (demand_df['sku'] == sku)].copy()
        # 강화된 피처 생성 함수 사용
        group_df_featured = create_rich_features(group_demand, aux_data)
        processed_groups.append(group_df_featured)

    train_df = pd.concat(processed_groups, ignore_index=True)
    del demand_df, processed_groups; gc.collect()

    # 피처/타겟 정의
    target = 'demand'
    features = [col for col in train_df.columns if col not in ['demand', 'date', 'launch_date', 'sku', 'city', 'country', 'family', 'colour', 'season']]
    categorical_features = ['month', 'dayofweek', 'is_holiday', 'is_launch_period']

    # 결측치 처리
    numerical_features = [f for f in features if f not in categorical_features]
    train_df[numerical_features] = train_df[numerical_features].fillna(-1)
    for col in categorical_features:
        train_df[col] = train_df[col].astype('category')

    X_train = train_df[features]
    y_train = train_df[target]
    y_train_log = np.log1p(y_train)

    print("최종 모델을 학습합니다...")
    # 포인트 2: 모델 파라미터 정교화 및 학습량 증대
    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt'
    }
    
    final_model = lgb.LGBMRegressor(**lgb_params)
    final_model.fit(X_train, y_train_log, categorical_feature=categorical_features)

    print(f"모델을 '{MODEL_PATH}' 파일로 저장합니다...")
    joblib.dump(final_model, MODEL_PATH)
    train_categories_map = {col: X_train[col].cat.categories for col in categorical_features if col in X_train}
    del X_train, y_train, y_train_log, train_df; gc.collect()

    # --- 2단계: 그룹별 예측 ---
    print("\n--- 2단계: 그룹별 순회 예측 시작 ---")
    model = joblib.load(MODEL_PATH)
    
    conn = sqlite3.connect(os.path.join(DATA_DIR, 'demand_train.db'))
    history_df = pd.read_sql_query("SELECT * FROM demand_train", conn)
    conn.close()
    history_df['date'] = pd.to_datetime(history_df['date'])
    
    submission_df = pd.read_csv(SUBMISSION_TEMPLATE_PATH)
    submission_df['date'] = pd.to_datetime(submission_df['date'])
    
    # 70점 코드의 구조를 그대로 사용 (미리 병합하지 않음)
    pred_df_base = submission_df.copy()
    
    all_preds = []
    for group_key, group_df_future in tqdm(pred_df_base.groupby(['city', 'sku'])):
        city, sku = group_key
        history_group = history_df[(history_df['city'] == city) & (history_df['sku'] == sku)].copy()
        
        full_series_df = pd.concat([history_group, group_df_future.drop(columns=['mean'])], ignore_index=True)
        # 강화된 피처 생성 함수 사용
        full_series_featured = create_rich_features(full_series_df, aux_data)
        
        X_pred = full_series_featured[full_series_featured['date'].isin(group_df_future['date'])][features]

        # 결측치 처리
        X_pred[numerical_features] = X_pred[numerical_features].fillna(-1)
        for col in categorical_features:
            if col in X_pred.columns:
                X_pred[col] = pd.Categorical(X_pred[col], categories=train_categories_map.get(col))

        log_preds = model.predict(X_pred)
        daily_preds = np.expm1(log_preds)
        daily_preds[daily_preds < 0] = 0
        
        group_df_future['mean'] = daily_preds.round().astype(int)
        all_preds.append(group_df_future[['date', 'city', 'sku', 'mean']])
    
    final_submission_df = pd.concat(all_preds, ignore_index=True)
    final_submission_df = pd.merge(submission_df.drop(columns=['mean']), final_submission_df, on=['date', 'city', 'sku'], how='left')
    final_submission_df['mean'] = final_submission_df['mean'].fillna(0).astype(int)
    
    output_path = 'forecast_submission_upgraded.csv'
    final_submission_df.to_csv(output_path, index=False)

    print(f"\n✅ 2단계 완료! 예측 결과가 '{output_path}' 파일에 저장되었습니다.")
    print(final_submission_df.head())
