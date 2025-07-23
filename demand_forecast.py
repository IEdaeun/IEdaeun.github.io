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
# 1. 피처 엔지니어링 함수
# ======================================================================================
def create_features(df):
    """데이터프레임에 날짜 기반 및 Lag/Rolling 피처를 생성합니다."""
    df = df.sort_values('date').copy()
    
    # 기본 날짜 피처 생성
    df['year'] = df['date'].dt.year
    df['days_since_launch'] = (df['date'] - df['launch_date']).dt.days

    grouped = df.groupby(['city', 'sku'])
    
    new_numeric_features = []
    for lag in [7, 14, 28]:
        col_name = f'demand_lag_{lag}'
        df[col_name] = grouped['demand'].shift(lag)
        new_numeric_features.append(col_name)
        
    for window in [7, 14, 28]:
        shifted_demand = grouped['demand'].shift(1)
        mean_col_name = f'demand_rolling_mean_{window}'
        df[mean_col_name] = shifted_demand.rolling(window, min_periods=1).mean()
        new_numeric_features.append(mean_col_name)

    df[new_numeric_features] = df[new_numeric_features].fillna(0).astype(np.float32)
    return df

# ======================================================================================
# 2. 메인 실행 로직
# ======================================================================================
if __name__ == "__main__":
    DATA_DIR = "./data"
    MODEL_PATH = 'demand_model_final.pkl'
    SUBMISSION_TEMPLATE_PATH = 'data/forecast_submission_template.csv'
    np.random.seed(42)
    random.seed(42)

    # --- 1단계: 모델 학습 및 저장 ---
    print("--- 1단계: 모델 학습 및 저장 시작 ---")
    
    # 원본 데이터 로딩
    conn = sqlite3.connect(os.path.join(DATA_DIR, 'demand_train.db'))
    demand_df = pd.read_sql_query("SELECT * FROM demand_train", conn)
    conn.close()
    sku_meta_df = pd.read_csv(os.path.join(DATA_DIR, 'sku_meta.csv'))
    
    demand_df['date'] = pd.to_datetime(demand_df['date'])
    sku_meta_df['launch_date'] = pd.to_datetime(sku_meta_df['launch_date'])

    # 학습에 사용할 그룹 키 샘플링
    all_groups = list(demand_df.groupby(['city', 'sku']).groups.keys())
    sample_fraction = 0.25
    sampled_groups_keys = random.sample(all_groups, int(len(all_groups) * sample_fraction))
    print(f"전체 {len(all_groups)}개 그룹 중 {len(sampled_groups_keys)}개 그룹을 샘플링합니다...")

    processed_groups = []
    for group_key in tqdm(sampled_groups_keys, desc="샘플 그룹 피처 생성 중"):
        city, sku = group_key
        group_demand = demand_df[(demand_df['city'] == city) & (demand_df['sku'] == sku)]
        group_sku = sku_meta_df[sku_meta_df['sku'] == sku]
        group_df = pd.merge(group_demand, group_sku, on='sku', how='left')
        group_df = create_features(group_df)
        processed_groups.append(group_df)

    train_df = pd.concat(processed_groups)
    del demand_df, processed_groups; gc.collect()

    train_df.fillna(-1, inplace=True)

    target = 'demand'
    features = [col for col in train_df.columns if col not in ['demand', 'date', 'launch_date', 'sku', 'city']]
    categorical_features = [col for col in features if train_df[col].dtype.name in ['category', 'object']]

    for col in categorical_features:
        train_df[col] = train_df[col].astype('category')

    X_train = train_df[features]
    y_train = train_df[target]
    y_train_log = np.log1p(y_train)

    print("최종 모델을 학습합니다...")
    lgb_params = {'objective': 'regression_l1', 'n_estimators': 300, 'learning_rate': 0.05, 'n_jobs': -1, 'seed': 42}
    final_model = lgb.LGBMRegressor(**lgb_params)
    final_model.fit(X_train, y_train_log, categorical_feature='auto')

    print(f"모델을 '{MODEL_PATH}' 파일로 저장합니다...")
    joblib.dump(final_model, MODEL_PATH)
    train_categories_map = {col: X_train[col].cat.categories for col in categorical_features}
    del X_train, y_train, y_train_log, train_df; gc.collect()

    # --- 2단계: 그룹별 예측 ---
    print("\n--- 2단계: 그룹별 순회 예측 시작 ---")
    model = joblib.load(MODEL_PATH)
    
    # ***** 핵심 수정: 예측을 위해 원본 데이터를 다시 로드 *****
    conn = sqlite3.connect(os.path.join(DATA_DIR, 'demand_train.db'))
    history_df = pd.read_sql_query("SELECT * FROM demand_train", conn)
    conn.close()
    history_df['date'] = pd.to_datetime(history_df['date'])
    
    submission_df = pd.read_csv(SUBMISSION_TEMPLATE_PATH)
    submission_df['date'] = pd.to_datetime(submission_df['date'])
    
    pred_df_base = pd.merge(submission_df, sku_meta_df, on=['sku'], how='left')
    del sku_meta_df; gc.collect()
    
    all_preds = []
    for group_key, group_df_future in tqdm(pred_df_base.groupby(['city', 'sku'])):
        city, sku = group_key
        history_group = history_df[(history_df['city'] == city) & (history_df['sku'] == sku)].copy()
        
        full_series_df = pd.concat([history_group, group_df_future], ignore_index=True)
        full_series_featured = create_features(full_series_df)
        
        X_pred = full_series_featured[full_series_featured['date'].isin(group_df_future['date'])][features]
        X_pred.fillna(-1, inplace=True)

        for col in categorical_features:
            if col in X_pred.columns:
                X_pred[col] = pd.Categorical(X_pred[col], categories=train_categories_map[col])
        
        log_preds = model.predict(X_pred)
        daily_preds = np.expm1(log_preds)
        daily_preds[daily_preds < 0] = 0
        
        group_df_future['mean'] = daily_preds.round().astype(int)
        all_preds.append(group_df_future[['date', 'city', 'sku', 'mean']])
        
    final_submission_df = pd.concat(all_preds, ignore_index=True)
    final_submission_df = pd.merge(submission_df.drop(columns=['mean']), final_submission_df, on=['date', 'city', 'sku'])
    final_submission_df = final_submission_df[['date', 'sku', 'city', 'mean']]
    
    output_path = 'forecast_submission_final.csv'
    final_submission_df.to_csv(output_path, index=False)

    print(f"\n✅ 2단계 완료! 예측 결과가 '{output_path}' 파일에 저장되었습니다.")
    print(final_submission_df.head())