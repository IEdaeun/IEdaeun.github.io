import pandas as pd
import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("--- '신제품 공개 행사' 및 '외부 효과' 기간 최종 분석 시작 ---")
DATA_DIR = "./data"
conn = sqlite3.connect(os.path.join(DATA_DIR, 'demand_train.db'))
demand_df = pd.read_sql_query("SELECT * FROM demand_train", conn)
conn.close()
demand_df['date'] = pd.to_datetime(demand_df['date'])

site_candidates_df = pd.read_csv(os.path.join(DATA_DIR, 'site_candidates.csv'))
city_country_map = site_candidates_df.drop_duplicates('city')[['city', 'country']].set_index('city')['country'].to_dict()
demand_df['country'] = demand_df['city'].map(city_country_map)

daily_demand_by_country = demand_df.groupby(['date', 'country'])['demand'].sum().unstack().fillna(0)

promo_events = {}
external_events = {}

# 1. 매년 '신제품 공개 행사' 탐색 (상대적 변화점 기준)
for year in [2018, 2019, 2020, 2021, 2022]:
    year_data = daily_demand_by_country[daily_demand_by_country.index.year == year]
    
    year_zscores = year_data.apply(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    uniqueness_scores = year_zscores.apply(lambda x: x - year_zscores.drop(x.name, axis=1).mean(axis=1), axis=0)
    
    best_event_info = {'score': -np.inf, 'country': None, 'start_date': None, 'end_date': None}

    for country in uniqueness_scores.columns:
        country_scores = uniqueness_scores[country]
        is_event_potential = (country_scores > 1.2).astype(int) # 기준을 약간 낮춰 탐지율 향상
        if is_event_potential.sum() < 15: continue

        event_groups = (is_event_potential.diff() != 0).cumsum()
        for group in event_groups[is_event_potential == 1].unique():
            period_dates = event_groups[event_groups == group].index
            duration = (period_dates.max() - period_dates.min()).days + 1
            
            if 30 <= duration <= 120:
                score = country_scores.loc[period_dates].mean()
                if score > best_event_info['score']:
                    best_event_info = {
                        'score': score, 'country': country,
                        'start_date': period_dates.min().date(),
                        'end_date': period_dates.max().date()
                    }
    
    if best_event_info['country']:
        promo_events[year] = (best_event_info['country'], best_event_info['start_date'], best_event_info['end_date'])
    else: # Fallback: 만약 못찾으면 최고점 월이라도 선택
        monthly_uniqueness = uniqueness_scores.resample('MS').mean()
        top_series = monthly_uniqueness.stack().idxmax()
        event_date, country = top_series
        start_date = event_date.date()
        end_date = (event_date + pd.offsets.MonthEnd(0)).date()
        promo_events[year] = (country, start_date, end_date)


# 2. 2020-2021년 단일 '외부 효과' 탐색 (동반 급증 기준)
external_event_data = daily_demand_by_country[daily_demand_by_country.index.year.isin([2020, 2021])].copy()

# 신제품 행사 기간 데이터 영향력 감소 (완전히 제외하지 않고 가중치 감소)
for year in [2020, 2021]:
    if year in promo_events:
        country, start_date, end_date = promo_events[year]
        external_event_data.loc[str(start_date):str(end_date), country] *= 0.1

# 모든 국가 수요의 합으로 '글로벌' 수요 트렌드 생성
global_demand = external_event_data.sum(axis=1)
baseline = global_demand.rolling(window=90, min_periods=30).mean().bfill().ffill()
baseline_std = global_demand.rolling(window=90, min_periods=30).std().bfill().ffill()

is_spike = (global_demand > baseline + 2.5 * baseline_std)
if is_spike.sum() > 0:
    spike_groups = (is_spike.diff() != 0).cumsum()
    top_group = spike_groups[is_spike].value_counts().idxmax()
    period_dates = spike_groups[spike_groups == top_group].index
    start_date = period_dates.min().date()
    end_date = period_dates.max().date()
    main_country = external_event_data.loc[period_dates].sum().idxmax() # 대표 국가
    external_events[start_date.year] = (main_country, start_date, end_date)


print("\n" + "="*50)
print("✅ 탐색 완료! 최종 이벤트 기간은 다음과 같습니다.")
print("--- 신제품 공개 행사 (한 국가만 급증) ---")
for year, (country, start, end) in promo_events.items():
    print(f"  - {year}년: {country} (기간: {start} ~ {end})")
if external_events:
    print("\n--- 외부 효과 (여러 국가 동반 급증) ---")
    for year, (country, start, end) in external_events.items():
        print(f"  - {year}년: {country} (대표) (기간: {start} ~ {end})")
print("="*50)

# 3. 시각화
fig, axes = plt.subplots(len(promo_events), 1, figsize=(15, 18), sharex=True)
if len(promo_events) == 1: axes = [axes]

plot_df = daily_demand_by_country.stack().reset_index()
plot_df.columns = ['date', 'country', 'demand']
plot_df['year'] = plot_df['date'].dt.year

for i, year in enumerate(sorted(promo_events.keys())):
    ax = axes[i]
    plot_data_year = plot_df[plot_df['year']==year]
    sns.lineplot(data=plot_data_year, x='date', y='demand', hue='country', ax=ax, legend=(i==0))
    
    if year in promo_events:
        p_country, p_start, p_end = promo_events[year]
        ax.axvspan(p_start, p_end + pd.Timedelta(days=1), color='red', alpha=0.3)
    
    if year in external_events:
        ext_country, ext_start, ext_end = external_events[year]
        ax.axvspan(ext_start, ext_end + pd.Timedelta(days=1), color='blue', alpha=0.3)

    ax.set_title(f'{year} Daily Total Demand by Country', fontsize=14)

plt.tight_layout()
plt.show()