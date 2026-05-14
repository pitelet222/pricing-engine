import pandas as pd

P = 'data/outputs/'

df = pd.read_csv('data/processed/avocado_features.csv', parse_dates=['Date'])
df['unique_id'] = df['region'] + '_' + df['is_organic'].map({0:'conventional',1:'organic'})

print('=== DATASET ===')
print(f'Rows: {len(df)}')
print(f'Date range: {df["Date"].min().date()} -> {df["Date"].max().date()}')
print(f'Regions: {df["region"].nunique()}   Series: {df["unique_id"].nunique()}')

print('\n=== PRICE BY TYPE ===')
for t, name in [(0,'Conventional'),(1,'Organic')]:
    g = df[df['is_organic']==t]['AveragePrice']
    print(f'  {name}: mean=${g.mean():.2f}  std={g.std():.2f}  min=${g.min():.2f}  max=${g.max():.2f}')

print('\n=== PRICE-VOLUME CORRELATION ===')
for t, name in [(0,'Conventional'),(1,'Organic')]:
    g = df[df['is_organic']==t]
    c = g['AveragePrice'].corr(g['log_total_volume'])
    print(f'  {name}: corr(price, log_vol) = {c:.3f}')

print('\n=== TOP 10 REGIONS BY MEDIAN VOLUME ===')
rv = df.groupby('region')['Total Volume'].median().sort_values(ascending=False).head(10)
print(rv.round(0).to_string())

r = pd.read_csv(P+'pricing_recommendations.csv')
print('\n=== PRICING RECS BY TYPE ===')
for t, name in [(0,'Conventional'),(1,'Organic')]:
    g = r[r['is_organic']==t]
    print(f'  {name}: median_uplift={g["revenue_change_pct"].median():.1f}%  median_price_change={g["price_change_pct"].median():.1f}%')

e = pd.read_csv(P+'price_elasticity.csv')
print('\n=== MOST ELASTIC (top 5) ===')
print(e.nsmallest(5,'elasticity')[['unique_id','AveragePrice','elasticity']].to_string(index=False))
print('\n=== LEAST ELASTIC / PRICING POWER (top 5) ===')
print(e.nlargest(5,'elasticity')[['unique_id','AveragePrice','elasticity']].to_string(index=False))

u = pd.read_csv(P+'uncertainty_pricing.csv')
print('\n=== HIGHEST REVENUE SPREAD (most uncertain) ===')
print(u.nlargest(5,'rev_spread_pct')[['unique_id','rev_spread_pct','downside_risk_pct','uplift_sharpe']].to_string(index=False))
print('\n=== BEST SHARPE (highest quality opportunities) ===')
print(u.nlargest(5,'uplift_sharpe')[['unique_id','opt_price_balanced','rev_uplift_balanced','uplift_sharpe']].to_string(index=False))

print('\n=== PER-SERIES MAE TOP 5 WORST ===')
ps = pd.read_csv(P+'forecast_per_series_mae.csv')
worst = ps[ps['Model']=='AutoETS'].nlargest(5,'MAE')[['unique_id','MAE']]
print(worst.to_string(index=False))
