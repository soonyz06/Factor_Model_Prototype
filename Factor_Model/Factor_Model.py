import time
import os
import pandas as pd
from yahooquery import Ticker
from financial_utils import (
    compute_values,
    get_sample,
    get_cum_score,
    plot_results,
    get_intervals,
    normalise_metrics,
    get_benchmark,
    get_period_dates,
    save_csv,
    income_cache,
    balance_cache
)
from tqdm import tqdm
from itertools import product, chain
import numpy as np
import config



# ------------------- Initialise -------------------


FIELDS = config.FIELDS

###Research BEFORE do
###PLAN -> data source, data storage, speed, scalability, use AI instead???
###bottom up approach using bayesian learning, remove arbitrary groupings? weights...
#vectorisation, avoid loops
#levers: industry neutralisation, neutral imputation, rank_normalisation, winsor, log normalise [DATA]

step = config.step
values = np.arange(0, 1 + step, step)
weights = list(product(values, repeat=len(FIELDS)))
weights = [list(w) for w in weights if abs(sum(w) - 1) <= config.tolerance]
weights = weights[::-1]
#weights.append([round(1/len(FIELDS), 2) for f in FIELDS])
labels = [FIELDS[weight.index(1)] if 1 in weight else weight for weight in weights]
FIELDS.append(["Beta", "Performance"]) #doesn't include ["Tickers", "Industry", "Score"]
FIELDS_1D = list(chain.from_iterable(FIELDS))
model_results =  [[] for k in weights]
benchmark_results = []

seed = 0
n = 600 
y = 3
read_only = True



print(f"\n\n------------------- {config.word1} {config.mode.title()}{config.word2} Portfolio.v{seed} (y={y}, n={n}) -------------------")
print(f"FACTORS: {FIELDS[:-1]}")
print(f"COMBINATIONS: {len(weights)}\n\n")



intervals = get_intervals(y)
tickers, industries = get_sample(n, seed)
iindustries = [i for i in range(len(industries))]
folder = f'backtests/factor_scores{seed}.'
os.makedirs(folder, exist_ok=True)
img_path = os.path.join(folder, "performance.png")

old_year = -1
for q, year in tqdm(intervals):
    if year!=old_year:
        income_cache.clear()
        balance_cache.clear()
        old_year = year

    print(f" Q{q}FY{str(year)[2:]}")
    csv_path = os.path.join(folder, f"Q{q}FY{year}.csv")
    existing_data = {}
    missing_columns = FIELDS_1D
    col_existing = 0
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            for _, row in df_existing.iterrows():
                ticker = row["Ticker"].upper()
                existing_data[ticker] = {}
                for field in df_existing.columns:
                    value = row.get(field)
                    existing_data[ticker][field] = value
            missing_columns = [field for field in FIELDS_1D if field not in df_existing.columns]
            col_existing = len(df_existing.columns)
        except Exception as e:
            print(f"Error reading existing CSV: {e}")

    if read_only or len(FIELDS_1D)+3<col_existing: print("WON'T SAVE CSV!!!")
    
    p = (len(tickers)-len(existing_data)) / len(tickers) 
    if not read_only: print(f"\nLOADING tickers ({p*100:.1f}% missing, {missing_columns})...")
    period = get_period_dates(year, q)
    benchmark = get_benchmark(period)
    data = []
    iterable = tqdm(tickers) if not read_only else tickers
    for i, tick in enumerate(iterable):
        symbol, iindustry = tick[0], tick[1]
        computed_fields = {}        
        try:
            if read_only:
                computed_fields = {k: v for k, v in existing_data[symbol].items() if k in FIELDS_1D}
            elif symbol not in existing_data:
                computed_fields = {}
                values = compute_values(symbol, period, benchmark, FIELDS_1D) 
                for field, val in zip(FIELDS_1D, values):
                    computed_fields[field] = val
            else: 
                computed_fields = {k: v for k, v in existing_data[symbol].items() if k in FIELDS_1D}
                if len(missing_columns)!=0: #adding new metric
                    values = compute_values(symbol, period, benchmark, missing_columns)
                    for field, val in zip(missing_columns, values):
                        computed_fields[field] = val
        except Exception as e:
            if not read_only: print(f"{symbol}: Error computing fields → {e}")
        
        record = {"Ticker": symbol}
        record["Industry"] = iindustry
        record["Score"] = 0
        record.update(computed_fields)
        if all(field in record for field in FIELDS_1D):
            data.append(record)
        else:
            if not read_only: print(f"{symbol}: Skipped due to incomplete metric set")
        time.sleep(0.1)
        if(not read_only and i>0 and i%50==0): time.sleep(15)

    if not read_only and len(data)>=len(existing_data) and len(FIELDS_1D)+3>=col_existing: #+3
        save_csv(data, csv_path, FIELDS_1D) #where is unnamed from



    #print("\n\n------------------- Backtesting Performance -------------------")


    
    print(f"\nBACKTEST Results")
    top_returns = []
    factor_signals, betas = normalise_metrics(data, iindustries, FIELDS[:-1]) ###factor timing: spreads or momentum within factors & vol targetting
    baseline = round(benchmark["performance"], 6)
    for i, weight in enumerate(weights):
        print(labels[i])
        current_return = get_cum_score(data, weight, factor_signals, betas, baseline) #backtest
        model_results[i].append(round(current_return, 6))
    benchmark_results.append(baseline)    
    scores = {item["Ticker"]: item["Score"] for item in data}
    save_csv(data, csv_path, FIELDS_1D, scores) 
    time.sleep(5)
        
if read_only: plot_results(model_results, intervals, img_path, labels, benchmark_results) 













