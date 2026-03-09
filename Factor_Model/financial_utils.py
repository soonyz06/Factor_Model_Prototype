import pandas as pd
from yahooquery import Ticker
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from sec_utils import (
    get_latest,
    get_balance,
    get_balance_sheet,
    calc_mult,
    calc_IV
)
import config
import copy
    
fx_cache = {}
income_cache = {}
balance_cache ={}

def get_sample(n, seed=42, csv_path="nasdaq.csv"):
    df = pd.read_csv(csv_path).dropna()
    industries = sorted(df['Industry'].dropna().unique().tolist())
    industry_to_index = {industry: i for i, industry in enumerate(industries)}
    sampled_df = df[['Symbol', 'Industry']].dropna().sample(n=n, random_state=seed)
    #industry_df = df[df['Industry'] == target][['Symbol', 'Industry']].dropna()
    #sampled_df = industry_df.sample(n=n, random_state=seed)
    
    tickers = [(row['Symbol'], industry_to_index[row['Industry']]) for _, row in sampled_df.iterrows()]
    return tickers, industries

def rank_normalise(col, fill=np.nan):
    col = np.asarray(col)
    mask = np.isfinite(col)
    filtered_vals = col[mask]
    sorted_unique = np.sort(np.unique(filtered_vals))
    if len(sorted_unique) == 1:
        rank_map = {sorted_unique[0]: 0.5}
    else:
        rank_map = {
            v: i / (len(sorted_unique) - 1)
            for i, v in enumerate(sorted_unique)
        }
    norm_vals = [rank_map.get(v, fill) if np.isfinite(v) else fill for v in col]
    return np.array(norm_vals)

import numpy as np

def z_normalise(col, lower_pct=0.01, upper_pct=0.99, fill=0):
    arr = np.asarray(col, dtype=float)
    mask = np.isfinite(arr)
    valid_vals = arr[mask]
    
    if valid_vals.size == 0:
        return np.full(arr.shape, fill)
    lower_bound = np.percentile(valid_vals, lower_pct * 100)
    upper_bound = np.percentile(valid_vals, upper_pct * 100)
    capped_vals = np.clip(valid_vals, lower_bound, upper_bound)
    mean = np.mean(capped_vals)
    std = np.std(capped_vals)
    std = std if std > 0 else 1e-6
    norm_vals = np.full(arr.shape, fill, dtype=float)
    capped_input = np.clip(arr[mask], lower_bound, upper_bound)
    norm_vals[mask] = (capped_input - mean) / std
    return norm_vals

def get_intervals(n):
    now = date.today()
    years = [now.year - n + i + 1 for i in range(n)]
    interval_months = config.interval_months
    result = []
    for year in years:
        for i, month in enumerate(interval_months):
            interval_start = date(year, month, 3)
            if year < now.year or (year == now.year and interval_start < now):
                if not(month<4 and year==2023): #
                    result.append((i + 1, year))  
    return result

def get_period_dates(year, q): 
    now = date.today()
    interval_months = config.interval_months
    month = interval_months[q-1]
    start_date = date(year, month, 3)
    if len(interval_months)>1 and q<4:
        next_month = interval_months[q]
        next_year = year
    else:
        next_month = interval_months[0]
        next_year = year+1
    interval_end = date(next_year, next_month, 3)
    if year < now.year or (year == now.year and interval_end < now):
        end_date = interval_end
    else:
        end_date = now
    return (start_date, end_date)

def get_close(ticker, curr_date):
    start_date = curr_date - timedelta(days=5)
    end_date = curr_date
    df = ticker.history(start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval='1d').reset_index()
    if df.empty:
        return np.nan
    return df.iloc[-1]['adjclose']

def get_price_change(ticker, end_date, units, end_price=None, lag=0, unit_type='months', sign=1):
    def get_offset(units):
        if unit_type == 'days':
            offset = timedelta(days=units)
        elif unit_type == 'weeks':
            offset = timedelta(weeks=units)
        elif unit_type == 'months':
            offset = relativedelta(months=units)
        else:
            raise ValueError(f"Unsupported unit_type: {unit_type}")
        return offset
    
    if lag!=0 or end_price is None:
        end_date = end_date - get_offset(lag)
        end_price = get_close(ticker, end_date)
    start_date = end_date - get_offset(units)
    start_price = get_close(ticker, start_date)
    if np.isnan(start_price) or np.isnan(end_price): return np.nan
    return round((end_price / start_price - 1) * 100, 4) * sign

def compute_values(ticker, period, benchmark, missing_columns):
    stock = Ticker(ticker)
    start_date = period[0]
    end_date = period[1]
    start_price = get_close(stock, start_date)
    if np.isnan(start_price): raise ValueError(f"Start Price")
    y0 = y0 = start_date.year - 1 if (start_date.month, start_date.day) < (4, 3) else start_date.year
    L = 2025 - (y0 - 1)
    metric_map = {}
    price_data = config.price_data
    beta_data = config.beta_data
    sec_data =  config.sec_data

    #Price 
    if any(col in price_data for col in missing_columns):
        market_map = { 
            "HML_5": lambda: get_price_change(stock, start_date, 12*5, start_price, sign=-1), #lambda x: func(10, x)
            "HML_3": lambda: get_price_change(stock, start_date, 12*3, start_price, sign=-1),
            "HML_2": lambda: get_price_change(stock, start_date, 12*2, start_price, sign=-1),
            "UMD_12": lambda: get_price_change(stock, start_date, 12, lag=1),
            "UMD_6": lambda: get_price_change(stock, start_date, 6, lag=1),
            "UMD_3": lambda: get_price_change(stock, start_date, 3, lag=1),
            "STR_1": lambda: get_price_change(stock, start_date, 1, start_price, sign=-1),
            "STR_0": lambda: get_price_change(stock, start_date, 1, start_price, unit_type="weeks", sign=-1),
            "Performance": lambda: round((get_close(stock, end_date) / start_price - 1), 4)
        }
        metric_map.update(market_map)
        
    #Beta
    if any(col in beta_data for col in missing_columns):
        Beta = get_beta(stock, start_date, benchmark)
        beta_map = {
            'BAB_1': lambda: Beta * -1 if not np.isnan(Beta) else np.nan,
            "Beta": lambda: Beta
        }
        metric_map.update(beta_map)


    #SEC
    if any(col in sec_data for col in missing_columns): 
        #Cache
        file = f"{ticker}_{y0}".upper()
        income_df = income_cache.get(file, None)
        balance_df = balance_cache.get(file, None)
        if income_df is None:
            income_df = stock.income_statement(frequency='annual', trailing=False)
            if isinstance(income_df, str): raise ValueError("Income Statement")
            income_df = income_df.sort_values('asOfDate')
            income_cache[file] = income_df
        if balance_df is None:
            balance_df = stock.balance_sheet(frequency='annual')
            if isinstance(balance_df, str): raise ValueError("Balance Sheet")
            balance_df = balance_df.sort_values('asOfDate')
            balance_cache[file] = balance_df            
        fx_rate = 1
        currency = income_df.iloc[-L]['currencyCode']
        if currency.upper() != 'USD':
            if currency.upper() in fx_cache:
                fx_rate = fx_cache[currency.upper()]
            else:
                fx_ticker = Ticker(f"{currency.upper()}USD=X")
                fx_data = fx_ticker.price
                fx_rate = fx_data[f"{currency.upper()}USD=X"]['regularMarketPrice']
                fx_cache[currency.upper()] = fx_rate

        revenue = get_latest(income_df, 'OperatingRevenue', -L)      
        gross_profit = get_latest(income_df, 'GrossProfit', -L)
        operating_income = get_latest(income_df, "OperatingIncome", -L)
        net_income = get_latest(income_df, "PretaxIncome", -L)*.8        
        latest_bs = balance_df.iloc[-L]
        latest_bs2 = balance_df.iloc[-(L+1)] #In-TA/Book>
        cash, total_debt, mezzaine, equity, shares, assets = get_balance_sheet(latest_bs)
        
        mc = (start_price*shares)
        ev =  mc + (total_debt + mezzaine - cash) * fx_rate
        gp_iv = ((gross_profit*0.8)*10 + cash - total_debt - mezzaine)
        ebit_iv = ((operating_income*0.8)*10 + cash - total_debt - mezzaine)
        ni_iv = (net_income*10)
        ivs = [np.nan, np.nan, np.nan]
        if any(col in ["IV0", "IV1", "IV2"] for col in missing_columns) and shares > 0: #reorganise everything :)
            denom = mc / shares 
            ivs = calc_IV(income_df, balance_df, fx_rate, L)
            ivs = [(i/denom-1)*100 if not np.isnan(i) else np.nan for i in ivs]

        sec_map = {
            'IV0': lambda: ivs[0],
            'IV1': lambda: ivs[1],
            'IV2': lambda: ivs[2],
            'NCAV': lambda: calc_mult((get_balance(latest_bs, ['CurrentAssets', 'OtherCurrentAssets'])-get_balance(latest_bs, ['TotalLiabilitiesNetMinorityInterest']))*fx_rate, mc, change=1),
            'PB': lambda:   calc_mult(mc, equity*fx_rate, sign=-1),
            'GP_IV': lambda: calc_mult(gp_iv*fx_rate, mc, change=1),
            'EBIT_IV': lambda: calc_mult(ebit_iv*fx_rate, mc, change=1),
            'NI_IV': lambda: calc_mult(ni_iv*fx_rate, mc, change=1),
            
            'GP_EV': lambda: calc_mult(gross_profit*fx_rate, ev),
            'EBIT_EV': lambda: calc_mult(operating_income*fx_rate, ev),
            'NI_MC': lambda: calc_mult(net_income*fx_rate, mc),

            'REV_G': lambda: calc_mult(revenue, get_latest(income_df, 'OperatingRevenue', -(L+1)), mode=2, fill=np.nan, change=1),
            
            'GPOA': lambda: calc_mult(gross_profit, assets),
            'ROA': lambda: calc_mult(operating_income*.8, assets),
            'ROE': lambda: calc_mult(net_income, equity),
            
            'IA_G': lambda: calc_mult(assets, get_balance_sheet(latest_bs2, 2), mode=2, fill=np.nan, change=1, sign=-1),
            'IE_G': lambda: calc_mult(equity, get_balance_sheet(latest_bs2, 3), mode=2, fill=np.nan, change=1, sign=-1),
            
            'GMGN': lambda: calc_mult(gross_profit, revenue, mode=2, fill=np.nan),
            'OMGN': lambda: calc_mult(operating_income, revenue, mode=2, fill=np.nan),
            'NMGN': lambda: calc_mult(net_income, revenue, mode=2, fill=np.nan),
            
            'BIG': lambda: np.log(len(str(max(int(revenue*fx_rate), 1))))       
        }
        metric_map.update(sec_map)
        
    values=[]
    errors = []
    for key in missing_columns:
        func = metric_map.get(key)
        if func is not None:
            try:
                values.append(func())
            except Exception:
                values.append(np.nan)
                errors.append(key)
    if len(errors)>0: print(f"{ticker}: ERROR computing {errors}")
    return values

def normalise_metrics(raw_data, iindustries, factors):
    def col_avg(a): #combine metrics to form factor
        s = np.nansum(a, axis=0)
        cnt = np.sum(~np.isnan(a), axis=0)
        return np.divide(s, cnt, out=np.full_like(s, np.nan), where=cnt>0) 

    data = copy.deepcopy(raw_data)
    #neutralise metrics
    neut_bool = config.neut_bool
    metrics = list(chain.from_iterable(factors))
    for metric in metrics + ["Beta"]:  
        for industry in iindustries:
            industry_entries = [d for d in data if d['Industry'] == industry]
            values = [d.get(metric, np.nan) for d in industry_entries]
            values = [v for v in values if not np.isnan(v)]
            median = np.nanmedian(values) if(len(values) > 0) else 0
            m_median = median if(neut_bool) else 0
            b_median = median if(neut_bool) else 1
            for d in industry_entries:
                val = d.get(metric, np.nan)
                if metric=="Beta":
                    if np.isnan(val):
                        d["Beta"] = b_median #industry median vs reject null beta
                    else:
                        w = config.w_shrink
                        d["Beta"] = w * d["Beta"] + (1 - w) * b_median #shrinkage towards industry median
                else:
                    if np.isnan(val) :
                        d[metric] = np.nan  #d[metric]=0 for neutral imputation vs available
                    else:
                        d[metric] = np.round(val - m_median, decimals=6)                 
                    
    #composite factor signal
    factor_signals = []
    spreads = []
    for factor in factors: 
        metrics = []
        for metric in factor:
            values = [d.get(metric, np.nan) for d in data]
            metrics.append(z_normalise(rank_normalise(values), fill=np.nan)) #z_noramlise(rank_normalise(values), fill=np.nan): reduce noise vs dampen signal?
        composite = col_avg(metrics)
        normalised_composite = z_normalise(composite) #average of available for metrics and neutral imputation for factors
        factor_signals.append(normalised_composite) #[normalised_composite for each stock] for each factor (comparable)
    betas = [d["Beta"] for d in data]
    return factor_signals, betas

def get_benchmark(period, units_vol=1, units_corr=5, ticker="SPY"):
    def fetch_log_returns(tq, start, end, lag=1):
        hist = tq.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval='1d')
        if hist.empty:
            return pd.Series(dtype='float64')
        prices = hist.reset_index()[['date', 'adjclose']].dropna()
        prices['log_return'] = np.log(prices['adjclose'] / prices['adjclose'].shift(lag))
        return prices.set_index('date')['log_return'].dropna()
    
    benchmark = Ticker(ticker)
    start_date = period[0]
    end_date = period[1]
    start_vol = start_date - relativedelta(months=units_vol * 12)
    start_corr = start_date - relativedelta(months=units_corr * 12)
    vol_returns = fetch_log_returns(benchmark, start_vol, start_date, lag=1)
    corr_returns = fetch_log_returns(benchmark, start_corr, start_date, lag=3)
    start_price = get_close(benchmark, start_date)
    end_price = get_close(benchmark, end_date)
    performance = round(((end_price / start_price) - 1) * 100, 4)
    return {'vol': vol_returns, 'corr': corr_returns, 'performance': performance}

def get_beta(ticker, start_date, bench_returns, units_vol=1, units_corr=5): 
    def fetch_log_returns(tq, start, end, lag=1):
        hist = tq.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval='1d')
        if hist.empty:
            return pd.Series(dtype='float64')
        prices = hist.reset_index()[['date', 'adjclose']].dropna()
        prices['log_return'] = np.log(prices['adjclose'] / prices['adjclose'].shift(lag))
        return prices.set_index('date')['log_return'].dropna()

    start_vol = start_date - relativedelta(months=units_vol * 12)
    start_corr = start_date - relativedelta(months=units_corr * 12)
    stock_vol_ret = fetch_log_returns(ticker, start_vol, start_date, lag=1)
    stock_corr_ret = fetch_log_returns(ticker, start_corr, start_date, lag=3)
    if stock_vol_ret.empty or stock_corr_ret.empty: return np.nan
    bench_vol_ret = bench_returns['vol'].reindex(stock_vol_ret.index)
    bench_corr_ret = bench_returns['corr'].reindex(stock_corr_ret.index)
    vol_df = pd.DataFrame({'stock': stock_vol_ret, 'benchmark': bench_vol_ret}).dropna()
    corr_df = pd.DataFrame({'stock': stock_corr_ret, 'benchmark': bench_corr_ret}).dropna()

    min_obs_vol = config.min_obs_vol
    min_obs_corr = config.min_obs_corr
    if len(vol_df) < min_obs_vol or len(corr_df) < min_obs_corr:
        return np.nan
    corr = corr_df['stock'].corr(corr_df['benchmark'])
    vol_stock = vol_df['stock'].std()
    vol_bench = vol_df['benchmark'].std()
    raw_beta = corr * (vol_stock / vol_bench)
    w = config.w_shrink
    beta = w * raw_beta + (1 - w) * 1 #shrink towards industry median?
    return round(beta, 4)

def get_cum_score(data, weight, factor_signals, betas, baseline):
    def weight_normalise(signal, beta):
        signals = np.asarray(signal, dtype=np.float64)
        betas = np.asarray(beta, dtype=np.float64)
        mask = signals != 0
        if config.mode == "dollar":  
            adj = np.mean(signals[mask]) if np.any(mask) else 0.0
        else:
            masked_signals = signals[mask]
            masked_betas = betas[mask]
            beta_exposure = np.dot(masked_signals, masked_betas)
            beta_squared = np.dot(masked_betas, masked_betas)
            adj = (beta_exposure / beta_squared) * masked_betas if beta_squared != 0 else np.zeros_like(masked_betas)
        full_adj = np.zeros_like(signals)
        full_adj[mask] = adj
        neutralised = signals - full_adj #remove unwanted exposure
        if config.direction == "long":
            neutralised = np.where(neutralised > 0, neutralised, 0.0)
            if config.mode == "beta": final_weights = neutralised / (np.dot(neutralised, beta) or 1e-6)
            else: final_weights = neutralised / (np.sum(np.abs(neutralised)) or 1e-6)
        elif config.direction == "short":
            neutralised = np.where(neutralised < 0, neutralised, 0.0)
            if config.mode == "beta": final_weights = -neutralised / (np.dot(neutralised, beta) or 1e-6)
            else: final_weights = neutralised / (np.sum(np.abs(neutralised)) or 1e-6)
        else:
            final_weights = neutralised / (np.sum(np.abs(neutralised)) or 1e-6) ###add vol targetting instead of gross exposure = 1
        dollar_exposure = final_weights.sum()
        beta_exposure = np.dot(final_weights, beta)
        return final_weights, (dollar_exposure, beta_exposure)
    
    raw_signal = [
        sum(weight[j] * factor_signals[j][i] for j in range(len(factor_signals)))
        for i in range(len(data))
    ]
    if config.positions != -1:
        positions_to_keep = config.positions // 2
        sorted_indices = sorted(range(len(raw_signal)), key=lambda i: raw_signal[i])
        final_signal = [0] * len(raw_signal)
        for i in sorted_indices[:positions_to_keep] + sorted_indices[-positions_to_keep:]:
            final_signal[i] = raw_signal[i]
        raw_signal = final_signal
    normalised_signal, exposures = weight_normalise(raw_signal, betas)
    
    longs = 0
    shorts = 0
    for i, d in enumerate(data):
        w = normalised_signal[i]
        if config.mode=="equal": w = 1/len(data)
        if(w>0):
            longs += w * d["Performance"]
        elif(w<0):
            shorts += w * d["Performance"]
        d["Score"] = w
    longs*=100
    shorts*=100
    current_return = (longs+shorts)
    print(f"- Net Returns: {current_return:.2f}% [{longs:.2f}% + {shorts:.2f}%]")
    print(f"- Benchmark Returns: {baseline:.2f}%")
    print(f"- Dollar Exposure: {exposures[0]:.2f}")
    print(f"- Beta Exposure: {exposures[1]:.2f}")
    return current_return

def plot_results(data, intervals, img_path, labels=["Model"], benchmark=None):
    data = np.array(data)
    if benchmark is not None:
        labels.append("[Benchmark]")
        data = np.vstack([data, benchmark])
        
    multipliers = 1 + data / 100
    starting_capital = 1000
    x_labels = [f'Q{q}FY{year}' for q, year in intervals] + ['Today']
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    table_data = []
    column_labels = ["Label", "Net Return", "Correlation"]
    for i, series in enumerate(multipliers):
        corr =  np.corrcoef(series, benchmark)[0, 1] if np.std(series) != 0 and np.std(benchmark) != 0 else np.nan
        series = np.insert(series, 0, 1.0)
        cumulative = starting_capital * np.cumprod(series)
        time_steps = np.arange(len(cumulative))
        table_data.append([labels[i], round((cumulative[-1]/starting_capital - 1)*100, 2), round(corr, 2)])
        color = 'black' if labels[i] == "[Benchmark]" else None
        plt.plot(time_steps, cumulative, label=f'{labels[i]} Return: {(cumulative[-1]/starting_capital - 1)*100:.2f}%, Corr: {corr:.2f}', color=color) 
        print(f"Weight: {str(labels[i]):<5} | Total Return: ${int(cumulative[-1])} | Net Return: {(cumulative[-1]/starting_capital - 1)*100:.2f}% | Correlation: {corr:.2f}")

    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=90, ha='right')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(img_path, dpi=300)
    plt.show()

def save_csv(data, csv_path, FIELDS_1D, score=None): 
    df_final = None
    if score is None:
        final_z_metrics = []
        for d in data:
            record = {"Ticker": d["Ticker"], "Industry": d["Industry"], "Score": 0}
            for field in FIELDS_1D:
                value = d.get(field)
                if isinstance(value, (int, float)) and pd.notna(value):
                    record[field] = round(value, 4)
                else:
                    record[field] = value
            final_z_metrics.append(record)
        df_final = pd.DataFrame(final_z_metrics)
    else:
        try:
            df_final = pd.read_csv(csv_path)
        except:
            return 
        df_final["Score"] = np.zeros(len(df_final), dtype=float)
        df_final.loc[df_final["Ticker"].isin(score), "Score"] = df_final["Ticker"].map(score)
        df_final.sort_values(by='Score', ascending=False, inplace=True)
        
    if df_final is not None and len(df_final)!=0:
        df_final.to_csv(csv_path, index=False)
        print("SAVED updated results\n")











