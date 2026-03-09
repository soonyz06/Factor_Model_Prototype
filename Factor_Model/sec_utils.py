import pandas as pd
from yahooquery import Ticker
import numpy as np

def cap_floor(val, min_val, max_val): #Define bounds for val
    return max(min(val, max_val), min_val)

def get_latest(df, field, i=-1, fill=np.nan): 
    if field not in df.columns:
        return fill
    value = df[field].iloc[i]
    return value

def get_balance(bs, fields):
    total_value = 0
    for f in fields:
        if f in bs:
            val = bs.get(f, 0)
            total_value += val if pd.notna(val) else 0
    return total_value

def get_balance_sheet(bs, mode=1):
    cash = get_balance(bs, ['CashCashEquivalentsAndShortTermInvestments'])
    if cash == 0: get_balance(bs, ['CashAndCashEquivalents', 'ShortTermInvestments', 'OtherShortTermInvestments'])
    total_debt = get_balance(bs, ['CurrentDebt', 'LongTermDebt']) 
    mezzaine = get_balance(bs, ['MinorityInterest', 'PreferredStock'])
    equity = get_balance(bs, ['TotalEquityGrossMinorityInterest'])
    shares = get_balance(bs, ['OrdinarySharesNumber'])
    assets = equity + total_debt - cash
    if mode == 1: return cash, total_debt, mezzaine, equity, shares, assets
    if mode == 2: return assets
    if mode == 3: return equity
    
def calc_mult(n, d, mode=1, fill=999999, change=0, sign=1):
    return (n / d - change) * 100 * sign if (d>0 and (n>0 or mode==1)) else fill
        
def calc_IV(income_df, balance_df, fx_rate, L): 
    latest_bs = balance_df.iloc[-L]
    Revenue = get_latest(income_df, "OperatingRevenue", -L)
    #Interest = get_latest(income_df, "InterestExpense", -L)
    cash, total_debt, mezzaine, equity, shares, assets = get_balance_sheet(latest_bs)
    OperatingIncome = get_latest(income_df, "OperatingIncome", -L)
    if np.isnan(OperatingIncome):
        OperatingIncome = get_latest(income_df, "PretaxIncome", -L)
        cash, total_debt, mezzaine = 0, 0, 0
        #return [np.nan, np.nan, np.nan]

    rev_past = get_latest(income_df, 'OperatingRevenue', -(L+1))
    rev_now = Revenue
    if (rev_now>0 and rev_past>0):
        rev_growth = (rev_now/rev_past-1)
    else:
        return [np.nan, np.nan, np.nan]
    cur_margin = OperatingIncome/Revenue if Revenue >0 else -999
    roic = (OperatingIncome)*0.8/assets if assets>0 else 999
    e = ( len(str(max(int(Revenue*fx_rate), 0))) - 1 - 3 ) / 10
    ###gpoa instead of roe ?
    #debt_equity = liabilities/equity if equity>0 else 999
    #dol = Interest/OperatingIncome if OperatingIncome>0 else 999
    
    r_bound = 0.01
    discount = 0.05*3
    x = 0
    x -= cap_floor(rev_growth/100, -r_bound, r_bound) #Revenue
    x -= cap_floor(cur_margin/100, -r_bound, r_bound) #Margins
    x -= cap_floor(roic/100, -r_bound, r_bound) #Profitability
    x -= cap_floor(e/100, -r_bound, r_bound) #Size
    #x += cap_floor((debt_equity-1)/100, 0, r_bound) #Financial risk
    #x += cap_floor(dol/100, 0, r_bound) #Operating Leverage (fixed cost)
    discount += x*3
    
    values = [-1, -1, -1]
    rev_gs = [-1, -1, -1]
    margin_ds = [-1, -1, -1]
    rev_mins = [-0.03, 0.01, 0.05]
    rev_maxes = [0.02, 0.1, 0.18] 
    rev_mults = [0.1, 0.45, 0.8]
    margin_mins = [x/2 for x in rev_mins]
    margin_maxes = [x/2 for x in rev_maxes]
    margin_mults = [x/2 for x in rev_mults] #40 70 180
    n = 10
    for i in range(3):
        rev_min = rev_mins[i]
        rev_max = rev_maxes[i]
        rev_mult = rev_mults[i]
        margin_min = margin_mins[i]
        margin_max = margin_maxes[i]
        margin_mult = margin_mults[i]
        if rev_growth<0 and i==0: rev_mult = rev_mults[2] 
        if roic<0 and i==0: margin_mult = margin_mults[2]
        if(i==0): r = cap_floor(discount, 0.1, 0.2)
        elif(i==1): r = cap_floor(discount, 0.08, 0.15)
        elif(i==2): r = cap_floor(discount, 0.06, 0.1)
        
        rev_g = cap_floor(rev_growth*rev_mult, rev_min, rev_max) #Estimates revenue growth for next n years
        margin_d = cap_floor((roic*margin_mult/100)*n, margin_min, margin_max) #Estimates margin expansion for the next n years
        margin_g = margin_d/n
        rev_gs[i] = round(rev_g*100, 0)
        margin_ds[i] = round(margin_d*100, 0)
        #Growing Annuity
        ga = 0
        Margin = 0 if(cur_margin<0 and i==2) else cur_margin
        for t in range(1, n + 1):
            Revenue *= (1+rev_g) 
            Margin += margin_g
            if Margin>=1: Margin=0.99
            Interest = 0#*= (1+rev_g)
            cf_t = (Revenue * Margin - Interest)*.8 #Assuming 20% tax rate
            pv_t = cf_t / ((1 + r) ** t)
            ga += pv_t
            
        #Terminal Value
        g = i/100 
        C = cf_t*(1+g)
        tv = (C)/(r-g)
        tv = tv/((1+r)**(n))
        if(tv<0): tv=0
        npv = ga + tv
        npv = npv + cash - total_debt - mezzaine #use debt instead of interest
        npv*= fx_rate
        npv = npv/shares if shares>0 else np.nan
        values[i] = npv #Estimated intrinsic value of each share (Price)
    return values
