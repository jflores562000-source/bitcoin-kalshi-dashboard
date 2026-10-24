"""BTC 15-minute Kalshi research dashboard. Educational model, not a trading signal."""
import math
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title='BTC / Kalshi 15m Lab', layout='wide')
st.title('₿ Bitcoin / Kalshi 15-Minute Lab')
st.caption('Research dashboard • Live Coinbase spot prices • Kalshi market details entered manually • No automated trading')
BASE='https://api.exchange.coinbase.com'
HEAD={'User-Agent':'btc-kalshi-research/1.0','Accept':'application/json'}

@st.cache_data(ttl=25, show_spinner=False)
def ticker():
    r=requests.get(BASE+'/products/BTC-USD/ticker',headers=HEAD,timeout=12)
    r.raise_for_status()
    return float(r.json()['price'])

@st.cache_data(ttl=90, show_spinner=False)
def candles(days):
    # Coinbase Exchange REST returns up to 300 1-minute bars per request.
    end=datetime.now(timezone.utc).replace(second=0,microsecond=0)
    start=end-timedelta(days=days)
    rows=[]
    cursor=start
    while cursor<end:
        nxt=min(cursor+timedelta(minutes=299),end)
        r=requests.get(BASE+'/products/BTC-USD/candles',headers=HEAD,
            params={'granularity':60,'start':cursor.isoformat(),'end':nxt.isoformat()},timeout=20)
        if r.status_code==429:
            raise RuntimeError('Coinbase rate limit reached. Wait and retry with fewer days.')
        r.raise_for_status()
        rows.extend(r.json())
        cursor=nxt
        time.sleep(.12)
    if not rows: raise RuntimeError('No candle data returned.')
    df=pd.DataFrame(rows,columns=['unix','low','high','open','close','volume'])
    df['time']=pd.to_datetime(df['unix'],unit='s',utc=True)
    df=df.drop_duplicates('time').sort_values('time').set_index('time')
    for c in ['low','high','open','close','volume']: df[c]=pd.to_numeric(df[c])
    return df

def probability_up(price,target,minutes_left,sigma_per_sqrt_min,drift_per_min=0):
    if minutes_left<=0: return float(price>target)
    if price<=0 or target<=0: return float('nan')
    sd=max(sigma_per_sqrt_min*math.sqrt(minutes_left),1e-9)
    z=(math.log(price/target)+drift_per_min*minutes_left)/sd
    return .5*(1+math.erf(z/math.sqrt(2)))

def estimated_sigma(df,lookback):
    x=np.log(df['close']).diff().dropna().tail(lookback)
    return float(x.std(ddof=1)) if len(x)>3 else float('nan')

def fee_per_contract(price_cents, fee_rate=.07):
    # Kalshi's standard taker fee formula is often ceil(0.07 * P * (1-P) * 100) cents;
    # confirm actual fee schedule/market-specific fees. User can override below.
    p=price_cents/100
    return math.ceil(fee_rate*p*(1-p)*100-1e-12)

st.sidebar.header('Configuration')
days=st.sidebar.slider('Historical data (days)',1,7,2)
lookback=st.sidebar.select_slider('Volatility lookback (1-minute returns)',options=[15,30,60,120,240],value=60)
refresh=st.sidebar.button('↻ Refresh live data')
if refresh:
    ticker.clear(); candles.clear()
try:
    df=candles(days)
    spot=ticker()
except Exception as exc:
    st.error(f'Could not retrieve Coinbase data: {exc}. Internet access and Coinbase availability are required.')
    st.stop()

now=datetime.now(timezone.utc)
window_start=now.replace(minute=(now.minute//15)*15,second=0,microsecond=0)
window_end=window_start+timedelta(minutes=15)
remaining=max((window_end-now).total_seconds()/60,0)
sigma=estimated_sigma(df,lookback)
logret=np.log(df['close']).diff().dropna()
change5=(spot/float(df['close'].iloc[-6])-1)*100 if len(df)>6 else float('nan')
change15=(spot/float(df['close'].iloc[-16])-1)*100 if len(df)>16 else float('nan')

st.subheader('1. Live BTC monitor')
a,b,c,d=st.columns(4)
a.metric('Coinbase BTC/USD',f'${spot:,.2f}')
b.metric('5-minute move',f'{change5:+.3f}%')
c.metric('15-minute move',f'{change15:+.3f}%')
d.metric('Current UTC window ends',window_end.strftime('%H:%M UTC'),f'{remaining:.1f} min remaining',delta_color='off')
st.line_chart(df['close'].tail(120),height=250)
st.caption('Coinbase BTC-USD is a reference spot feed, NOT necessarily the exact price source, timestamp, or settlement rule used by Kalshi. Confirm the market rules before using any estimate.')

st.subheader('2. Kalshi target / probability model')
st.info('Copy the actual target, Up ask, Down ask, and settlement time from the current Kalshi market. Do not substitute the Coinbase opening price for the actual target.')
c1,c2,c3,c4=st.columns(4)
target=c1.number_input('Kalshi target (USD)',min_value=1.0,value=float(round(spot)),step=10.0,format='%.2f')
upask=c2.number_input('UP contract ask (¢)',min_value=1,max_value=99,value=50)
downask=c3.number_input('DOWN contract ask (¢)',min_value=1,max_value=99,value=50)
minutes=c4.number_input('Minutes until settlement',min_value=0.01,max_value=15.0,value=float(round(max(remaining,.01),2)),step=.25)
c5,c6,c7=st.columns(3)
volmult=c5.slider('Volatility stress multiplier',.5,3.0,1.0,.1)
feeup=c6.number_input('UP fee per contract (¢)',min_value=0.0,max_value=50.0,value=float(fee_per_contract(upask)),step=.1)
feedown=c7.number_input('DOWN fee per contract (¢)',min_value=0.0,max_value=50.0,value=float(fee_per_contract(downask)),step=.1)
pu=probability_up(spot,target,minutes,sigma*volmult)
pdwn=1-pu
breakeven_up=(upask+feeup)/100
breakeven_down=(downask+feedown)/100
result=pd.DataFrame([
 {'Outcome':'UP','Model probability':pu,'Ask ($)':upask/100,'Fee ($)':feeup/100,'Break-even probability':breakeven_up,'Model EV / contract ($)':pu-(upask+feeup)/100},
 {'Outcome':'DOWN','Model probability':pdwn,'Ask ($)':downask/100,'Fee ($)':feedown/100,'Break-even probability':breakeven_down,'Model EV / contract ($)':pdwn-(downask+feedown)/100},
])
st.dataframe(result.style.format({'Model probability':'{:.1%}','Ask ($)':'${:.2f}','Fee ($)':'${:.2f}','Break-even probability':'{:.1%}','Model EV / contract ($)':'${:+.3f}'}),hide_index=True,use_container_width=True)
st.caption('Model: zero-drift lognormal/Brownian approximation using trailing 1-minute realized volatility. This is NOT a validated forecasting edge. Positive model EV does not establish real profitability; spreads, fees, settlement feed differences, latency and model error matter.')

st.subheader('3. Walk-forward historical test')
st.write('For each completed 15-minute UTC window, we approximate the newly established target with the first available 1-minute candle close, then make a prediction 5 minutes into the window using only past data. This is a **proxy test**, not a backtest of actual Kalshi quotes or target rules.')
@st.cache_data(show_spinner=False)
def backtest(data,lookback,threshold):
    # Index is minute candle open; close is the end-of-minute observed price.
    closes=data['close'].copy()
    idx=closes.index
    ret=np.log(closes).diff()
    records=[]
    starts=pd.date_range(idx.min().ceil('15min'),idx.max().floor('15min'),freq='15min',tz='UTC')
    for start in starts:
        # Target proxy = close of first minute; prediction after minute 5; settlement = close of minute 15.
        t0=start; tp=start+timedelta(minutes=4); te=start+timedelta(minutes=14)
        if not all(t in closes.index for t in (t0,tp,te)): continue
        past=ret.loc[:tp].tail(lookback).dropna()
        if len(past)<lookback: continue
        # Avoid windows with missing minute candles in the volatility lookback.
        if len(ret.loc[tp-timedelta(minutes=lookback-1):tp])!=lookback: continue
        s=float(past.std(ddof=1))
        if not np.isfinite(s) or s<=0: continue
        target=float(closes.loc[t0]); current=float(closes.loc[tp]); settle=float(closes.loc[te])
        p=probability_up(current,target,10,s)
        outcome=int(settle>target)
        predicted=int(p>=threshold) if p>=threshold else (0 if p<=1-threshold else None)
        records.append({'window_start':start,'target_proxy':target,'price_at_prediction':current,'settlement_proxy':settle,'p_up':p,'up_actual':outcome,'prediction':predicted,'correct':int(predicted==outcome) if predicted is not None else np.nan,'brier':(p-outcome)**2})
    return pd.DataFrame(records)
threshold=st.slider('Confidence threshold for issuing a directional prediction',.50,.90,.60,.05)
bt=backtest(df,lookback,threshold)
if bt.empty:
    st.warning('Not enough complete data for a backtest. Try a longer history or shorter lookback.')
else:
    bets=bt.dropna(subset=['correct'])
    a,b,c,d=st.columns(4)
    a.metric('Completed proxy windows',len(bt))
    b.metric('Predictions made',len(bets))
    c.metric('Accuracy when predicted',f'{bets.correct.mean():.1%}' if len(bets) else 'N/A')
    d.metric('Brier score (lower better)',f'{bt.brier.mean():.3f}')
    st.caption(f'Coverage: {len(bets)/len(bt):.1%} of windows. A 50/50 probability forecast has Brier score 0.250. Accuracy is conditional on issuing a prediction, and is NOT profitability.')
    bt['probability_bucket']=pd.cut(bt.p_up,bins=np.linspace(0,1,11),include_lowest=True)
    calibration=bt.groupby('probability_bucket',observed=False).agg(n=('up_actual','size'),predicted_up=('p_up','mean'),observed_up=('up_actual','mean')).reset_index()
    st.write('Probability calibration by bucket')
    st.dataframe(calibration,hide_index=True,use_container_width=True)
    st.write('Most recent completed windows')
    st.dataframe(bt.tail(30).iloc[::-1],hide_index=True,use_container_width=True)
    st.download_button('Download proxy backtest CSV',bt.to_csv(index=False).encode(),file_name='btc_15m_proxy_backtest.csv',mime='text/csv')

st.subheader('4. Optional actual Kalshi trade log')
st.write('Upload your own historical trade log to measure REAL net profit/loss. CSV columns: `outcome` (UP or DOWN), `won` (1 or 0), `ask_cents`, `fee_cents`, `contracts`. Include only settled positions. Fees should include the actual amount paid.')
up=st.file_uploader('Historical Kalshi positions CSV',type=['csv'])
if up is not None:
    try:
        trades=pd.read_csv(up)
        required={'outcome','won','ask_cents','fee_cents','contracts'}
        if not required.issubset(trades.columns): raise ValueError('Missing columns: '+', '.join(sorted(required-set(trades.columns))))
        for k in ['won','ask_cents','fee_cents','contracts']: trades[k]=pd.to_numeric(trades[k],errors='raise')
        if not trades['won'].isin([0,1]).all(): raise ValueError('won must be 0 or 1')
        if (trades[['ask_cents','fee_cents','contracts']]<0).any().any(): raise ValueError('Negative values not allowed')
        trades['net_usd']=trades['contracts']*(100*trades['won']-trades['ask_cents']-trades['fee_cents'])/100
        a,b,c=st.columns(3)
        a.metric('Actual net P/L',f'${trades.net_usd.sum():+,.2f}')
        b.metric('Trade win rate',f'{trades.won.mean():.1%}')
        c.metric('Trades',len(trades))
        st.line_chart(trades.net_usd.cumsum())
        st.dataframe(trades,use_container_width=True,hide_index=True)
    except Exception as exc:
        st.error(f'Could not parse trade log: {exc}')

st.warning('Research only. No prediction is certain. In particular, increasing stake sizes after losses does not create a statistical edge and can cause rapid bankroll depletion.')
