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
st.caption('Research dashboard • Live Coinbase spot prices • Kalshi market data auto-loaded when available • No automated trading') 
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

KALSHI_API = 'https://external-api.kalshi.com/trade-api/v2'

def _number(value):
    try:
        if value is None or value == '': return None
        return float(value)
    except (TypeError, ValueError): return None

def _cents(m, dollar_key, cent_key):
    v = _number(m.get(dollar_key))
    if v is not None: return round(v * 100, 2)
    return _number(m.get(cent_key))

@st.cache_data(ttl=12, show_spinner=False)
def current_kalshi():
    """Public, read-only Kalshi market discovery; no login or trading keys."""
    response = requests.get(KALSHI_API + '/markets',
        params={'series_ticker':'KXBTC15M','status':'open','limit':100},
        headers=HEAD, timeout=12)
    response.raise_for_status()
    markets = response.json().get('markets', [])
    utc_now = datetime.now(timezone.utc)
    eligible=[]
    for m in markets:
        try:
            close=datetime.fromisoformat(m['close_time'].replace('Z','+00:00'))
            opened=datetime.fromisoformat(m['open_time'].replace('Z','+00:00')) if m.get('open_time') else None
        except (KeyError, ValueError, TypeError): continue
        if close <= utc_now or (opened and opened > utc_now): continue
        strike=_number(m.get('floor_strike'))
        if strike is None or strike<=0: continue
        eligible.append((close,m,strike))
    if not eligible: raise ValueError('No active market with a published target was returned. Try refreshing in a moment.')
    close,m,strike=min(eligible,key=lambda item:item[0])
    return {'ticker':m['ticker'],'target':strike,'close':close.isoformat(),
            'upask':_cents(m,'yes_ask_dollars','yes_ask'),
            'downask':_cents(m,'no_ask_dollars','no_ask'),
            'retrieved':utc_now.isoformat()}

st.sidebar.header('Configuration')
days=st.sidebar.slider('Historical data (days)',1,7,2)
lookback=st.sidebar.select_slider('Volatility lookback (1-minute returns)',options=[15,30,60,120,240],value=60)
refresh=st.sidebar.button('↻ Refresh live data')
if refresh:
    ticker.clear(); candles.clear(); current_kalshi.clear()
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

st.subheader('2. Live Kalshi target / probability model')
auto = st.toggle('Automatically load current Kalshi market', value=True)
market = None
if auto:
    try:
        market=current_kalshi()
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(market['retrieved'])).total_seconds()
        st.success(f"Market: {market['ticker']} • Snapshot age: {age:.0f} seconds • Data from Kalshi public API")
    except Exception as exc:
        st.warning(f'Kalshi auto-load unavailable: {exc}. Turn off auto-load for manual entry.')
if market:
    target=market['target']
    upask=market['upask']
    downask=market['downask']
    minutes=max(0,(datetime.fromisoformat(market['close'])-datetime.now(timezone.utc)).total_seconds()/60)
    a,b,c,d=st.columns(4)
    a.metric('Kalshi target',f'${target:,.2f}')
    b.metric('UP ask',f'{upask:g}¢' if upask is not None else 'Unavailable')
    c.metric('DOWN ask',f'{downask:g}¢' if downask is not None else 'Unavailable')
    d.metric('Minutes remaining',f'{minutes:.2f}')
    st.caption('Refresh live data in the sidebar for a new snapshot. This is NOT a continuously streaming quote. Verify the ticker and settlement rules in Kalshi.')
else:
    c1,c2,c3,c4=st.columns(4)
    target=c1.number_input('Kalshi target (USD)',min_value=1.0,value=float(round(spot)),step=10.0,format='%.2f')
    upask=c2.number_input('UP contract ask (¢)',min_value=1,max_value=99,value=50)
    downask=c3.number_input('DOWN contract ask (¢)',min_value=1,max_value=99,value=50)
    minutes=c4.number_input('Minutes until settlement',min_value=0.01,max_value=15.0,value=float(round(max(remaining,.01),2)),step=.25)
volmult=st.slider('Volatility stress multiplier',.5,3.0,1.0,.1)
if upask is None or downask is None or minutes<=0:
    st.warning('Quotes are unavailable or this market has ended. Probability comparison is disabled; refresh the market.')
else:
    st.caption('Fee estimates below use a simplified standard fee formula; verify the exact Kalshi fee for your order in the trade preview.')
    feeup=fee_per_contract(upask)
    feedown=fee_per_contract(downask)
    pu=probability_up(spot,target,minutes,sigma*volmult)
    pdwn=1-pu
    result=pd.DataFrame([
     {'Outcome':'UP','Model probability':pu,'Ask ($)':upask/100,'Estimated fee ($)':feeup/100,'Break-even probability':(upask+feeup)/100,'Model EV / contract ($)':pu-(upask+feeup)/100},
     {'Outcome':'DOWN','Model probability':pdwn,'Ask ($)':downask/100,'Estimated fee ($)':feedown/100,'Break-even probability':(downask+feedown)/100,'Model EV / contract ($)':pdwn-(downask+feedown)/100},
    ])
    st.dataframe(result.style.format({'Model probability':'{:.1%}','Ask ($)':'${:.2f}','Estimated fee ($)':'${:.2f}','Break-even probability':'{:.1%}','Model EV / contract ($)':'${:+.3f}'}),hide_index=True,use_container_width=True)
    st.warning('Model probabilities are UNVALIDATED. Coinbase spot differs from Kalshi’s settlement reference; apparent positive EV may be entirely spurious. Do not treat this table as a trading signal.')

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
