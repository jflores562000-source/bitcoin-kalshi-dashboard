"""BTC 15-minute Kalshi research dashboard. Educational model, not a trading signal."""
import math
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import tracker
import auto_sell_research

st.set_page_config(page_title='BTC / Kalshi 15m Lab', layout='wide', initial_sidebar_state='collapsed')
st.title('₿ Bitcoin / Kalshi 15-Minute Lab')
st.caption('Bitcoin direction research • live summary first • detailed research in the second tab')
BASE='https://api.exchange.coinbase.com'
HEAD={'User-Agent':'btc-kalshi-research/1.0','Accept':'application/json'}

@st.cache_data(ttl=3, show_spinner=False)
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

@st.cache_data(ttl=3, show_spinner=False)
def current_kalshi():
    """Discover the current 15-minute event by Eastern-time event ticker.

    If the API does not publish a target or executable quote, fail closed.
    """
    from zoneinfo import ZoneInfo
    now = datetime.now(timezone.utc)
    et = now.astimezone(ZoneInfo('America/New_York'))
    start = et.replace(minute=(et.minute // 15) * 15, second=0, microsecond=0)
    diagnostics = []
    for offset in (0, -15, 15):
        candidate = start + timedelta(minutes=offset)
        event_ticker = 'KXBTC15M-' + candidate.strftime('%y%b%d%H%M').upper()
        try:
            r = requests.get(KALSHI_API + '/events/' + event_ticker,
                params={'with_nested_markets':'true'}, headers=HEAD, timeout=12)
            if r.status_code == 404:
                diagnostics.append(event_ticker + ': not found')
                continue
            r.raise_for_status()
            payload = r.json()
            markets = payload.get('markets') or payload.get('event', {}).get('markets') or []
            for m in markets:
                close_str = m.get('close_time')
                if not close_str: continue
                close = datetime.fromisoformat(close_str.replace('Z','+00:00'))
                opened = datetime.fromisoformat(m['open_time'].replace('Z','+00:00')) if m.get('open_time') else None
                if close <= now or (opened and opened > now): continue
                strike = _number(m.get('floor_strike'))
                if strike is None:
                    # Some binary events publish the threshold in the subtitle.
                    import re
                    subtitle = str(m.get('yes_sub_title') or m.get('no_sub_title') or '')
                    match = re.search(r'Target\s*price\s*:\s*\$?([\d,]+(?:\.\d+)?)', subtitle, re.I)
                    if match: strike = _number(match.group(1).replace(',',''))
                upask = _cents(m,'yes_ask_dollars','yes_ask')
                downask = _cents(m,'no_ask_dollars','no_ask')
                upbid = _cents(m,'yes_bid_dollars','yes_bid')
                downbid = _cents(m,'no_bid_dollars','no_bid')
                if not strike or strike <= 0:
                    diagnostics.append(event_ticker + ': target unpublished')
                    continue
                if upask is None or downask is None or not (0 < upask < 100 and 0 < downask < 100):
                    diagnostics.append(event_ticker + ': quotes unavailable')
                    continue
                return {'ticker':m['ticker'],'target':strike,'close':close.isoformat(),
                        'upask':upask,'downask':downask,'upbid':upbid,'downbid':downbid,'retrieved':now.isoformat()}
            if not markets: diagnostics.append(event_ticker + ': no nested markets')
        except (requests.RequestException, ValueError, KeyError) as exc:
            diagnostics.append(event_ticker + ': ' + str(exc)[:120])
    raise ValueError('No verified live market found. ' + '; '.join(diagnostics))

st.sidebar.header('Configuration')
st.sidebar.caption('Kalshi market and reference spot poll every 3 seconds while the page is open. Historical candles remain cached.')
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

# Two tabs keep the live question visible without deleting prior research.
simple_tab, research_tab = st.tabs(['🎯 Shortcut / Live', '📊 Research & history'])

with simple_tab:
    @st.fragment(run_every='10s')
    def simple_direction_view():
        st.subheader('Current 15-minute Bitcoin cycle')
        try:
            market = current_kalshi()
            price = ticker()
            target = float(market['target'])
            end = datetime.fromisoformat(market['close'])
            minutes_left = (end - datetime.now(timezone.utc)).total_seconds()/60
            if not (0 < minutes_left <= 15 and target > 0):
                raise ValueError('No active market with a valid target and closing time')
            # Use at most the last 60 minutes rather than an aged on-page snapshot.
            recent = candles(days)
            last_age = (datetime.now(timezone.utc) - recent.index[-1].to_pydatetime()).total_seconds()
            if last_age > 240:
                raise ValueError('Recent Coinbase history is stale')
            v15 = estimated_sigma(recent, 15)
            v60 = estimated_sigma(recent, 60)
            if not (math.isfinite(v15) and math.isfinite(v60) and v15 > 0 and v60 > 0):
                raise ValueError('Not enough recent volatility observations')
            vol = 0.6*v15 + 0.4*v60
            gap = price-target
            # Exactly the same experimental probability method as the detailed screen.
            if minutes_left < 1:
                touch_p = float('nan')
                opposite_p = float('nan')
            else:
                touch_z = abs(math.log(price/target))/(vol*math.sqrt(minutes_left))
                touch_p = math.erfc(touch_z/math.sqrt(2))
                opposite_z = abs(math.log(price/target))/(vol*math.sqrt(minutes_left-2/3))
                opposite_p = 0.5*math.erfc(opposite_z/math.sqrt(2))
            p_up = (1-opposite_p) if gap>0 else opposite_p
            p_down = 1-p_up
            current_side = 'UP' if gap>0 else ('DOWN' if gap<0 else 'AT TARGET')
            a,b,c = st.columns(3)
            a.metric('Time remaining', f'{minutes_left:.1f} min')
            b.metric('Price versus target', f'${gap:+,.2f}')
            c.metric('Current side', current_side)
            st.divider()
            if not (math.isfinite(p_up) and math.isfinite(p_down)) or gap == 0:
                st.markdown('### ⚪ No dependable direction estimate')
                st.info('This simplified model cannot produce a defensible estimate within the final minute or exactly at the target.')
            else:
                lean = 'UP' if p_up >= p_down else 'DOWN'
                st.markdown(f'### Model currently leans **{lean}**')
                left, right, cross = st.columns(3)
                left.metric('Finish UP (model)', f'{p_up:.1%}')
                right.metric('Finish DOWN (model)', f'{p_down:.1%}')
                cross.metric('Touch target (model)', f'{touch_p:.1%}')
                if abs(gap) < 25 or max(p_up,p_down) < .65:
                    st.info('Price is close to the target or the model is uncertain; outcome could change easily.')
                st.caption('The model direction always matches the current Coinbase price relative to the target. It has NOT yet shown predictive skill beyond this simple baseline.')
            st.caption(f'Coinbase reference: ${price:,.2f}  •  Kalshi target: ${target:,.2f}  •  Live market: {market["ticker"]}')
            st.warning('EXPERIMENTAL, NOT VERIFIED ODDS. Coinbase is not Kalshi’s CF Benchmarks BRTI settlement index. These percentages are not a buy/sell recommendation.')
            st.caption('Forecast accuracy is being evaluated at 12, 9, 6 and 3 minutes in the Research & history tab. Keep the browser active to record checkpoints.')
        except Exception as exc:
            st.info(f'Live direction is temporarily unavailable: {exc}. No prediction is shown without current market data.')

    simple_direction_view()

    # Section 2 appears on the shortcut tab.
    @st.fragment(run_every="3s")
    def live_kalshi_panel():
        st.subheader('2. Live Kalshi target / probability model')
        auto = st.toggle('Automatically load current Kalshi market', value=True)
        market = None
        if auto:
            try:
                market=current_kalshi()
                age=(datetime.now(timezone.utc)-datetime.fromisoformat(market['retrieved'])).total_seconds()
                st.success(f"Market: {market['ticker']} • Snapshot age: {age:.0f} seconds • Data from Kalshi public API")
            except Exception as exc:
                st.warning(f'Kalshi auto-load unavailable: {exc}. No live quote or target is available. Turn off auto-load only to enter independently verified values.')
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
            st.caption('Auto-refresh every 3 seconds while this page is open; this is polling, not a streaming feed. Quotes can change between updates. Verify market ticker and settlement rules on Kalshi.')
        else:
            c1,c2,c3,c4=st.columns(4)
            target=c1.number_input('Kalshi target (USD)',min_value=1.0,value=float(round(spot)),step=10.0,format='%.2f')
            upask=c2.number_input('UP contract ask (¢)',min_value=1,max_value=99,value=50)
            downask=c3.number_input('DOWN contract ask (¢)',min_value=1,max_value=99,value=50)
            minutes=c4.number_input('Minutes until settlement',min_value=0.01,max_value=15.0,value=float(round(max(remaining,.01),2)),step=.25)
        volmult=st.slider('Volatility stress multiplier',.5,3.0,1.0,.1)
        if (auto and market is None) or upask is None or downask is None or minutes<=0:
            st.warning('Verified live market data unavailable. Probability comparison is disabled until a valid Kalshi market is loaded or you switch to manual mode.')
        else:
            st.caption('Fee estimates below use a simplified standard fee formula; verify the exact Kalshi fee for your order in the trade preview.')
            feeup=fee_per_contract(upask)
            feedown=fee_per_contract(downask)
            try:
                live_spot=ticker()
            except Exception:
                live_spot=spot
                st.warning('Coinbase reference spot refresh failed; showing last available spot. Probability model may be stale.')
            pu=probability_up(live_spot,target,minutes,sigma*volmult)
            pdwn=1-pu
            result=pd.DataFrame([
             {'Outcome':'UP','Model probability':pu,'Ask ($)':upask/100,'Estimated fee ($)':feeup/100,'Break-even probability':(upask+feeup)/100,'Model EV / contract ($)':pu-(upask+feeup)/100},
             {'Outcome':'DOWN','Model probability':pdwn,'Ask ($)':downask/100,'Estimated fee ($)':feedown/100,'Break-even probability':(downask+feedown)/100,'Model EV / contract ($)':pdwn-(downask+feedown)/100},
            ])
            st.dataframe(result.style.format({'Model probability':'{:.1%}','Ask ($)':'${:.2f}','Estimated fee ($)':'${:.2f}','Break-even probability':'{:.1%}','Model EV / contract ($)':'${:+.3f}'}),hide_index=True,use_container_width=True)
            st.warning('Model probabilities are UNVALIDATED. Coinbase spot differs from Kalshi’s settlement reference; apparent positive EV may be entirely spurious. Do not treat this table as a trading signal.')
        # Log one fixed-horizon forecast per market; only explicit Kalshi final results count.
        tracker.render(market if auto else None, live_spot if 'live_spot' in locals() else spot,
                       sigma*volmult, probability_up, fee_per_contract)


    live_kalshi_panel()


    @st.fragment(run_every='10s')
    def auto_sell_panel():
        try:
            quote_market = current_kalshi()
        except Exception:
            quote_market = None
        auto_sell_research.render(quote_market)

    auto_sell_panel()

with research_tab:
    st.subheader('1. Live BTC monitor')
    a,b,c,d=st.columns(4)
    a.metric('Coinbase BTC/USD',f'${spot:,.2f}')
    b.metric('5-minute move',f'{change5:+.3f}%')
    c.metric('15-minute move',f'{change15:+.3f}%')
    d.metric('Current UTC window ends',window_end.strftime('%H:%M UTC'),f'{remaining:.1f} min remaining',delta_color='off')
    st.line_chart(df['close'].tail(120),height=250)
    st.caption('Coinbase BTC-USD is a reference spot feed, NOT necessarily the exact price source, timestamp, or settlement rule used by Kalshi. Confirm the market rules before using any estimate.')

    # -----------------------------------------------------------------------------
    # Live target crossing estimates. These use actual market TARGET from Kalshi,
    # but a Coinbase spot price and simplified stochastic model, NOT the BRTI index.
    # They are research estimates, not validated real-world settlement probabilities.
    # -----------------------------------------------------------------------------
    def _model_target_odds(reference_price, target, minutes_left, volatility_1m):
        """Approximate probability of touching strike and settling on opposite side.

        Under a driftless log Brownian model with minute-scale log-volatility:
          P(touch) = 2 Phi(-abs(log(S/K)) / (sigma*sqrt(T)))
        The final-minute settlement approximation uses the variance of the mean
        Brownian level over [T-1, T], which is sigma^2*(T-2/3). This approximates
        the 60-second average of a spot/index, not actual CF BRTI settlement.
        """
        values = (reference_price, target, minutes_left, volatility_1m)
        if not all(math.isfinite(x) for x in values):
            return (float('nan'), float('nan'))
        if reference_price <= 0 or target <= 0 or minutes_left <= 0 or volatility_1m <= 0:
            return (float('nan'), float('nan'))
        log_gap = abs(math.log(reference_price / target))
        touch_z = log_gap / (volatility_1m * math.sqrt(minutes_left))
        p_touch = math.erfc(touch_z / math.sqrt(2))
        if minutes_left < 1.0:
            # Previous BRTI values, which we don't have, already affect the final
            # 60-second average; do not invent a settlement probability.
            return (p_touch, float('nan'))
        avg_z = log_gap / (volatility_1m * math.sqrt(minutes_left - 2/3))
        p_opposite = 0.5 * math.erfc(avg_z / math.sqrt(2))
        return (min(1.0,max(0.0,p_touch)), min(1.0,max(0.0,p_opposite)))


    @st.fragment(run_every='10s')
    def live_target_crossing_panel():
        st.subheader('2A. Live Kalshi target-crossing estimates (experimental)')
        st.caption('Two different events: touching the target ANY time before close, versus FINISHING on the opposite side. These probabilities are a simplified model, not validated Kalshi odds.')
        try:
            mkt = current_kalshi()
            price = ticker()
            target = float(mkt['target'])
            close_time = datetime.fromisoformat(mkt['close'])
            remaining_live = (close_time - datetime.now(timezone.utc)).total_seconds() / 60
            if not 0 < remaining_live <= 15:
                raise ValueError('Current market is no longer active')
            hist_candles = candles(days)
            candle_age = (datetime.now(timezone.utc) - hist_candles.index[-1].to_pydatetime()).total_seconds()
            if candle_age > 240:
                raise ValueError('Historical volatility feed is stale')
            # Recent volatility reacts to conditions; 60m is the steadier reference.
            vol_recent = estimated_sigma(hist_candles, 15)
            vol_reference = estimated_sigma(hist_candles, max(60,lookback))
            if not (math.isfinite(vol_recent) and math.isfinite(vol_reference) and vol_recent > 0 and vol_reference > 0):
                raise ValueError('Not enough valid recent volatility observations')
            # A simple transparent blend; calibration must be checked out of sample.
            vol_base = 0.6 * vol_recent + 0.4 * vol_reference
            delta = price - target
            touch, opposite = _model_target_odds(price,target,remaining_live,vol_base)
            state_side = 'UP' if delta > 0 else 'DOWN' if delta < 0 else 'AT TARGET'
            opposite_side = 'DOWN' if delta > 0 else 'UP' if delta < 0 else 'EITHER'
            a,b,c = st.columns(3)
            a.metric('Coinbase vs. Kalshi target', f'${delta:+,.2f}')
            b.metric('Time left', f'{remaining_live:.1f} min')
            c.metric('Currently above/below', state_side)
            d,e = st.columns(2)
            d.metric('Model: touch or cross target', f'{touch:.1%}' if math.isfinite(touch) else 'Unavailable')
            e.metric(f'Model: finish {opposite_side}', f'{opposite:.1%}' if math.isfinite(opposite) else 'Unavailable')
            if remaining_live < 1:
                st.warning('Less than 60 seconds remain. The model cannot estimate the final 60-second benchmark average from spot alone, so settlement probability is unavailable.')
            vol_quiet = min(vol_recent,vol_reference)
            vol_active = max(vol_recent,vol_reference)
            quiet_touch, quiet_settle = _model_target_odds(price,target,remaining_live,vol_quiet)
            active_touch, active_settle = _model_target_odds(price,target,remaining_live,vol_active)
            if math.isfinite(quiet_settle) and math.isfinite(active_settle):
                st.caption(f'Volatility sensitivity: with recent 15m versus 60m volatility, modeled FINISH-OPPOSITE estimates range {min(quiet_settle,active_settle):.1%}–{max(quiet_settle,active_settle):.1%}. This is not a statistical confidence interval.')
            st.write('**If the present price/target gap stayed the same at each checkpoint:**')
            rows=[]
            for horizon in (12,9,6,3):
                if horizon > remaining_live + 0.1:
                    continue   # Only future states that may still occur in this cycle.
                pt,po = _model_target_odds(price,target,float(horizon),vol_base)
                rows.append({'Minutes remaining': horizon,
                             'Touch/cross target (model)': f'{pt:.1%}',
                             f'Finish {opposite_side} (model)': f'{po:.1%}'})
            if rows:
                st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            else:
                st.caption('Next checkpoint is below 3 minutes. Use current estimate above.')
            st.warning('EXPERIMENTAL / NOT CALIBRATED: Live target is from Kalshi, but starting price and volatility are from Coinbase. Kalshi BTC 15m contracts use the last-minute average of CF Benchmarks BRTI, not Coinbase spot. The model assumes driftless, smoothly distributed changes and can be badly wrong during jumps or index discrepancies. Treat as research, not a wager recommendation.')
            st.caption(f'Kalshi market: {mkt["ticker"]}. Coinbase-based 1-minute volatility: recent 15m = {100*vol_recent:.3f}%/√min; reference 60m = {100*vol_reference:.3f}%/√min. Model refreshes while page remains active.')
        except Exception as exc:
            st.info(f'Live target-crossing model unavailable: {exc}. The historical research further down the page is unaffected.')


    live_target_crossing_panel()

    # -----------------------------------------------------------------------------
    # Live 12/9/6/3-minute forecast audit — independent from legacy tracker.py.
    # Uses the SAME existing GitHub secret credentials, with a SEPARATE JSON log.
    # Only the explicit published Kalshi result is considered a verified outcome.
    # -----------------------------------------------------------------------------
    import base64
    import json
    from pathlib import Path

    CHECKPOINTS = (12, 9, 6, 3)
    CHECKPOINT_LOG_PATH = 'data/kalshi_checkpoint_predictions.json'
    CHECKPOINT_FALLBACK = Path('/tmp/kalshi_checkpoint_predictions.json')
    GITHUB_API_ROOT = 'https://api.github.com'


    def _checkpoint_credentials():
        try:
            token = str(st.secrets.get('GITHUB_LOG_TOKEN', ''))
            repo = str(st.secrets.get('GITHUB_LOG_REPO', ''))
            return token, repo
        except Exception:
            return '', ''


    def _checkpoint_store_read():
        token, repo = _checkpoint_credentials()
        if token and repo:
            response = requests.get(
                f'{GITHUB_API_ROOT}/repos/{repo}/contents/{CHECKPOINT_LOG_PATH}',
                params={'ref': 'main'},
                headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'},
                timeout=12)
            if response.status_code == 404:
                return [], None, 'GitHub (persistent)'
            response.raise_for_status()
            obj = response.json()
            rows = json.loads(base64.b64decode(obj['content']).decode('utf-8'))
            if not isinstance(rows, list):
                raise ValueError('Checkpoint log is not a JSON list')
            return rows, obj['sha'], 'GitHub (persistent)'
        if CHECKPOINT_FALLBACK.exists():
            rows = json.loads(CHECKPOINT_FALLBACK.read_text())
            if not isinstance(rows, list):
                raise ValueError('Local checkpoint log is not a JSON list')
            return rows, None, 'Temporary server file (not persistent)'
        return [], None, 'Temporary server file (not persistent)'


    def _checkpoint_store_write(rows, sha):
        token, repo = _checkpoint_credentials()
        if token and repo:
            body = {
                'message': 'Update Bitcoin checkpoint forecast audit',
                'branch': 'main',
                'content': base64.b64encode(json.dumps(rows, allow_nan=False, indent=2).encode()).decode(),
            }
            if sha:
                body['sha'] = sha
            result = requests.put(
                f'{GITHUB_API_ROOT}/repos/{repo}/contents/{CHECKPOINT_LOG_PATH}',
                headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'},
                json=body, timeout=15)
            result.raise_for_status()
        else:
            CHECKPOINT_FALLBACK.write_text(json.dumps(rows, allow_nan=False, indent=2))


    def _checkpoint_row(mkt, spot_price, vol, time_utc, checkpoint):
        """Make a pre-settlement snapshot; never reconstruct a missed checkpoint."""
        close_utc = datetime.fromisoformat(mkt['close'])
        actual_minutes = (close_utc - time_utc).total_seconds() / 60
        if not checkpoint - 0.60 <= actual_minutes <= checkpoint:
            return None
        target = float(mkt['target'])
        if not (math.isfinite(spot_price) and math.isfinite(target) and
                math.isfinite(vol) and vol > 0 and spot_price > 0 and target > 0):
            return None
        if spot_price == target:
            return None
        p_touch, p_opposite = _model_target_odds(spot_price, target, actual_minutes, vol)
        if not (math.isfinite(p_touch) and math.isfinite(p_opposite)):
            return None
        above = spot_price > target
        p_up = 1 - p_opposite if above else p_opposite
        return {
            'ticker': str(mkt['ticker']),
            'checkpoint_minutes': int(checkpoint),
            'observed_utc': time_utc.isoformat(),
            'close_utc': close_utc.isoformat(),
            'actual_minutes_remaining': round(actual_minutes, 3),
            'kalshi_target': round(target, 2),
            'coinbase_spot': round(float(spot_price), 2),
            'distance_from_target': round(spot_price - target, 2),
            'model_volatility_1m': round(float(vol), 9),
            'model_p_up': round(p_up, 6),
            'model_p_touch': round(p_touch, 6),
            'spot_side_up': int(above),
            'actual_up': None,
            'settlement_checked_utc': None,
        }


    def _checkpoint_settlement(ticker):
        """Only use final YES/NO as published by the Kalshi public API."""
        response = requests.get(
            f'https://external-api.kalshi.com/trade-api/v2/markets/{ticker}',
            headers=HEAD, timeout=12)
        response.raise_for_status()
        outcome = str(response.json().get('market', {}).get('result') or '').lower()
        if outcome == 'yes':
            return 1
        if outcome == 'no':
            return 0
        return None


    def _checkpoint_update(rows, market, spot_price, vol, now_utc):
        """Return changed rows and an explanation; isolated for deterministic tests."""
        changed = False
        notes = []
        if market is not None:
            for checkpoint in CHECKPOINTS:
                if any(r.get('ticker') == market['ticker'] and
                       r.get('checkpoint_minutes') == checkpoint for r in rows):
                    continue
                new_row = _checkpoint_row(market, spot_price, vol, now_utc, checkpoint)
                if new_row is not None:
                    rows.append(new_row)
                    changed = True
                    notes.append(f'Saved {checkpoint}-minute forecast')

        # Resolve at most 3 DISTINCT markets per refresh, once final results are published.
        # One API read can resolve all four checkpoint observations for a market.
        unresolved = {}
        for item in rows:
            if item.get('actual_up') is not None or not item.get('close_utc'):
                continue
            close_utc = datetime.fromisoformat(item['close_utc'])
            if now_utc < close_utc + timedelta(seconds=60):
                continue
            last = item.get('settlement_checked_utc')
            if last and (now_utc - datetime.fromisoformat(last)).total_seconds() < 180:
                continue
            unresolved[item['ticker']] = True
        for market_ticker in list(unresolved)[:3]:
            try:
                outcome = _checkpoint_settlement(market_ticker)
            except requests.RequestException:
                continue
            for item in rows:
                if item.get('ticker') != market_ticker or item.get('actual_up') is not None:
                    continue
                item['settlement_checked_utc'] = now_utc.isoformat()
                if outcome is not None:
                    item['actual_up'] = outcome
                changed = True
            if outcome is not None:
                notes.append(f'Verified settlement: {market_ticker}')
        return changed, notes


    def _checkpoint_report(rows):
        """Summarize exactly the logged, subsequently verified snapshots."""
        table = pd.DataFrame(rows)
        records = []
        if table.empty:
            return table, pd.DataFrame()
        for checkpoint in CHECKPOINTS:
            group = table[(table['checkpoint_minutes'] == checkpoint) &
                          table['actual_up'].notna()].copy()
            n = len(group)
            if n:
                prob = pd.to_numeric(group['model_p_up'], errors='coerce')
                actual = pd.to_numeric(group['actual_up'], errors='coerce')
                valid = prob.notna() & actual.notna() & prob.between(0, 1)
                group = group[valid]
                prob = prob[valid]
                actual = actual[valid]
                n = len(group)
            right = int(((prob >= 0.5).astype(int) == actual).sum()) if n else 0
            brier = float(((prob - actual)**2).mean()) if n else float('nan')
            records.append({'Time remaining': f'{checkpoint} min',
                            'Verified': n, 'Correct': right,
                            'Accuracy': f'{right/n:.1%}' if n else '—',
                            'Brier': f'{brier:.3f}' if n else '—',
                            'Data quality': 'Early sample' if n < 100 else 'Growing sample'})
        return table, pd.DataFrame(records)


    @st.fragment(run_every='10s')
    def checkpoint_accuracy_panel():
        st.subheader('2B. Live prediction accuracy by checkpoint')
        st.caption('Records real-time forecasts at 12, 9, 6 and 3 minutes before close, '
                   'then verifies against the Kalshi-published YES/NO result. '
                   'If the app is inactive, missed checkpoints are NOT backfilled.')
        try:
            rows, sha, location = _checkpoint_store_read()
        except Exception as exc:
            st.error(f'Could not load checkpoint research log: {exc}')
            return
        st.caption('Storage: ' + location)
        if 'not persistent' in location:
            st.warning('This research log is temporary until the existing GitHub logging secrets are configured.')
        live_market = None
        try:
            live_market = current_kalshi()
            current_spot = ticker()
            hist = candles(days)
            age = (datetime.now(timezone.utc) - hist.index[-1].to_pydatetime()).total_seconds()
            if age > 240:
                raise ValueError('Volatility history is stale')
            v15 = estimated_sigma(hist, 15)
            v60 = estimated_sigma(hist, max(60, lookback))
            if not (math.isfinite(v15) and math.isfinite(v60) and v15 > 0 and v60 > 0):
                raise ValueError('Volatility estimate unavailable')
            vol = 0.6*v15 + 0.4*v60
            now_utc = datetime.now(timezone.utc)
        except Exception as exc:
            live_market = None
            current_spot = float('nan')
            vol = float('nan')
            now_utc = datetime.now(timezone.utc)
            st.caption(f'Live checkpoint capture currently unavailable: {exc}')
        changed, notes = _checkpoint_update(rows, live_market, current_spot, vol, now_utc)
        if changed:
            try:
                _checkpoint_store_write(rows, sha)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code in (409, 422):
                    st.warning('Another browser updated this log simultaneously. This update was not saved; reload the page to retry.')
                else:
                    st.warning(f'Prediction snapshot could not be saved: {exc}')
            except Exception as exc:
                st.warning(f'Prediction snapshot could not be saved: {exc}')
        if notes:
            st.caption(' • '.join(notes))
        table, report = _checkpoint_report(rows)
        if report.empty:
            st.info('No checkpoint predictions saved yet. Leave this dashboard open during a cycle to begin collecting them.')
        else:
            st.dataframe(report, hide_index=True, use_container_width=True)
            if table['actual_up'].notna().any():
                settled = table[table['actual_up'].notna()].copy()
                settled['p_bucket'] = pd.cut(pd.to_numeric(settled['model_p_up']),
                                             bins=[0,.2,.4,.6,.8,1], include_lowest=True)
                calib = settled.groupby('p_bucket', observed=False).agg(
                    n=('actual_up','size'),
                    predicted_up=('model_p_up','mean'),
                    actual_up=('actual_up','mean')).reset_index()
                st.write('**Are the probability percentages calibrated?**')
                st.dataframe(calib, hide_index=True, use_container_width=True)
                st.caption('Calibration: when predictions average 80% UP, about 80% should finish UP over many independent markets. '
                           'The same market contributes multiple checkpoint rows, so do not treat them as independent observations.')
            st.write('**Most recent checkpoint forecasts**')
            st.dataframe(table.tail(20).iloc[::-1], hide_index=True, use_container_width=True)
            st.download_button('Download checkpoint accuracy CSV',
                               table.to_csv(index=False).encode(),
                               file_name='bitcoin_checkpoint_accuracy_log.csv', mime='text/csv',
                               key='checkpoint_audit_csv')
        st.warning('Research only. This driftless model always favors the side where Coinbase is currently trading relative to the target. '
                   'Raw directional accuracy therefore does NOT prove predictive skill. The important test is probability calibration '
                   'against actual Kalshi outcomes. Coinbase is not the CF Benchmarks BRTI settlement source.')
        st.caption('Tracking depends on an active Streamlit browser session. Closing the tab, sleeping the computer, '
                   'or browser tab throttling can prevent new observations. Verified settlement checks resume when the page runs again.')


    checkpoint_accuracy_panel()

    # -----------------------------------------------------------------------------
    # Historical swing / reversal research (Coinbase OHLC candles, not Kalshi odds)
    # -----------------------------------------------------------------------------
    SWING_HORIZONS = (12, 9, 6, 3)
    SWING_DOLLARS = (25, 50, 75, 100, 150)

    @st.cache_data(ttl=90, show_spinner=False)
    def swing_window_history(candle_data):
        """Return independent snapshots at 12/9/6/3 minutes from cycle end.

        Minute-bar index is candle OPEN time, so a bar at S+2 has a CLOSE at S+3,
        i.e., exactly 12 minutes before the end of the S to S+15 interval.
        Only *completed* 15-minute cycles with every 1-min bar are included.

        Historical target is a proxy: first 1-minute Coinbase candle CLOSE.
        Actual Kalshi target and settlement index may differ.
        """
        clean = candle_data.sort_index()
        if clean.empty:
            return pd.DataFrame()
        idx = clean.index
        end_of_last_complete_bar = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        starts = pd.date_range(idx.min().ceil('15min'),
                               idx.max().floor('15min'), freq='15min', tz='UTC')
        records = []
        for start in starts:
            # Don't use a partial final bar (or a still-open 15-minute cycle).
            if start.to_pydatetime() + timedelta(minutes=15) > end_of_last_complete_bar:
                continue
            times = pd.date_range(start, periods=15, freq='min')
            if not times.isin(idx).all():
                continue
            segment = clean.loc[times]
            target_proxy = float(segment['close'].iloc[0])
            finish = float(segment['close'].iloc[-1])
            if not np.isfinite([target_proxy, finish]).all():
                continue
            for mins in SWING_HORIZONS:
                current_idx = 14 - mins   # e.g. mins=12 -> bar S+2 closes at S+3
                here = float(segment['close'].iloc[current_idx])
                future = segment.iloc[current_idx + 1:]
                if (not np.isfinite(here) or future.empty or
                        not np.isfinite(future[['high', 'low', 'close']].to_numpy()).all()):
                    continue
                abs_move_to_finish = abs(finish - here)
                max_up = float(future['high'].max()) - here
                max_down = here - float(future['low'].min())
                # Ended strictly across the proxy target, not a temporary touch.
                if here == target_proxy or finish == target_proxy:
                    reversed_at_close = np.nan
                else:
                    reversed_at_close = int((here > target_proxy) != (finish > target_proxy))
                records.append({
                    'window_start_utc': start,
                    'minutes_left': mins,
                    'price_at_checkpoint': here,
                    'proxy_target': target_proxy,
                    'price_at_finish': finish,
                    'distance_from_proxy_target': abs(here - target_proxy),
                    'side_at_checkpoint': 'UP' if here > target_proxy else ('DOWN' if here < target_proxy else 'AT TARGET'),
                    'absolute_close_to_close_move': abs_move_to_finish,
                    'max_favorable_up_move': max_up,
                    'max_favorable_down_move': max_down,
                    'max_either_direction_move': max(max_up, max_down),
                    'opposite_side_at_finish': reversed_at_close,
                })
        return pd.DataFrame.from_records(records)


    def swing_rate_table(history, measure):
        """Unconditional historical touch or close-to-close movement percentages."""
        result = []
        metric = ('max_either_direction_move' if measure == 'touch' else
                  'absolute_close_to_close_move')
        for mins in SWING_HORIZONS:
            sample = history.loc[history['minutes_left'] == mins] if not history.empty else pd.DataFrame()
            row = {'Time left': f'{mins} min', 'Cycles': len(sample)}
            for amount in SWING_DOLLARS:
                n = len(sample)
                wins = int((sample[metric] >= amount).sum()) if n else 0
                row[f'\u2265${amount}'] = f'{wins/n:.1%} ({wins}/{n})' if n else 'No data'
            result.append(row)
        return pd.DataFrame(result)


    def reversal_by_distance_table(history, minutes_left):
        """Descriptive frequencies in distance bands, not calibrated odds."""
        sample = history.loc[(history['minutes_left'] == minutes_left) &
                             history['opposite_side_at_finish'].notna()]
        bands = [(0,25), (25,50), (50,75), (75,100), (100,150), (150,float('inf'))]
        rows = []
        for left, right in bands:
            grp = sample.loc[(sample['distance_from_proxy_target'] >= left) &
                             (sample['distance_from_proxy_target'] < right)]
            n = len(grp)
            flipped = int(grp['opposite_side_at_finish'].sum()) if n else 0
            band_label = f'${left}–${right}' if math.isfinite(right) else '$150+'
            rows.append({'Distance from target (proxy)': band_label,
                         'Reversed / observed': f'{flipped}/{n}',
                         'Observed reversal frequency': (f'{flipped/n:.1%}' if n else 'No data'),
                         'Sample quality': 'Small sample' if 0 < n < 30 else ('No data' if n == 0 else 'Preliminary')})
        return pd.DataFrame(rows)


    @st.fragment(run_every='15s')
    def swing_research_panel():
        st.subheader('3. Bitcoin swing & target-reversal research')
        st.write('At **12, 9, 6, or 3 minutes remaining**, see how frequently Bitcoin historically moved another $25–$150 before the cycle ended. This is Coinbase price history, **not verified Kalshi settlement history**.')
        # Refresh candle history within this fragment (the cache expires every 90 seconds).
        hist = swing_window_history(candles(days))
        if hist.empty:
            st.warning('Not enough complete 1-minute price candles to measure swings. Try 2–7 historical days.')
            return
        period = st.selectbox('Which recent history should we compare?',
            ['All loaded days', 'Last 24 hours', 'Last 6 hours', 'Last 1 hour'], index=0,
            key='swing_history_period')
        hours = {'Last 24 hours': 24, 'Last 6 hours': 6, 'Last 1 hour': 1}.get(period)
        if hours is not None:
            earliest = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=hours)
            hist = hist.loc[hist['window_start_utc'] >= earliest]
        if hist.empty:
            st.warning('No complete cycles in that time range. Choose a longer history.')
            return
        n_cycles = hist.window_start_utc.nunique()
        if n_cycles < 30:
            st.warning(f'Only {n_cycles} completed cycles in this time range. Percentages are descriptive and too sparse to trust as predictive odds.')

        # These are different events: crossing a level at *any time* vs FINISHING there.
        metric_name = st.radio('What should count as a swing?',
            ('Touch the amount before cycle ends', 'Finish at least that far away'),
            horizontal=True, key='swing_metric_choice')
        measure = 'touch' if metric_name.startswith('Touch') else 'finish'
        st.dataframe(swing_rate_table(hist, measure), hide_index=True, use_container_width=True)
        if measure == 'touch':
            st.caption('Example: $100 means price reached at least $100 ABOVE OR BELOW its checkpoint price at any moment before the end. This can happen even if the price later returns.')
        else:
            st.caption('Example: $100 means the end-of-cycle Coinbase price was at least $100 higher or lower than the checkpoint price.')
        st.caption(f'Sample: {hist.window_start_utc.nunique()} complete cycles. One observation per cycle at each checkpoint; 1-minute Coinbase candles include highs and lows. Rounded historical frequencies are not guarantees or validated forecasts.')

        st.markdown('**How often did Bitcoin finish on the opposite side of its target?**')
        st.caption('This second table concerns target reversals, not just a $25/$50 price movement. Historical targets are Coinbase approximations, not actual Kalshi targets.')
        selected = st.selectbox('Minutes remaining for target reversal table',
                                 options=list(SWING_HORIZONS), index=2, format_func=lambda v: f'{v} minutes', key='reversal_time_select')
        st.dataframe(reversal_by_distance_table(hist, selected),
                     hide_index=True, use_container_width=True)

        # Live snapshot: report distance and the nearest historical checkpoint. Avoid
        # presenting descriptive rates as current Kalshi settlement probabilities.
        st.markdown('**Current market comparison**')
        try:
            live_market = current_kalshi()
            live_price = ticker()
            end = datetime.fromisoformat(live_market['close'])
            mins_left = (end - datetime.now(timezone.utc)).total_seconds() / 60
            if mins_left <= 0 or mins_left > 15:
                raise ValueError('No active 15-minute market')
            distance = live_price - float(live_market['target'])
            closest_horizon = min(SWING_HORIZONS, key=lambda m: abs(m-mins_left))
            c1, c2, c3 = st.columns(3)
            c1.metric('Above / below Kalshi target*', f'${distance:+,.2f}')
            c2.metric('Time left', f'{mins_left:.1f} min')
            c3.metric('Nearest research checkpoint', f'{closest_horizon} min')
            historical_checkpoint = hist.loc[(hist['minutes_left'] == closest_horizon) &
                hist['opposite_side_at_finish'].notna()]
            gap = abs(distance)
            bands = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 150), (150, float('inf'))]
            bucket_low, bucket_high = next((low, high) for low, high in bands if low <= gap < high)
            comparable = historical_checkpoint.loc[
                (historical_checkpoint['distance_from_proxy_target'] >= bucket_low) &
                (historical_checkpoint['distance_from_proxy_target'] < bucket_high)]
            n_comp = len(comparable)
            flipped_comp = int(comparable['opposite_side_at_finish'].sum()) if n_comp else 0
            label = f'${bucket_low}–${bucket_high}' if math.isfinite(bucket_high) else '$150+'
            if n_comp >= 30:
                st.write(f'**Similar historical distances ({label}, {closest_horizon} min left):** {flipped_comp}/{n_comp} finished across the proxy target ({flipped_comp/n_comp:.1%} observed).')
            else:
                st.write(f'**Similar historical distances ({label}, {closest_horizon} min left):** {flipped_comp}/{n_comp} crossed by settlement. **Insufficient observations for dependable odds.**')
            st.caption('*Distance uses Coinbase spot versus Kalshi target; sources may differ. The comparable history uses proxy targets, not Kalshi target records. The selected checkpoint may differ from the actual remaining time. This does not predict the outcome of the current Kalshi market.')
        except Exception as exc:
            st.info('Live comparison unavailable until the current Kalshi target and Coinbase price can both be retrieved. Historical tables still work.')

        st.download_button('Download swing research CSV',
            hist.to_csv(index=False).encode('utf-8'),
            file_name='bitcoin_swing_history.csv', mime='text/csv', key='download_swing_history')


    swing_research_panel()

    st.subheader('4. Walk-forward historical test')
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

    st.subheader('5. Optional actual Kalshi trade log')
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
