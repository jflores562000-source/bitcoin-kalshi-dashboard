"""Kalshi public quote recorder and early-cycle hypothetical exit research. No trading."""
import base64
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

PATH = 'data/kalshi_quote_history.json'
LOCAL = Path('/tmp/kalshi_quote_history.json')
ENTRIES = (30, 35, 40)
EXITS = (45, 50, 55)


def credentials():
    try:
        return str(st.secrets.get('GITHUB_LOG_TOKEN', '')), str(st.secrets.get('GITHUB_LOG_REPO', ''))
    except Exception:
        return '', ''


def read_log():
    token, repo = credentials()
    if not (token and repo):
        return (json.loads(LOCAL.read_text()) if LOCAL.exists() else []), None, 'Temporary local storage'
    r = requests.get(f'https://api.github.com/repos/{repo}/contents/{PATH}',
                     headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'},
                     params={'ref': 'main'}, timeout=12)
    if r.status_code == 404:
        return [], None, 'GitHub persistent storage'
    r.raise_for_status()
    obj = r.json()
    return json.loads(base64.b64decode(obj['content']).decode()), obj['sha'], 'GitHub persistent storage'


def write_log(rows, sha):
    token, repo = credentials()
    if not (token and repo):
        LOCAL.write_text(json.dumps(rows))
        return
    body = {'message': 'Record Kalshi public quotes for exit research', 'branch': 'main',
            'content': base64.b64encode(json.dumps(rows, allow_nan=False).encode()).decode()}
    if sha:
        body['sha'] = sha
    r = requests.put(f'https://api.github.com/repos/{repo}/contents/{PATH}', json=body,
                     headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'}, timeout=15)
    r.raise_for_status()


def valid_price(value):
    try:
        n = float(value)
        return round(n, 2) if 0 < n < 100 else None
    except (ValueError, TypeError):
        return None


def record(market):
    rows, sha, store = read_log()
    now = datetime.now(timezone.utc)
    if market is not None:
        close = datetime.fromisoformat(market['close'])
        elapsed = 15 - (close - now).total_seconds() / 60
        if 0 <= elapsed < 15:
            last = next((r for r in reversed(rows) if r['ticker'] == market['ticker']), None)
            # Store at most one observation per ~10 seconds, while browser is active.
            if last is None or (now - datetime.fromisoformat(last['observed_utc'])).total_seconds() >= 9:
                row = {'ticker': market['ticker'], 'observed_utc': now.isoformat(),
                       'close_utc': market['close'], 'elapsed_min': round(elapsed, 3)}
                for side in ('up', 'down'):
                    for quote in ('ask', 'bid'):
                        row[f'{side}_{quote}'] = valid_price(market.get(f'{side}{quote}'))
                rows.append(row)
                try:
                    write_log(rows, sha)
                except Exception as exc:
                    st.warning(f'Quote snapshot could not be saved: {exc}')
    return rows, store


def analyze(rows):
    """Conservative displayed-quote touch test, NOT proof of executable fills.

    One earliest eligible entry per market/side/threshold; later observed bid must
    meet exit threshold. Missing bid data is never inferred from opposite asks.
    Unfinished windows are excluded. Failed exits valued at zero (worst-case),
    not marked as a verified settlement loss.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    needed = {'ticker', 'observed_utc', 'close_utc', 'elapsed_min', 'up_ask', 'down_ask', 'up_bid', 'down_bid'}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    df['observed_utc'] = pd.to_datetime(df['observed_utc'], utc=True, errors='coerce')
    df['close_utc'] = pd.to_datetime(df['close_utc'], utc=True, errors='coerce')
    df = df.dropna(subset=['observed_utc', 'close_utc']).sort_values('observed_utc')
    now = pd.Timestamp.now(tz='UTC')
    results = []
    for ticker, g in df.groupby('ticker'):
        if g['close_utc'].iloc[-1] + pd.Timedelta(seconds=15) > now:
            continue
        for side in ('up', 'down'):
            ask, bid = f'{side}_ask', f'{side}_bid'
            g = g.copy()
            g[ask] = pd.to_numeric(g[ask], errors='coerce')
            g[bid] = pd.to_numeric(g[bid], errors='coerce')
            for entry in ENTRIES:
                eligible = g[(g['elapsed_min'] >= 0) & (g['elapsed_min'] <= 4) &
                             (g[ask] > 0) & (g[ask] <= entry)]
                if eligible.empty:
                    continue
                first = eligible.iloc[0]
                later = g[g['observed_utc'] > first['observed_utc']]
                for target in EXITS:
                    if target <= first[ask]:
                        continue
                    hits = later[later[bid] >= target]
                    success = not hits.empty
                    results.append({'ticker': ticker, 'side': side, 'entry_ceiling': entry,
                                    'exit_target': target, 'observed_entry_ask': float(first[ask]),
                                    'target_bid_observed': success,
                                    'gross_cents_if_target_hit': round(target - float(first[ask]), 2),
                                    'first_target_time_min': round(float(hits.iloc[0]['elapsed_min']), 2) if success else None})
    return pd.DataFrame(results)


def render(market):
    st.subheader('Buy Low / Auto-Sell Research')
    st.caption('Records public UP/DOWN bid and ask quotes while this page stays active. No orders are placed.')
    try:
        rows, store = record(market)
    except Exception as exc:
        st.warning(f'Quote research unavailable: {exc}')
        return
    st.caption(f'Storage: {store} · {len(rows)} saved snapshots')
    if 'Temporary' in store:
        st.warning('Temporary storage can disappear on restart. Configure the existing GitHub secrets for persistence.')
    results = analyze(rows)
    if results.empty:
        st.info('Collect complete cycles first. Results appear once an eligible early entry and subsequent quote history exist.')
    else:
        summary = results.groupby(['entry_ceiling', 'exit_target'], as_index=False).agg(
            opportunities=('ticker', 'size'), target_observed=('target_bid_observed', 'sum'),
            mean_entry_ask=('observed_entry_ask', 'mean'))
        summary['observed_hit_rate'] = (100 * summary['target_observed'] / summary['opportunities']).round(1)
        st.dataframe(summary.rename(columns={'entry_ceiling': 'Buy at or below (¢)',
                         'exit_target': 'Sell target (¢)', 'opportunities': 'Observed opportunities',
                         'target_observed': 'Bid reached target', 'mean_entry_ask': 'Avg observed buy ask (¢)',
                         'observed_hit_rate': 'Observed bid-touch rate (%)'}),
                     hide_index=True, use_container_width=True)
        st.download_button('Download early entry research CSV', results.to_csv(index=False).encode(),
                           file_name='kalshi_auto_sell_research.csv', mime='text/csv')
    st.warning('Research only: sampled bid-touch rates are NOT fill probabilities or verified net returns. '
               'Unobserved moves, order queue, size, fees, spread, partial fills, and unsold settlement '
               'can materially change results. No profitability claim until trade-level execution and settlement data are added.')
