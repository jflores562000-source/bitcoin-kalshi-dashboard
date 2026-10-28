"""Kalshi public quote recorder and early-cycle hypothetical exit research. No trading."""
import base64
import json
import math
import time
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


def fee_cents(price_cents):
    """Kalshi general trading fee for ONE contract, in cents: ceil(0.07 * P * (1 - P)).
    Same formula app.py uses. Confirm the exact fee in Kalshi's trade preview."""
    p = float(price_cents) / 100
    return math.ceil(0.07 * p * (1 - p) * 100 - 1e-12)


def kalshi_result(ticker):
    """Kalshi's published result for a finished market: 1 = UP (yes) won, 0 = DOWN (no) won, None = not published yet."""
    r = requests.get(f'https://external-api.kalshi.com/trade-api/v2/markets/{ticker}', timeout=8)
    r.raise_for_status()
    result = str(r.json().get('market', {}).get('result') or '').lower()
    return {'yes': 1, 'no': 0}.get(result)


def load_outcomes(tickers, max_new=5):
    """Look up Kalshi results a few at a time (the page refreshes every few seconds).
    Finished results never change, so each one is kept for the rest of the browser session."""
    store = st.session_state.setdefault('kalshi_outcomes', {})
    checked = st.session_state.setdefault('kalshi_outcomes_checked', {})
    now, fetched = time.time(), 0
    for ticker in tickers:
        if store.get(ticker) is not None or fetched >= max_new or now - checked.get(ticker, 0) < 90:
            continue
        checked[ticker] = now
        fetched += 1
        try:
            store[ticker] = kalshi_result(ticker)
        except Exception:
            pass  # try again in 90 seconds; never break the page
    return {t: v for t, v in store.items() if v is not None}


def add_net_returns(results, outcomes):
    """Adds what the bid-touch table leaves out: the real result of every trade, after fees.

    Hit  -> sold at the target:   target - entry ask - buy fee - sell fee
    Miss -> held to settlement:   (100 if that side won else 0) - entry ask - buy fee   (no fee at settlement)
    Trades whose market has no Kalshi result yet get no number at all. Hits are NOT counted early,
    because that would leave out the slow-arriving losses and make the average look better than it is.
    """
    out = results.copy()
    up_won = out['ticker'].map(outcomes)                  # NaN when Kalshi's result is not known yet
    won = up_won.where(out['side'] == 'up', 1 - up_won)   # did THIS side win?
    ask, target = out['observed_entry_ask'], out['exit_target']
    hit = out['target_bid_observed'].astype(bool)
    net_hit = target - ask - ask.map(fee_cents) - target.map(fee_cents)
    net_miss = 100 * won - ask - ask.map(fee_cents)
    out['settled_win'] = won
    out['net_cents_after_fees'] = net_hit.where(hit, net_miss).where(won.notna())
    return out


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
        keys = ['entry_ceiling', 'exit_target']
        summary = summary.merge(results.groupby(keys, as_index=False).agg(separate_markets=('ticker', 'nunique')),
                                on=keys, how='left')
        # --- NEW: what the bid-touch rate leaves out (misses, fees) ---
        finished = sorted(results['ticker'].unique())
        outcomes = load_outcomes(finished)
        shown = results
        if outcomes:
            shown = add_net_returns(results, outcomes)
            known = shown.dropna(subset=['net_cents_after_fees'])
            if not known.empty:
                hit_rows = known[known['target_bid_observed'].astype(bool)]
                miss_rows = known[~known['target_bid_observed'].astype(bool)]
                summary = (summary
                           .merge(known.groupby(keys, as_index=False).agg(
                               with_result=('ticker', 'size'), avg_net=('net_cents_after_fees', 'mean')), on=keys, how='left')
                           .merge(hit_rows.groupby(keys, as_index=False).agg(
                               avg_gain=('net_cents_after_fees', 'mean')), on=keys, how='left')
                           .merge(miss_rows.groupby(keys, as_index=False).agg(
                               avg_miss=('net_cents_after_fees', 'mean')), on=keys, how='left'))
                # hit rate p where p * avg gain + (1 - p) * avg miss = 0 (only meaningful when misses lose money)
                summary['break_even_rate'] = (100 * -summary['avg_miss'] / (summary['avg_gain'] - summary['avg_miss'])
                                              ).where(summary['avg_miss'] < 0).round(1)
                for col in ('avg_net', 'avg_gain', 'avg_miss'):
                    summary[col] = summary[col].round(2)
        st.dataframe(summary.rename(columns={'entry_ceiling': 'Buy at or below (¢)',
                         'exit_target': 'Sell target (¢)', 'opportunities': 'Observed opportunities',
                         'target_observed': 'Bid reached target', 'mean_entry_ask': 'Avg observed buy ask (¢)',
                         'observed_hit_rate': 'Observed bid-touch rate (%)',
                         'separate_markets': 'Separate markets',
                         'with_result': 'Trades with Kalshi result',
                         'avg_net': 'Avg result per contract AFTER fees (¢)',
                         'avg_gain': 'Avg result when target hit (¢)',
                         'avg_miss': 'Avg result when target missed (¢)',
                         'break_even_rate': 'Hit rate needed to break even (%)'}),
                     hide_index=True, use_container_width=True)
        if outcomes:
            st.caption(f'Kalshi results loaded for {len(outcomes)} of {len(finished)} finished markets. '
                       'A negative "after fees" number means the plan loses money on average, even when the hit rate '
                       'looks high: misses are held to settlement, and wins are small next to losses. '
                       'Compare "Observed bid-touch rate" with "Hit rate needed to break even".')
        else:
            st.info('Loading Kalshi results so the after-fees numbers can be added. Refresh in a minute. '
                    'If this stays here, Kalshi results may be unreachable from this server.')
        st.download_button('Download early entry research CSV', shown.to_csv(index=False).encode(),
                           file_name='kalshi_auto_sell_research.csv', mime='text/csv')
    st.warning('Research only: sampled bid-touch rates are NOT fill probabilities or verified net returns. '
               'Unobserved moves, order queue, size, fees, spread, partial fills, and unsold settlement '
               'can materially change results. No profitability claim until trade-level execution and settlement data are added.')
