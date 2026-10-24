"""Optional persistent Kalshi research log. No order placement or private Kalshi access."""
import base64
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd
import streamlit as st

LOCAL_PATH = Path('/tmp/btc_kalshi_research_log.json')  # Ephemeral on Streamlit Cloud.
GITHUB_API = 'https://api.github.com'
LOG_PATH = 'data/kalshi_predictions.json'


def settings():
    try:
        token = st.secrets.get('GITHUB_LOG_TOKEN', '')
        repo = st.secrets.get('GITHUB_LOG_REPO', '')
        return str(token), str(repo)
    except Exception:
        return '', ''


def github_read(token, repo):
    url = f'{GITHUB_API}/repos/{repo}/contents/{LOG_PATH}'
    r = requests.get(url, headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'},
                     params={'ref': 'main'}, timeout=12)
    if r.status_code == 404:
        return [], None
    r.raise_for_status()
    obj = r.json()
    data = json.loads(base64.b64decode(obj['content']).decode('utf-8'))
    if not isinstance(data, list):
        raise ValueError('Research log must be a JSON array')
    return data, obj['sha']


def github_write(token, repo, rows, sha):
    url = f'{GITHUB_API}/repos/{repo}/contents/{LOG_PATH}'
    body = {'message': 'Update Kalshi research prediction log',
            'content': base64.b64encode(json.dumps(rows, indent=2, allow_nan=False).encode()).decode(),
            'branch': 'main'}
    if sha:
        body['sha'] = sha
    r = requests.put(url, json=body,
                     headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'},
                     timeout=15)
    r.raise_for_status()


def load():
    token, repo = settings()
    if token and repo:
        rows, sha = github_read(token, repo)
        return rows, sha, 'GitHub repository (persistent)'
    if LOCAL_PATH.exists():
        return json.loads(LOCAL_PATH.read_text()), None, 'Temporary server file (NOT persistent)'
    return [], None, 'Temporary server file (NOT persistent)'


def save(rows, sha):
    token, repo = settings()
    if token and repo:
        github_write(token, repo, rows, sha)
    else:
        LOCAL_PATH.write_text(json.dumps(rows, indent=2, allow_nan=False))


def result_from_kalshi(ticker):
    r = requests.get(f'https://external-api.kalshi.com/trade-api/v2/markets/{ticker}', timeout=10)
    r.raise_for_status()
    m = r.json().get('market', {})
    outcome = str(m.get('result') or '').lower()
    # Only explicit published Kalshi results are accepted. Never infer from Coinbase.
    if outcome in ('yes', 'no'):
        return 1 if outcome == 'yes' else 0
    return None


def render(market, spot, sigma, probability_up, fee_per_contract):
    st.subheader('5. Live prediction performance tracker')
    st.caption('Records one observation per market around five minutes before close while this page is open. '
               'Only Kalshi-published final results count as verified outcomes. No trades are placed.')
    try:
        rows, sha, store = load()
    except Exception as exc:
        st.error(f'Prediction log could not be read: {exc}')
        return
    if 'NOT persistent' in store:
        st.warning('Tracking is currently temporary: Streamlit Cloud may delete this log on restart. '
                   'Configure GitHub storage before relying on long-term records. Download CSV regularly.')
    else:
        st.success('Prediction log storage: ' + store)
    changed = False
    if market is not None and math.isfinite(sigma) and sigma > 0:
        now = datetime.now(timezone.utc)
        close = datetime.fromisoformat(market['close'])
        minutes = (close - now).total_seconds() / 60
        exists = any(r.get('ticker') == market['ticker'] for r in rows)
        # Fixed observation horizon prevents picking a favorable entry time after the fact.
        if 4.5 <= minutes <= 5.5 and not exists:
            p = probability_up(spot, market['target'], minutes, sigma)
            if math.isfinite(p):
                rows.append({'ticker': market['ticker'], 'observed_utc': now.isoformat(),
                             'close_utc': market['close'], 'minutes_remaining': round(minutes, 3),
                             'kalshi_target': market['target'], 'coinbase_spot': spot,
                             'up_ask_cents': market['upask'], 'down_ask_cents': market['downask'],
                             'p_up': round(p, 6), 'volatility': sigma,
                             'up_fee_est_cents': fee_per_contract(market['upask']),
                             'down_fee_est_cents': fee_per_contract(market['downask']),
                             'actual_up': None, 'settled_checked_utc': None})
                changed = True
    # Check only a small number of completed, unresolved markets per refresh.
    unresolved = [r for r in rows if r.get('actual_up') is None and r.get('close_utc')]
    now = datetime.now(timezone.utc)
    for record in unresolved[:3]:
        if datetime.fromisoformat(record['close_utc']) >= now:
            continue
        last = record.get('settled_checked_utc')
        if last and (now - datetime.fromisoformat(last)).total_seconds() < 90:
            continue
        try:
            actual = result_from_kalshi(record['ticker'])
            record['settled_checked_utc'] = now.isoformat()
            if actual is not None:
                record['actual_up'] = actual
            changed = True
        except requests.RequestException:
            pass
    if changed:
        try:
            save(rows, sha)
        except Exception as exc:
            st.error(f'Could not save prediction log: {exc}. Data may not persist.')
    if not rows:
        st.info('No live observations yet. Leave the dashboard open through the five-minute-before-close point of a market.')
        return
    table = pd.DataFrame(rows)
    settled = table[table['actual_up'].notna()].copy()
    a, b, c, d = st.columns(4)
    a.metric('Logged markets', len(table))
    b.metric('Verified settlements', len(settled))
    if len(settled):
        actual = settled['actual_up'].astype(float)
        probs = settled['p_up'].astype(float)
        brier = ((probs - actual) ** 2).mean()
        correct = ((probs >= .5).astype(int) == actual).mean()
        c.metric('Directional accuracy', f'{correct:.1%}')
        d.metric('Brier score', f'{brier:.3f}')
        st.caption('Accuracy and Brier score describe this small observational sample; neither proves a trading edge.')
        settled['bucket'] = pd.cut(probs, bins=[0,.2,.4,.6,.8,1], include_lowest=True)
        cal = settled.groupby('bucket', observed=False).agg(
            n=('actual_up','size'), mean_predicted=('p_up','mean'), observed_up=('actual_up','mean')).reset_index()
        st.write('Live prediction calibration')
        st.dataframe(cal, hide_index=True, use_container_width=True)
    else:
        c.metric('Directional accuracy', 'Pending')
        d.metric('Brier score', 'Pending')
    st.dataframe(table.tail(50).iloc[::-1], hide_index=True, use_container_width=True)
    st.download_button('Download live research log CSV', table.to_csv(index=False).encode(),
                       file_name='kalshi_live_prediction_log.csv', mime='text/csv', key='live_log_csv')
