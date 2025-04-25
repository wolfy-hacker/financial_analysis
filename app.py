import io
import tempfile
import os
import re

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Dark theme + sidebar styling
st.markdown("""
  <style>
    body { background-color: #111; color: #eee; }
    .sidebar .sidebar-content { background-color: #222; }
    .stMetric > div[data-testid="metric-container"] {
      background: #333 !important;
      border: 1px solid #555; border-radius: 8px; padding:10px;
    }
  </style>
""", unsafe_allow_html=True)
px.defaults.template = "plotly_dark"

from camelot.io import read_pdf
import tabula
import pdfplumber

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ————————————
# 1. Load & preprocess data (multi‑fallback)
# ————————————
def load_data(uploaded_file):
    """
    Load CSV or PDF with fallbacks:
      1) CSV via pandas
      2) Camelot (lattice → stream)
      3) Tabula
      4) pdfplumber + regex
    Then:
      • strip currency/commas/percent/parens → numeric
      • promote first row to header if needed
      • drop empty/duplicate columns
      • pivot metrics×years → DateTimeIndex
    """
    # 1) read raw
    if uploaded_file.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded_file, header=0, dtype=str)
    else:
        # PDF → Camelot → Tabula → pdfplumber
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.getbuffer())
            path = tmp.name

        tables = read_pdf(path, pages='all', flavor='lattice') \
              or read_pdf(path, pages='all', flavor='stream')
        if tables:
            df = pd.concat([t.df for t in tables], ignore_index=True)
        else:
            try:
                dfs = tabula.read_pdf(path, pages='all', lattice=True)
                df = pd.concat(dfs, ignore_index=True)
            except:
                rows = []
                with pdfplumber.open(path) as pdf:
                    for pg in pdf.pages:
                        for line in (pg.extract_text() or "").split('\n'):
                            parts = re.split(r'\s{2,}', line.strip())
                            if len(parts) > 1:
                                rows.append(parts)
                if not rows:
                    os.remove(path)
                    st.error("Failed to extract any table from PDF.")
                    st.stop()
                df = pd.DataFrame(rows[1:], columns=rows[0])
        os.remove(path)

    # 2) strip symbols & coerce to numeric
    df = df.replace({
        r'₹\s*': '', r',': '', r'%': '',
        r'\(': '-', r'\)': ''
    }, regex=True)
    df_header = df.copy()
    df = df.apply(pd.to_numeric, errors='coerce')

    # 3) promote first row if all columns are NaN or numeric
    cols = df.columns
    if cols.isnull().all() or all(isinstance(c,(int,float)) for c in cols):
        df = df_header.copy()
        df.columns = df.iloc[0].astype(str)
        df = df.drop(df.index[0]).reset_index(drop=True)

    # 4) clean up column names
    df.columns = [
        str(c).replace('₹','').replace('&','and').strip().replace(' ','_')
        for c in df.columns
    ]

    # 5) drop empty and duplicate columns
    df = df.loc[:, df.columns.astype(bool)]               # drop '' names
    df = df.loc[:, ~df.columns.duplicated()]               # drop duplicates

    # 6) pivot metrics×years if relevant
    if len(df.columns)>1 and all(c.isdigit() for c in df.columns):
        df = df.set_index(df.columns[0]).T
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.loc[~df.index.isna()]

    return df.sort_index()

# ————————————
# 2. Compute advanced financial ratios
# ————————————
def compute_ratios(df, st_type):
    """
    Compute a small set of ratios based on the detected statement type.
    """
    if st_type == "Income Statement":
        # Profitability ratios
        if 'revenue' in df.columns and 'grossProfit' in df.columns:
            df['Gross_Margin'] = df['grossProfit'] / df['revenue'] * 100
        if 'revenue' in df.columns and 'ebitda' in df.columns:
            df['EBITDA_Margin'] = df['ebitda'] / df['revenue'] * 100
        if 'revenue' in df.columns and 'netIncome' in df.columns:
            df['Net_Profit_Margin'] = df['netIncome'] / df['revenue'] * 100

    elif st_type == "Balance Sheet":
        # Liquidity
        if 'totalCurrentAssets' in df.columns and 'otherLiabilities' in df.columns:
            df['Current_Ratio'] = df['totalCurrentAssets'] / df['otherLiabilities']
        if all(k in df.columns for k in ['cashAndBank','receivables','otherLiabilities']):
            df['Quick_Ratio'] = (df['cashAndBank'] + df['receivables']) / df['otherLiabilities']
        # Leverage
        if 'borrowing' in df.columns and 'reserves' in df.columns:
            df['Debt_to_Equity'] = df['borrowing'] / df['reserves']
        # ROE (needs netIncome + reserves)
        if 'netIncome' in df.columns and 'reserves' in df.columns:
            df['ROE'] = df['netIncome'] / df['reserves'] * 100

    elif st_type == "Cash Flow":
        # Operating Cash Flow Ratio
        if 'operatingCashFlow' in df.columns and 'totalCurrentAssets' in df.columns:
            df['Operating_Cash_Flow_Ratio'] = df['operatingCashFlow'] / df['totalCurrentAssets']
        # Free Cash Flow Yield (if freeCashFlow exists)
        if 'freeCashFlow' in df.columns and 'grossPPE' in df.columns:
            df['Free_Cash_Flow_Yield'] = df['freeCashFlow'] / df['grossPPE'] * 100

    return df

# ————————————
# 3. KPI cards
# ————————————
def show_kpis(df):
    # pick up only numeric (float) columns
    num_cols = df.select_dtypes(float).columns
    if len(num_cols) == 0:
        st.warning("No numeric columns available for Key Performance Indicators.")
        return

    pct = df[num_cols].pct_change()
    avg_growth = pct.mean() * 100
    vol = pct.std() * 100

    st.subheader("Key Performance Indicators")
    # now len(num_cols) is guaranteed >= 1
    cols1 = st.columns(len(num_cols))
    for i, c in enumerate(num_cols):
        cols1[i].metric(c, f"{df[c].iloc[-1]:,.0f}", f"{avg_growth[c]:.2f}%")

    cols2 = st.columns(len(num_cols))
    for i, c in enumerate(num_cols):
        cols2[i].metric(f"{c} vol.", f"{vol[c]:.2f}%")

# ————————————
# 4. Time series plot
# ————————————
def ts_plot(df, cols):
    fig = px.line(df, x=df.index, y=cols, title="Time Series")
    st.plotly_chart(fig, use_container_width=True)



# ____________
st.sidebar.subheader("Model Hyperparameters")
arima_p = st.sidebar.number_input("ARIMA p", min_value=0, max_value=5, value=1, step=1)
arima_d = st.sidebar.number_input("ARIMA d", min_value=0, max_value=2, value=0, step=1)
arima_q = st.sidebar.number_input("ARIMA q", min_value=0, max_value=5, value=1, step=1)

lstm_seq_len = st.sidebar.number_input("LSTM Sequence Length", min_value=1, max_value=100, value=10, step=1)
lstm_epochs   = st.sidebar.number_input("LSTM Epochs", min_value=1, max_value=500, value=50, step=1)


# ————————————
# 5a. Prophet forecast
# ————————————
import plotly.express as px
from prophet import Prophet
prophet_seasonal = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
def forecast_prophet(df, col, freq):
    # prepare ds/y for Prophet
    ts = df[[col]].dropna().reset_index()
    # reset_index names index column 'date' or whatever; rename to ds/y
    idx_name = ts.columns[0]
    ts = ts.rename(columns={idx_name: 'ds', col: 'y'})
    if ts.empty:
        st.error("No data to forecast with Prophet.")
        return

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(ts)
    future = m.make_future_dataframe(periods=12, freq=freq)
    fc = m.predict(future)

    fig = px.line(fc, x='ds', y=['yhat','yhat_upper','yhat_lower'],
                  title=f"{col} – Prophet Forecast")
    st.plotly_chart(fig, use_container_width=True)

# ————————————
# 5b. ARIMA forecast
# ————————————
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

def forecast_arima(df, col, p, d, q, steps=12):
    series = df[col].dropna()
    # ← guard against empty or too‐short series
    if series.empty:
        st.error(f"No data in '{col}' to fit ARIMA.")
        return
    if len(series) <= (p + d + q):
        st.error(f"Not enough points ({len(series)}) for ARIMA({p},{d},{q}).")
        return

    model = ARIMA(series, order=(p, d, q)).fit()
    fc    = model.get_forecast(steps=steps).summary_frame()
    idx   = series.index.append(fc.index)
    vals  = np.concatenate([series.values, fc["mean"].values])

    fig = px.line(x=idx, y=vals, title=f"{col} – ARIMA({p},{d},{q})")
    st.plotly_chart(fig, use_container_width=True)

# ————————————
# 5c. PyTorch LSTM forecast
# ————————————
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

def forecast_lstm(df, features, target, freq, seq_len, epochs, lr=0.001):
    data = df[features + [target]].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, :-1])
        y.append(scaled[i, -1])
    X, y = map(np.array, (X, y))
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=16, shuffle=True)

    model = LSTMModel(input_size=X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss_fn(pred, yb).backward()
            opt.step()

    model.eval()
    seq = tx[-1:].clone()
    preds = []
    for _ in range(12):
        with torch.no_grad():
            p = model(seq).item()
        preds.append(p)
        seq = torch.cat([seq[:,1:,:], torch.tensor([[[*seq[0,-1,1:].numpy(), p]]], dtype=torch.float32)], dim=1)

    fut_idx = pd.date_range(df.index[-1], periods=13, freq=freq)[1:]
    inv = scaler.inverse_transform(np.hstack([np.zeros((12, scaled.shape[1]-1)), np.array(preds)[:,None]]))[:, -1]
    fig = px.line(x=df.index.append(fut_idx), y=np.concatenate([df[target].values, inv]),
                  title=f"{target} – LSTM")
    st.plotly_chart(fig, use_container_width=True)

# ————————————
# 6. Anomaly detection
# ————————————
def detect_anomalies(df, col):
    iso = IsolationForest(contamination=0.05)
    df['anomaly'] = iso.fit_predict(df[[col]].fillna(0))
    bad = df[df['anomaly'] == -1]
    fig = px.scatter(df, x=df.index, y=col, title=f"{col} Anomalies")
    fig.add_scatter(x=bad.index, y=bad[col], mode='markers',
                    marker=dict(color='red', size=8), name='Anomaly')
    st.plotly_chart(fig, use_container_width=True)

# ————————————
# 7. Clustering + PCA
# ————————————
def cluster_pca(df, n_clusters):
    # compute %‑change and pivot so rows=series, cols=time
    pct = df.select_dtypes(float).pct_change().dropna().T

    # guard against empty or too‑few series
    if pct.empty or pct.shape[0] < 2:
        st.error("Not enough numeric series for clustering & PCA.")
        return

    # warn if clusters > samples
    if n_clusters > pct.shape[0]:
        st.warning(f"Only {pct.shape[0]} series available but {n_clusters} clusters requested. Reducing clusters.")
        n_clusters = pct.shape[0]

    # run PCA + KMeans
    coords = PCA(n_components=2).fit_transform(pct)
    labels = KMeans(n_clusters=n_clusters).fit_predict(pct)

    # assemble results
    out = pd.DataFrame({
        'PC1': coords[:,0],
        'PC2': coords[:,1],
        'Cluster': labels
    }, index=pct.index)

    # plot
    fig = px.scatter(
        out, x='PC1', y='PC2',
        color='Cluster', hover_name=out.index,
        title="Clustering & PCA"
    )
    st.plotly_chart(fig, use_container_width=True)

# ————————————
# 8. Export report
# ————————————
def export_report(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Raw_Data')
    st.download_button("Download Excel Report",
                       data=buf.getvalue(),
                       file_name="financial_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ————————————
# Main App
# ————————————
st.title("🔍 Financial Statement Analyzer")
# 1) First-run tutorial
if 'tutorial_shown' not in st.session_state:
    with st.expander("👋 Welcome—Quick Tutorial"):
        st.markdown("""
        1. Upload your CSV or PDF  
        2. Map Date & core fields in sidebar  
        3. Review Raw Data, KPIs, Ratios  
        4. Add Custom Ratios, run Forecasts & Anomalies  
        5. Download your Excel report  
        """)
    st.session_state.tutorial_shown = True

# 2) Help & Glossary
with st.sidebar.expander("🛈 Help & Glossary", expanded=True):
    st.markdown("""
    **Data Requirements**  
    • A Date/Period column (YYYY‑MM‑DD or similar)  
    • Numeric values may include “₹”, “%”, “(…)”—we clean them  
    • Unmapped fields skip dependent ratios

    **Key Terms**  
    -Gross Margin = Gross Profit ÷ Revenue × 100  
    -ROE = Net Income ÷ Equity × 100  
    -FCF Yield = FCF ÷ PPE × 100
    """)
uploader = st.file_uploader(
    "Upload CSV or PDF",
    type=['csv','pdf'],
    key="uploader_main"
)
if uploader:
    df = load_data(uploader)

    # 3) Column Mapper
    st.sidebar.markdown("### 🔧 Column Mapper")
    cols = df.columns.tolist()
    mapping = {'date': st.sidebar.selectbox("Date column", ["<use index>"]+cols)}
    core = ['Net_Profit','Reserves','Borrowings','Total_Current_Assets','Other_Liabilities','Cash_and_Bank','Receivables']
    for field in core:
        mapping[field] = st.sidebar.selectbox(f"Map {field}", ["<none>"]+cols)
    if mapping['date'] != "<use index>":
        df.index = pd.to_datetime(df[mapping['date']], errors='coerce')
    for std, colname in mapping.items():
        if std!='date' and colname not in ("<none>","<use index>"):
            df.rename(columns={colname: std}, inplace=True)

    # 4) Detect statement type
    st.sidebar.markdown("### 📊 Statement Type")
    lc = [c.lower() for c in df.columns]
    if any("revenue" in c for c in lc) or "net_profit" in df.columns:
        st_type = "Income Statement"
    elif any("asset" in c for c in lc) and any("liability" in c for c in lc):
        st_type = "Balance Sheet"
    elif any("cash" in c for c in lc):
        st_type = "Cash Flow"
    else:
        st_type = "Unknown"
    st.sidebar.success(st_type)
    packs = {
      "Income Statement": "- Gross Margin\n- EBITDA Margin\n- Net Profit Margin",
      "Balance Sheet":   "- Current Ratio\n- Quick Ratio\n- Debt/Equity",
      "Cash Flow":       "- Operating Cash Flow Ratio\n- Free Cash Flow Yield"
    }
    st.sidebar.markdown(packs.get(st_type,""))

    # 5) Custom Ratio Builder
    st.sidebar.markdown("### ➕ Custom Ratio")
    num = st.sidebar.selectbox("Numerator", ["<none>"]+cols)
    den = st.sidebar.selectbox("Denominator", ["<none>"]+cols)
    cname = st.sidebar.text_input("Ratio Name")
    if st.sidebar.button("Add") and cname and num!="<none>" and den!="<none>":
        df[cname] = df[num].astype(float)/df[den].astype(float)
        st.sidebar.success(f"Added {cname}")

    # 6) Compute & show
    df = compute_ratios(df, st_type)
    st.subheader("Raw Data"); st.dataframe(df)
    show_kpis(df)

    # 7) Financial Ratios with inline tooltips
    ratio_defs = {
      'Gross_Margin': 'Gross Profit ÷ Revenue × 100',
      'EBITDA_Margin': 'EBITDA ÷ Revenue × 100',
      'Net_Profit_Margin':'Net Income ÷ Revenue × 100',
      'Current_Ratio':'Current Assets ÷ Current Liabilities',
      'Quick_Ratio':'(Cash+Receivables) ÷ Current Liabilities',
      'Debt_to_Equity':'Debt ÷ Equity',
      'ROE':'Net Income ÷ Equity × 100',
      'Operating_Cash_Flow_Ratio':'CFO ÷ Current Assets',
      'Free_Cash_Flow_Yield':'FCF ÷ PPE × 100'
    }
    if st_type=="Income Statement":
        to_plot = ['Gross_Margin','EBITDA_Margin','Net_Profit_Margin']
    elif st_type=="Balance Sheet":
        to_plot = ['Current_Ratio','Quick_Ratio','Debt_to_Equity','ROE']
    elif st_type=="Cash Flow":
        to_plot = ['Operating_Cash_Flow_Ratio','Free_Cash_Flow_Yield']
    else:
        to_plot = []
    if to_plot:
        missing = [r for r in to_plot if r not in df.columns]
        if not missing:
            st.subheader("🏷️ Financial Ratios")
            cols_html = st.columns(len(to_plot))
            for i,r in enumerate(to_plot):
                cols_html[i].markdown(
                  f"**{r}** <span title='{ratio_defs.get(r,'')}' style='cursor:help'>ⓘ</span>",
                  unsafe_allow_html=True
                )
            st.dataframe(df[to_plot])
            st.plotly_chart(px.line(df, x=df.index, y=to_plot, title="Ratios Over Time"), use_container_width=True)
        else:
            st.warning(f"Missing for ratios: {', '.join(missing)}")

    # 8) Time Series Plots
    st.subheader("Time Series Plots")
    numeric = df.select_dtypes(float).columns.tolist()
    sel = st.multiselect("Pick series", numeric, default=numeric[:2])
    if sel: ts_plot(df, sel)

# 9) Forecasting Models (with robust auto‑freq)
st.subheader("Forecasting Models")
numeric = df.select_dtypes(include=np.number).columns.tolist()
if not numeric:
    st.warning("No numeric series for forecasting.")
else:
    model  = st.selectbox("Model", ["Prophet", "ARIMA", "LSTM"])
    target = st.selectbox("Target", numeric)

    # auto‑detect freq
    try:
        df.index = pd.to_datetime(df.index, errors='coerce')
    except:
        pass

    freq_opts = ['Auto', 'D', 'M', 'Q', 'A']
    freq_sel  = st.selectbox("Frequency", freq_opts)
    if freq_sel == 'Auto' and len(df.index) > 1:
        try:
            delta = (df.index[1] - df.index[0]).days
            freq  = 'Q' if delta > 80 else 'M' if delta > 25 else 'D'
            st.caption(f"Auto‑freq = {freq}")
        except:
            freq = 'M'
            st.caption("Auto‑freq failed, using 'M'")
    else:
        freq = freq_sel

    if model == "Prophet" and st.button("Run Prophet"):
        forecast_prophet(df, target, freq)

    elif model == "ARIMA" and st.button("Run ARIMA"):
        p = st.sidebar.number_input("p", 0, 5, 1)
        d = st.sidebar.number_input("d", 0, 2, 0)
        q = st.sidebar.number_input("q", 0, 5, 1)
        forecast_arima(df, target, p, d, q)

    elif model == "LSTM" and st.button("Run LSTM"):
        feats = st.multiselect(
            "Features (exclude target)",
            [c for c in numeric if c != target],
            default=[c for c in numeric if c != target]
        )
        seq_len = st.sidebar.number_input("Seq Len", 1, 100, 10)
        epochs  = st.sidebar.number_input("Epochs", 1, 500, 50)
        forecast_lstm(df, feats, target, freq, seq_len=seq_len, epochs=epochs)

    # 10) Anomalies
    st.subheader("Anomaly Detection")
    ac = st.selectbox("Series", numeric, key="anom")
    if st.button("Detect Anomalies"):
        detect_anomalies(df, ac)

    # 11) Clustering & PCA
    st.subheader("Clustering & PCA")
    k = st.slider("Clusters",2,5,3)
    if st.button("Run Clustering"):
        cluster_pca(df, k)

    # 12) Export
    st.subheader("Export Report")
    export_report(df)