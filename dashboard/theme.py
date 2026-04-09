from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg-app: #0a0a0c;
          --bg-shell: #111113;
          --bg-card: #151517;
          --bg-elevated: #1b1b1f;
          --bg-panel: rgba(15, 15, 17, 0.94);
          --border-soft: rgba(255,255,255,0.08);
          --border-strong: rgba(231,163,107,0.16);
          --text-main: #f4f1ec;
          --text-sub: #b8aea4;
          --text-muted: #7e756d;
          --accent-warm: #a56a43;
          --accent-glow: #e7a36b;
          --accent-copper: #5b3424;
          --accent-cream: #f5e6d6;
          --chart-pink: #f24ccf;
          --chart-rose: #ff5b7f;
          --chart-blue: #2f7dff;
          --chart-violet: #7a6cff;
          --success: #32d39a;
        }

        html, body, [class*="css"]  {
          font-family: "Segoe UI Variable", "Aptos", "Trebuchet MS", sans-serif;
        }

        .stApp {
          background:
            radial-gradient(circle at 10% 0%, rgba(245, 230, 214, 0.42), transparent 20%),
            radial-gradient(circle at 28% 0%, rgba(232, 161, 95, 0.55), transparent 25%),
            radial-gradient(circle at 92% 0%, rgba(183, 91, 58, 0.22), transparent 24%),
            radial-gradient(circle at 0% 100%, rgba(122, 108, 255, 0.14), transparent 18%),
            linear-gradient(180deg, #1b1212 0%, #0a0a0c 32%, #0b0912 100%);
          color: var(--text-main);
        }

        [data-testid="stAppViewContainer"] {
          background: transparent;
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        .stDeployButton,
        #MainMenu,
        footer {
          display: none !important;
        }

        [data-testid="stSidebar"] {
          background: rgba(17,17,19,0.96);
          border-right: 1px solid rgba(255,255,255,0.06);
          backdrop-filter: blur(18px);
          min-width: 18rem;
          max-width: 18rem;
        }

        [data-testid="stSidebar"] * {
          color: var(--text-main);
        }

        .block-container {
          padding-top: 1.65rem;
          padding-bottom: 2rem;
          max-width: 1380px;
        }

        [data-testid="stMainBlockContainer"] {
          position: relative;
        }

        [data-testid="stMainBlockContainer"] .block-container {
          background: linear-gradient(180deg, rgba(12,12,14,0.94), rgba(10,10,12,0.96));
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 30px;
          box-shadow: 0 25px 80px rgba(0,0,0,0.32), inset 0 1px 0 rgba(255,255,255,0.03);
          padding-left: 1.7rem;
          padding-right: 1.7rem;
        }

        h1, h2, h3, h4 {
          color: var(--text-main);
          letter-spacing: -0.02em;
        }

        p, li, label, .stMarkdown, .stCaption {
          color: var(--text-sub);
        }

        [data-testid="stMetric"] {
          background: linear-gradient(180deg, rgba(24,24,27,0.98), rgba(17,17,19,0.98));
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 22px;
          padding: 1rem 1rem;
          min-height: 118px;
          box-shadow: 0 18px 50px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.03);
        }

        [data-testid="stMetricLabel"] {
          color: var(--text-muted);
          font-size: 0.76rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }

        [data-testid="stMetricValue"] {
          color: var(--text-main);
          font-size: 1.55rem;
          letter-spacing: -0.03em;
        }

        .dashboard-stat-card {
          min-height: 116px;
          background: linear-gradient(180deg, rgba(24,24,27,0.98), rgba(17,17,19,0.98));
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 22px;
          padding: 1.1rem 1rem;
          box-shadow: 0 18px 50px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.03);
        }

        .dashboard-stat-card-compact {
          min-height: 104px;
          padding: 1rem 0.95rem;
          margin-bottom: 0.32rem;
        }

        .dashboard-stat-label {
          color: var(--text-muted);
          font-size: 0.76rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          margin-bottom: 0.7rem;
        }

        .dashboard-stat-value {
          color: var(--text-main);
          font-size: clamp(1.6rem, 1.8vw, 2rem);
          line-height: 1.15;
          letter-spacing: -0.045em;
          word-break: break-word;
          overflow-wrap: anywhere;
        }

        .dashboard-stat-value-compact {
          font-size: clamp(1.16rem, 1.35vw, 1.45rem);
          line-height: 1.28;
          letter-spacing: -0.028em;
        }

        .dashboard-stat-value-positive {
          color: #42d39f;
          text-shadow: 0 0 22px rgba(66, 211, 159, 0.12);
        }

        .dashboard-stat-value-negative {
          color: #ff7a8e;
          text-shadow: 0 0 22px rgba(255, 122, 142, 0.12);
        }

        .dashboard-delete-spacer {
          height: 1.9rem;
        }

        .stButton > button,
        .stDownloadButton > button {
          background: linear-gradient(135deg, rgba(165,106,67,0.36), rgba(91,52,36,0.28));
          border: 1px solid rgba(231,163,107,0.22);
          border-radius: 15px;
          color: var(--text-main);
          min-height: 2.9rem;
          font-weight: 500;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
          border-color: rgba(231,163,107,0.42);
          box-shadow: 0 0 0 1px rgba(231,163,107,0.18), 0 10px 28px rgba(165,106,67,0.18);
        }

        .stTextInput > div > div > input,
        .stNumberInput input,
        .stDateInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        .stTextArea textarea {
          background: rgba(21,21,23,0.92);
          border: 1px solid var(--border-soft);
          color: var(--text-main);
          border-radius: 14px;
        }

        .stTextInput > div, .stNumberInput > div, .stDateInput > div, .stSelectbox > div {
          border-radius: 14px;
        }

        .stDataFrame, div[data-testid="stTable"], [data-testid="stJson"] {
          background: rgba(21,21,23,0.92);
          border: 1px solid var(--border-soft);
          border-radius: 18px;
          overflow: hidden;
        }

        [data-baseweb="tab-list"] {
          gap: 0.35rem;
          background: rgba(21,21,23,0.92);
          padding: 0.35rem;
          border-radius: 16px;
          border: 1px solid var(--border-soft);
        }

        button[data-baseweb="tab"] {
          border-radius: 12px;
          color: var(--text-sub);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
          background: linear-gradient(135deg, rgba(165,106,67,0.34), rgba(91,52,36,0.24));
          color: var(--text-main);
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
          background: linear-gradient(180deg, rgba(21,21,23,0.98), rgba(15,15,17,0.98));
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 24px;
          box-shadow: 0 18px 50px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.03);
          padding: 0.1rem 0.25rem 0.5rem 0.25rem;
        }

        [data-testid="stExpander"] {
          background: linear-gradient(180deg, rgba(21,21,23,0.98), rgba(15,15,17,0.98));
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 20px;
        }

        .dashboard-kicker {
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.10em;
          font-size: 0.72rem;
        }

        .dashboard-panel-title {
          color: var(--text-main);
          font-size: 1.1rem;
          font-weight: 600;
          margin-bottom: 0.45rem;
        }

        .dashboard-help {
          color: var(--text-sub);
          font-size: 0.92rem;
        }

        .dashboard-topbar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 1rem;
          margin-bottom: 1rem;
          color: var(--text-muted);
        }

        .dashboard-breadcrumb {
          color: var(--text-muted);
          font-size: 0.95rem;
        }

        .dashboard-status-chip,
        .dashboard-hero-chip,
        .dashboard-panel-action {
          border-radius: 14px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(165,106,67,0.24), rgba(91,52,36,0.18));
          color: var(--text-main);
          padding: 0.7rem 0.95rem;
          font-size: 0.92rem;
          white-space: nowrap;
        }

        .dashboard-hero {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .dashboard-hero-left {
          display: flex;
          align-items: center;
          gap: 1rem;
          min-width: 0;
        }

        .dashboard-hero-icon {
          width: 52px;
          height: 52px;
          border-radius: 16px;
          background: linear-gradient(135deg, rgba(242,76,207,0.88), rgba(122,108,255,0.88));
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-size: 1.25rem;
          box-shadow: 0 12px 30px rgba(242,76,207,0.24);
        }

        .dashboard-hero-copy {
          min-width: 0;
        }

        .dashboard-hero-title {
          color: var(--text-main);
          font-size: 2.22rem;
          line-height: 1.04;
          letter-spacing: -0.04em;
          font-weight: 600;
        }

        .dashboard-hero-subtitle {
          color: var(--text-sub);
          font-size: 0.98rem;
          margin-top: 0.18rem;
        }

        .dashboard-page-heading {
          margin-bottom: 1.2rem;
        }

        .dashboard-page-heading-row {
          display: flex;
          align-items: center;
          gap: 0.8rem;
          margin-bottom: 0.18rem;
        }

        .dashboard-page-icon {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 3.15rem;
          height: 3.15rem;
          border-radius: 16px;
          background: linear-gradient(135deg, rgba(242,76,207,0.28), rgba(122,108,255,0.24));
          border: 1px solid rgba(255,255,255,0.08);
          font-size: 1.45rem;
          line-height: 1;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }

        .dashboard-page-tabs {
          display: inline-flex;
          gap: 0.45rem;
          flex-wrap: wrap;
          background: rgba(21,21,23,0.92);
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 16px;
          padding: 0.35rem;
          margin-bottom: 1.15rem;
        }

        .dashboard-page-tab {
          padding: 0.68rem 1rem;
          border-radius: 12px;
          color: var(--text-sub);
          font-size: 0.93rem;
          background: transparent;
        }

        .dashboard-page-tab.active {
          background: linear-gradient(135deg, rgba(165,106,67,0.34), rgba(91,52,36,0.24));
          color: var(--text-main);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }

        .dashboard-panel-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 1rem;
          margin-bottom: 0.8rem;
        }

        .sidebar-brand {
          display: flex;
          align-items: center;
          gap: 0.85rem;
          margin-bottom: 1rem;
          padding-top: 0.35rem;
        }

        .sidebar-brand-mark {
          position: relative;
          width: 44px;
          height: 44px;
        }

        .sidebar-brand-mark .ring {
          position: absolute;
          border: 2px solid var(--accent-cream);
          border-radius: 50%;
          inset: 0;
          opacity: 0.92;
        }

        .sidebar-brand-mark .ring-two {
          inset: 8px;
          border-color: var(--accent-glow);
        }

        .sidebar-brand-mark .ring-three {
          inset: 16px;
          border-color: var(--accent-warm);
        }

        .sidebar-brand-title {
          color: var(--text-main);
          font-size: 1.28rem;
          font-weight: 600;
          letter-spacing: -0.03em;
        }

        .sidebar-brand-subtitle {
          color: var(--text-muted);
          font-size: 0.82rem;
        }

        [data-testid="stSidebar"] .stButton > button {
          justify-content: flex-start;
          width: 100%;
          min-height: 2.85rem;
          padding: 0.72rem 1rem;
          border-radius: 15px;
          background: transparent;
          border: 1px solid transparent;
          box-shadow: none;
          position: relative;
          transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }

        [data-testid="stSidebar"] .stButton {
          margin-bottom: 0.32rem;
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
          background:
            radial-gradient(circle at 72% 50%, rgba(231,163,107,0.14), transparent 18%),
            linear-gradient(90deg, rgba(165,106,67,0.42), rgba(91,52,36,0.18));
          border-color: rgba(231,163,107,0.24);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 8px 24px rgba(0,0,0,0.16);
        }

        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
          background: rgba(255,255,255,0.025);
          border-color: rgba(255,255,255,0.06);
        }

        [data-testid="stSidebar"] .stButton > button p {
          color: var(--text-main);
          font-size: 0.98rem;
          font-weight: 500;
          letter-spacing: -0.01em;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        [data-testid="stSidebar"] .dashboard-kicker {
          margin: 0.35rem 0 0.9rem;
          letter-spacing: 0.18em;
          font-size: 0.68rem;
        }

        .sidebar-mini-chart {
          position: relative;
          height: 7.2rem;
          margin: 0.1rem 0 1.1rem;
          padding: 0.9rem 0.85rem 0.8rem;
          border-radius: 18px;
          border: 1px solid rgba(255,255,255,0.07);
          background:
            radial-gradient(circle at 78% 28%, rgba(242,76,207,0.16), transparent 26%),
            radial-gradient(circle at 24% 22%, rgba(231,163,107,0.18), transparent 28%),
            linear-gradient(180deg, rgba(22,22,25,0.98), rgba(15,15,17,0.96));
          overflow: hidden;
        }

        .sidebar-mini-chart::after {
          content: "";
          position: absolute;
          inset: auto 0 0 0;
          height: 45%;
          background: linear-gradient(180deg, transparent, rgba(0,0,0,0.18));
          pointer-events: none;
        }

        .sidebar-mini-chart-bars {
          position: absolute;
          left: 0.85rem;
          right: 0.85rem;
          bottom: 0.8rem;
          top: 1.2rem;
          display: flex;
          align-items: flex-end;
          gap: 0.33rem;
        }

        .sidebar-mini-chart-bars span {
          flex: 1 1 0;
          border-radius: 999px 999px 4px 4px;
          background: linear-gradient(180deg, #ff7b92 0%, #f24ccf 38%, #7a6cff 68%, #2f7dff 100%);
          box-shadow: 0 0 18px rgba(122,108,255,0.16);
          opacity: 0.96;
        }

        .sidebar-footer {
          margin-top: 2.5rem;
          padding-top: 1rem;
          border-top: 1px solid rgba(255,255,255,0.06);
        }

        .sidebar-footer-title {
          color: var(--text-main);
          font-size: 0.98rem;
          margin-bottom: 0.75rem;
        }

        .sidebar-footer-link {
          color: var(--text-sub);
          font-size: 0.88rem;
          margin-bottom: 0.5rem;
        }

        .stAlert {
          border-radius: 18px;
        }

        .dashboard-mpl-chart {
          width: 100%;
        }

        .dashboard-mpl-chart svg {
          width: 100%;
          height: auto;
          display: block;
        }

        @media (max-width: 960px) {
          [data-testid="stMainBlockContainer"] .block-container {
            border-radius: 22px;
            padding-left: 1rem;
            padding-right: 1rem;
          }

          .dashboard-hero {
            flex-direction: column;
            align-items: flex-start;
          }

          .dashboard-hero-title {
            font-size: 1.55rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
