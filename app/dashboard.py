import streamlit as st
import httpx
import pandas as pd

API_BASE = "http://localhost:8000/v1"

st.set_page_config(page_title="Semantic Cache Telemetry", layout="wide")
st.title("⚡ Semantic Cache Proxy Dashboard")
st.caption("Real-time telemetry and vector similarity monitoring for local Ollama models")

# Sidebar Controls
st.sidebar.header("Proxy Controls")
if st.sidebar.button("🔄 Refresh Stats"):
    st.rerun()

if st.sidebar.button("🧹 Reset Telemetry Counters"):
    try:
        httpx.post(f"{API_BASE}/analytics/reset")
        st.sidebar.success("Telemetry reset successfully!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to reset: {e}")

# Fetch Current Metrics
try:
    resp = httpx.get(f"{API_BASE}/analytics/stats", timeout=5.0)
    stats = resp.json()
except Exception:
    st.error("Cannot connect to FastAPI proxy. Ensure it is running on port 8000.")
    st.stop()

# Key Performance Indicators
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Requests", stats.get("total_requests", 0))
col2.metric("Cache Hits", stats.get("hits", 0))
col3.metric("Cache Misses", stats.get("misses", 0))
col4.metric("Hit Ratio", f"{stats.get('hit_ratio_percent', 0.0)}%")

st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Compute & Latency Savings")
    saved_sec = stats.get("estimated_saved_seconds", 0.0)
    st.info(f"⏱️ **Estimated Inference Time Saved:** {saved_sec:.2f} seconds")

    chart_data = pd.DataFrame({
        "Type": ["Hits", "Misses", "Bypasses"],
        "Count": [stats.get("hits", 0), stats.get("misses", 0), stats.get("bypasses", 0)]
    }).set_index("Type")
    st.bar_chart(chart_data)

with col_right:
    st.subheader("Live Query Tester")
    user_query = st.text_input("Enter a prompt to test cache behavior:", "What is Python?")
    threshold = st.slider("Similarity Threshold", min_value=0.70, max_value=0.99, value=0.88, step=0.01)
    bypass = st.checkbox("Bypass Cache (X-Cache-Bypass)")

    if st.button("Send Test Request"):
        headers = {
            "X-Similarity-Threshold": str(threshold),
            "X-Cache-Bypass": "true" if bypass else "false",
        }
        payload = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": user_query}],
            "temperature": 0.0,
        }
        try:
            test_resp = httpx.post(f"{API_BASE}/chat/completions", json=payload, headers=headers, timeout=60.0)
            res_headers = test_resp.headers
            res_json = test_resp.json()

            lookup_status = res_headers.get("X-Cache-Lookup", "UNKNOWN")
            latency = res_headers.get("X-Lookup-Latency-ms", "0")
            similarity = res_headers.get("X-Cache-Similarity", "0.0")

            if lookup_status == "HIT":
                st.success(f"**HIT** — Served in {latency} ms (Similarity: {similarity})")
            elif lookup_status == "BYPASS":
                st.warning(f"**BYPASSED** — Upstream executed")
            else:
                st.info(f"**MISS** — Upstream executed in {latency} ms and indexed")

            assistant_msg = res_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            st.text_area("LLM Response", assistant_msg, height=120)
        except Exception as e:
            st.error(f"Request failed: {e}")