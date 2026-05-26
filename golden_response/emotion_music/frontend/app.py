"""
Streamlit Frontend Dashboard
Real-time emotion detection + music generation UI.
"""

import streamlit as st
import cv2
import numpy as np
import requests
import base64
import json
import time
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from PIL import Image
import threading

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎵 Emotion Music Composer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = "http://localhost:8000"
EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised", "Calm"]

EMOTION_COLORS = {
    "Happy":     "#FFD700",
    "Sad":       "#4169E1",
    "Angry":     "#FF4500",
    "Calm":      "#32CD32",
    "Fearful":   "#9370DB",
    "Disgusted": "#8B4513",
    "Surprised": "#FF69B4",
    "Neutral":   "#808080"
}

EMOTION_VALENCE = {
    "Happy": (0.8, 0.7),   "Sad": (-0.7, -0.3),
    "Angry": (-0.5, 0.9),  "Calm": (0.4, -0.6),
    "Fearful": (-0.6, 0.5),"Disgusted": (-0.4, 0.2),
    "Surprised": (0.3, 0.8),"Neutral": (0.0, 0.0)
}


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .emotion-badge {
        padding: 8px 16px; border-radius: 20px;
        font-size: 18px; font-weight: bold;
        display: inline-block; margin: 4px;
    }
    .metric-card {
        background: #1e2130; border-radius: 12px;
        padding: 16px; text-align: center;
    }
    .stButton > button {
        border-radius: 8px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ── State Init ────────────────────────────────────────────────────────────────

if "session_id"         not in st.session_state: st.session_state.session_id = None
if "running"            not in st.session_state: st.session_state.running = False
if "emotion_history"    not in st.session_state: st.session_state.emotion_history = []
if "current_emotion"    not in st.session_state: st.session_state.current_emotion = "Neutral"
if "current_probs"      not in st.session_state: st.session_state.current_probs = {e: 1/8 for e in EMOTIONS}
if "music_generated"    not in st.session_state: st.session_state.music_generated = False
if "genre"              not in st.session_state: st.session_state.genre = "Classical"


# ── Helpers ──────────────────────────────────────────────────────────────────

def encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()

def call_api(endpoint: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{API_BASE}/{endpoint}", json=payload, timeout=5)
        return r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def create_session():
    r = requests.post(f"{API_BASE}/session/create",
                      params={"genre": st.session_state.genre})
    data = r.json()
    st.session_state.session_id = data["session_id"]
    return data["session_id"]


# ── Valence-Arousal Wheel ─────────────────────────────────────────────────────

def draw_emotion_wheel(current_emotion: str, probs: dict) -> go.Figure:
    fig = go.Figure()

    # Background quadrant labels
    fig.add_annotation(x=0.6, y=0.6,   text="Excited",    showarrow=False, font=dict(color="#555", size=11))
    fig.add_annotation(x=-0.6, y=0.6,  text="Distressed", showarrow=False, font=dict(color="#555", size=11))
    fig.add_annotation(x=0.6, y=-0.6,  text="Relaxed",    showarrow=False, font=dict(color="#555", size=11))
    fig.add_annotation(x=-0.6, y=-0.6, text="Bored",      showarrow=False, font=dict(color="#555", size=11))

    # Plot all emotions as small circles
    for e, (v, a) in EMOTION_VALENCE.items():
        size  = max(8, probs.get(e, 0) * 60)
        color = EMOTION_COLORS.get(e, "#aaa")
        fig.add_trace(go.Scatter(
            x=[v], y=[a], mode="markers+text",
            marker=dict(size=size, color=color, opacity=0.7,
                        line=dict(color="white", width=1)),
            text=[e], textposition="top center",
            textfont=dict(size=9, color="white"),
            name=e, showlegend=False
        ))

    # Highlight current emotion
    if current_emotion in EMOTION_VALENCE:
        v, a = EMOTION_VALENCE[current_emotion]
        fig.add_trace(go.Scatter(
            x=[v], y=[a], mode="markers",
            marker=dict(size=30, color=EMOTION_COLORS.get(current_emotion, "white"),
                        symbol="star", line=dict(color="white", width=2)),
            name="Current", showlegend=False
        ))

    # Axes
    fig.add_hline(y=0, line_dash="dot", line_color="#444")
    fig.add_vline(x=0, line_dash="dot", line_color="#444")

    fig.update_layout(
        xaxis=dict(range=[-1.1, 1.1], title="Valence →", showgrid=False,
                   color="white", zeroline=False),
        yaxis=dict(range=[-1.1, 1.1], title="Arousal ↑", showgrid=False,
                   color="white", zeroline=False),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=320, margin=dict(l=40, r=20, t=20, b=40)
    )
    return fig


# ── Confidence Bars ───────────────────────────────────────────────────────────

def draw_confidence_bars(probs: dict) -> go.Figure:
    emotions = list(probs.keys())
    values   = [probs[e] * 100 for e in emotions]
    colors   = [EMOTION_COLORS.get(e, "#aaa") for e in emotions]

    fig = go.Figure(go.Bar(
        x=values, y=emotions, orientation='h',
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside", textfont=dict(color="white", size=10)
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 110], showgrid=False, color="white", title="Confidence %"),
        yaxis=dict(color="white"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=280, margin=dict(l=10, r=60, t=10, b=30)
    )
    return fig


# ── Emotion Timeline ──────────────────────────────────────────────────────────

def draw_timeline(history: list) -> go.Figure:
    if not history:
        return go.Figure()

    timestamps = [h["timestamp"] for h in history]
    emotions   = [h["emotion"] for h in history]
    confs      = [h["confidence"] for h in history]
    colors_    = [EMOTION_COLORS.get(e, "#aaa") for e in emotions]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=emotions,
        mode="lines+markers",
        marker=dict(color=colors_, size=10),
        line=dict(color="#444", width=2),
        text=[f"{e} ({c:.0%})" for e, c in zip(emotions, confs)],
        hovertemplate="%{text}<br>%{x}<extra></extra>"
    ))
    fig.update_layout(
        xaxis=dict(color="white", title="Time"),
        yaxis=dict(color="white", categoryorder="array",
                   categoryarray=EMOTIONS),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=220, margin=dict(l=10, r=10, t=10, b=40)
    )
    return fig


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🎵 Emotion-Driven Music Composer")
st.caption("Real-time emotion detection → original music generation")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Controls")
    st.session_state.genre = st.selectbox(
        "🎼 Genre", ["Classical", "Ambient", "Jazz", "Lo-Fi", "Cinematic"]
    )
    volume = st.slider("🔊 Volume", 0, 100, 70)

    st.divider()
    if not st.session_state.running:
        if st.button("▶️ Start Session", use_container_width=True):
            sid = create_session()
            st.session_state.running = True
            st.session_state.emotion_history = []
            st.success(f"Session started: `{sid[:8]}...`")
    else:
        if st.button("⏹️ Stop Session", use_container_width=True):
            st.session_state.running = False
            requests.post(f"{API_BASE}/playback/stop")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏸️ Pause"):
            requests.post(f"{API_BASE}/playback/pause")
    with col2:
        if st.button("▶️ Resume"):
            requests.post(f"{API_BASE}/playback/resume")

    st.divider()
    if st.session_state.session_id:
        st.subheader("📥 Download")
        sid = st.session_state.session_id
        st.markdown(f"[⬇️ Download MIDI]({API_BASE}/download/midi/{sid})")
        st.markdown(f"[⬇️ Download WAV]({API_BASE}/download/wav/{sid})")

    st.divider()
    st.subheader("📊 Session Info")
    if st.session_state.session_id:
        st.code(f"ID: {st.session_state.session_id[:8]}...")
    st.metric("Events Logged", len(st.session_state.emotion_history))


# ── Main Columns ──────────────────────────────────────────────────────────────

col_cam, col_viz = st.columns([1, 1])

with col_cam:
    st.subheader("📷 Live Feed")
    cam_placeholder     = st.empty()
    emotion_placeholder = st.empty()

with col_viz:
    st.subheader("🎭 Emotion Wheel")
    wheel_placeholder = st.empty()
    st.subheader("📊 Confidence")
    bar_placeholder   = st.empty()

st.subheader("⏱️ Session Timeline")
timeline_placeholder = st.empty()

st.subheader("🎵 Music Status")
music_placeholder = st.empty()

# ── Live Loop ─────────────────────────────────────────────────────────────────

if st.session_state.running and st.session_state.session_id:
    cap = cv2.VideoCapture(0)
    last_music_time = 0
    music_interval  = 12   # Generate new music every 12 seconds

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Cannot access webcam.")
            break

        frame_b64 = encode_frame(frame)

        # Call detect-emotion API
        result = call_api("detect-emotion", {
            "frame_b64":  frame_b64,
            "session_id": st.session_state.session_id
        })

        if result.get("success"):
            fusion = result["result"]
            emotion = fusion.get("dominant_emotion", "Neutral")
            probs   = fusion.get("probabilities", {e: 1/8 for e in EMOTIONS})
            conf    = fusion.get("confidence", 0.0)

            st.session_state.current_emotion = emotion
            st.session_state.current_probs   = probs
            st.session_state.emotion_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "emotion":   emotion,
                "confidence":conf
            })

            # Display frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(rgb, channels="RGB", use_column_width=True)

            color = EMOTION_COLORS.get(emotion, "#aaa")
            emotion_placeholder.markdown(
                f'<div class="emotion-badge" style="background:{color};color:#000">'
                f'🎭 {emotion} — {conf:.0%}</div>',
                unsafe_allow_html=True
            )

            # Update charts
            wheel_placeholder.plotly_chart(
                draw_emotion_wheel(emotion, probs),
                use_container_width=True, key=f"wheel_{time.time()}"
            )
            bar_placeholder.plotly_chart(
                draw_confidence_bars(probs),
                use_container_width=True, key=f"bar_{time.time()}"
            )
            timeline_placeholder.plotly_chart(
                draw_timeline(st.session_state.emotion_history[-30:]),
                use_container_width=True, key=f"tl_{time.time()}"
            )

            # Generate music periodically or on big emotion shift
            now = time.time()
            if now - last_music_time > music_interval:
                music_result = call_api("generate-music", {
                    "session_id":     st.session_state.session_id,
                    "emotion_result": fusion,
                    "genre":          st.session_state.genre,
                    "save_midi":      True
                })
                if music_result.get("success"):
                    last_music_time = now
                    music_placeholder.success(
                        f"🎵 Generated — Emotion: **{music_result['emotion']}** | "
                        f"Genre: **{music_result['genre']}** | "
                        f"Tempo: **{music_result['tempo']} BPM** | "
                        f"Tokens: **{music_result['num_tokens']}**"
                    )

        time.sleep(0.15)  # ~6 FPS

    cap.release()

elif not st.session_state.running:
    # Show placeholder when not running
    col_cam.info("👆 Click **Start Session** in the sidebar to begin.")

    if st.session_state.emotion_history:
        timeline_placeholder.plotly_chart(
            draw_timeline(st.session_state.emotion_history),
            use_container_width=True
        )
