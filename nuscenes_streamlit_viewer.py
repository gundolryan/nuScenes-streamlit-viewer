import base64
import json
import os
import threading
import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from nuscenes.nuscenes import NuScenes

CAMS = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]

DEFAULT_DATAROOT = "input your directory"
DEFAULT_VERSION = "v1.0-trainval"
DEFAULT_FPS = 12
TILE_W, TILE_H = 480, 270
PRELOAD_BEFORE = 10
PRELOAD_AFTER = 10

# Keep deployment logs concise.
for _name in [
    "streamlit",
    "streamlit.runtime",
    "streamlit.runtime.scriptrunner_utils.script_run_context",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)


@st.cache_resource(show_spinner=False)
def load_nusc(version: str, dataroot: str) -> NuScenes:
    return NuScenes(version=version, dataroot=dataroot, verbose=False)


@st.cache_data(show_spinner=False)
def scene_name_list(version: str, dataroot: str) -> List[str]:
    nusc = load_nusc(version, dataroot)
    return sorted([s["name"] for s in nusc.scene])


def get_scene_by_name(nusc: NuScenes, scene_name: str) -> Optional[dict]:
    for s in nusc.scene:
        if s["name"] == scene_name:
            return s
    return None


@st.cache_data(show_spinner=False)
def build_fullstream_tokens(version: str, dataroot: str, scene_name: str) -> List[Dict[str, str]]:
    nusc = load_nusc(version, dataroot)
    scene = get_scene_by_name(nusc, scene_name)
    if scene is None:
        return []

    first_sample = nusc.get("sample", scene["first_sample_token"])
    curr = {cam: first_sample["data"][cam] for cam in CAMS}

    out = []
    while True:
        if any(not tok for tok in curr.values()):
            break
        out.append(curr.copy())
        nxt = {}
        for cam in CAMS:
            sd = nusc.get("sample_data", curr[cam])
            nxt[cam] = sd["next"]
        curr = nxt
    return out


@st.cache_data(show_spinner=False)
def build_keyframe_tokens(version: str, dataroot: str, scene_name: str) -> List[Dict[str, str]]:
    nusc = load_nusc(version, dataroot)
    scene = get_scene_by_name(nusc, scene_name)
    if scene is None:
        return []
    out = []
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        out.append({cam: sample["data"][cam] for cam in CAMS})
        sample_token = sample["next"]
    return out


@st.cache_data(show_spinner=False)
def build_sample_frame_indices(version: str, dataroot: str, scene_name: str) -> List[int]:
    nusc = load_nusc(version, dataroot)
    full = build_fullstream_tokens(version, dataroot, scene_name)
    key = build_keyframe_tokens(version, dataroot, scene_name)
    if not full or not key:
        return []

    # Build frame index and timestamp arrays on CAM_FRONT stream.
    front_to_idx = {t["CAM_FRONT"]: i for i, t in enumerate(full)}
    full_front_tokens = [t["CAM_FRONT"] for t in full]
    full_ts = [
        int(nusc.get("sample_data", tok)["timestamp"])
        for tok in full_front_tokens
    ]

    indices: List[int] = []
    for t in key:
        front_tok = t["CAM_FRONT"]
        if front_tok in front_to_idx:
            indices.append(int(front_to_idx[front_tok]))
            continue

        # Fallback: find nearest frame by timestamp so sample count stays aligned
        # with scene keyframes even when one camera stream is slightly shorter.
        key_ts = int(nusc.get("sample_data", front_tok)["timestamp"])
        nearest = min(
            range(len(full_ts)),
            key=lambda i: abs(full_ts[i] - key_ts),
        )
        indices.append(int(nearest))
    return indices


@st.cache_data(show_spinner=False)
def build_sample_markers(version: str, dataroot: str, scene_name: str) -> List[dict]:
    nusc = load_nusc(version, dataroot)
    key = build_keyframe_tokens(version, dataroot, scene_name)
    indices = build_sample_frame_indices(version, dataroot, scene_name)
    n = min(len(key), len(indices))
    markers = []
    for i in range(n):
        front_tok = key[i]["CAM_FRONT"]
        sd = nusc.get("sample_data", front_tok)
        markers.append(
            {
                "sample_idx_1based": i + 1,
                "frame_idx": int(indices[i]),
                "timestamp_us": int(sd["timestamp"]),
            }
        )
    return markers


def _read_cam_rgb(nusc: NuScenes, dataroot: str, sd_token: str, cam_name: str) -> np.ndarray:
    sd = nusc.get("sample_data", sd_token)
    img_path = os.path.join(dataroot, sd["filename"])
    bgr = cv2.imread(img_path)
    if bgr is None:
        bgr = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    else:
        bgr = cv2.resize(bgr, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    cv2.putText(bgr, cam_name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _frame_to_b64_jpg(rgb: np.ndarray, quality: int = 80) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _precompute_scene_b64_frames_raw(
    version: str,
    dataroot: str,
    scene_name: str,
    max_frames: int = 0,
) -> List[str]:
    # No Streamlit cache calls here (safe for background thread usage).
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    scene = get_scene_by_name(nusc, scene_name)
    if scene is None:
        return []

    first_sample = nusc.get("sample", scene["first_sample_token"])
    curr = {cam: first_sample["data"][cam] for cam in CAMS}

    tokens: List[Dict[str, str]] = []
    while True:
        if any(not tok for tok in curr.values()):
            break
        tokens.append(curr.copy())
        nxt = {}
        for cam in CAMS:
            sd = nusc.get("sample_data", curr[cam])
            nxt[cam] = sd["next"]
        curr = nxt

    if max_frames and max_frames > 0:
        tokens = tokens[:max_frames]

    out: List[str] = []
    for t in tokens:
        imgs = [_read_cam_rgb(nusc, dataroot, t[c], c) for c in CAMS]
        top = np.hstack(imgs[:3])
        bot = np.hstack(imgs[3:])
        grid = np.vstack([top, bot])
        out.append(_frame_to_b64_jpg(grid, quality=80))
    return out


@st.cache_data(show_spinner=True)
def precompute_scene_b64_frames(
    version: str,
    dataroot: str,
    scene_name: str,
    max_frames: int = 0,
) -> List[str]:
    nusc = load_nusc(version, dataroot)
    tokens = build_fullstream_tokens(version, dataroot, scene_name)
    if max_frames and max_frames > 0:
        tokens = tokens[:max_frames]

    b64_frames: List[str] = []
    for t in tokens:
        imgs = [_read_cam_rgb(nusc, dataroot, t[c], c) for c in CAMS]
        top = np.hstack(imgs[:3])
        bot = np.hstack(imgs[3:])
        grid = np.vstack([top, bot])
        b64_frames.append(_frame_to_b64_jpg(grid, quality=80))

    return b64_frames


def render_player_html(
    frame_b64_list: List[str],
    sample_frame_indices: List[int],
    sample_markers: List[dict],
    fps_default: int,
    width_px: int = 1460,
):
    data_json = json.dumps(frame_b64_list)
    sample_json = json.dumps(sample_frame_indices)
    marker_json = json.dumps(sample_markers)
    fps_options = [8, 10, 12, 15]
    if fps_default not in fps_options:
        fps_options.append(fps_default)
        fps_options = sorted(fps_options)
    options_html = "".join(
        f'<option value="{x}" {"selected" if x == fps_default else ""}>{x}</option>'
        for x in fps_options
    )
    html = f"""
    <div style="font-family: sans-serif;">
      <style>
        .ctrl-btn {{
          width: 92px;
          height: 36px;
          border: 1px solid #d6d6d6;
          border-radius: 8px;
          background: #fafafa;
          cursor: pointer;
          font-size: 14px;
        }}
        .ctrl-btn:hover {{
          background: #f2f2f2;
        }}
      </style>
      <img id="viewer" style="width:{width_px}px; max-width:100%; border:1px solid #ddd; border-radius:8px;" />
      <div style="display:flex; gap:8px; align-items:center; margin-top:10px; flex-wrap:nowrap;">
        <button id="prevBtn" class="ctrl-btn">⏮ Prev</button>
        <button id="toggleBtn" class="ctrl-btn">⏸ Pause</button>
        <button id="nextBtn" class="ctrl-btn">⏭ Next</button>
        <label style="margin-left:4px;">FPS</label>
        <select id="fpsSel">
          {options_html}
        </select>
        <label style="margin-left:8px;">Frame</label>
        <input id="frameSlider" type="range" min="0" max="0" value="0" style="width:280px; margin:0;" />
        <span id="frameText" style="min-width:86px;"></span>
        <label style="margin-left:8px;">Sample</label>
        <input id="sampleSlider" type="range" min="0" max="0" value="0" style="width:280px; margin:0;" />
        <span id="sampleText" style="min-width:86px;"></span>
      </div>
    </div>

    <script>
      const frames = {data_json};
      const sampleFrameIndices = {sample_json};
      const sampleMarkers = {marker_json};
      const imgEl = document.getElementById('viewer');
      const slider = document.getElementById('frameSlider');
      const frameText = document.getElementById('frameText');
      const fpsSel = document.getElementById('fpsSel');
      const toggleBtn = document.getElementById('toggleBtn');
      const prevBtn = document.getElementById('prevBtn');
      const nextBtn = document.getElementById('nextBtn');
      const sampleSlider = document.getElementById('sampleSlider');
      const sampleText = document.getElementById('sampleText');

      let idx = 0;
      let timer = null;
      let playing = true;

      function show(i) {{
        if (!frames.length) return;
        idx = ((i % frames.length) + frames.length) % frames.length;
        imgEl.src = 'data:image/jpeg;base64,' + frames[idx];
        slider.value = idx;
        frameText.textContent = (idx + 1) + ' / ' + frames.length;
        if (sampleFrameIndices.length) {{
          let sIdx = 0;
          while (sIdx + 1 < sampleFrameIndices.length && sampleFrameIndices[sIdx + 1] <= idx) sIdx++;
          sampleSlider.value = sIdx;
          sampleText.textContent = (sIdx + 1) + ' / ' + sampleFrameIndices.length;
        }}
      }}

      function curFps() {{ return parseInt(fpsSel.value || '12', 10); }}

      function start() {{
        stop();
        playing = true;
        toggleBtn.textContent = '⏸ Pause';
        timer = setInterval(() => {{
          show(idx + 1);
        }}, Math.max(1, Math.round(1000 / curFps())));
      }}

      function stop() {{
        if (timer) {{ clearInterval(timer); timer = null; }}
        playing = false;
        toggleBtn.textContent = '▶ Play';
      }}

      slider.max = Math.max(0, frames.length - 1);
      sampleSlider.max = Math.max(0, sampleFrameIndices.length - 1);
      slider.addEventListener('input', () => show(parseInt(slider.value, 10)));
      sampleSlider.addEventListener('input', () => {{
        const s = parseInt(sampleSlider.value, 10);
        if (sampleFrameIndices.length) show(sampleFrameIndices[s]);
      }});
      fpsSel.addEventListener('change', () => {{ if (playing) start(); }});
      toggleBtn.addEventListener('click', () => {{ if (playing) stop(); else start(); }});
      prevBtn.addEventListener('click', () => show(idx - 1));
      nextBtn.addEventListener('click', () => show(idx + 1));

      show(0);
      start();
    </script>
    """
    # Keep iframe height tight so next widgets appear right below the video.
    components.html(html, height=620, scrolling=False)


def _background_preload_worker(version: str, dataroot: str, scenes: List[str]):
    for s in scenes:
        try:
            _precompute_scene_b64_frames_raw(version, dataroot, s, max_frames=0)
        except Exception:
            continue


def main():
    st.set_page_config(page_title="nuScenes Viewer", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1rem; padding-bottom: 0.8rem; }
          h1 { margin-top: 0.1rem; margin-bottom: 0.5rem; }
          details { margin-bottom: 0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("nuScenes Scene Viewer")

    with st.sidebar:
        dataroot = st.text_input("Dataset Path", value=DEFAULT_DATAROOT)
        version = st.selectbox("Version", ["v1.0-trainval", "v1.0-mini"], index=0)
        search_text = st.text_input("Search", value="")
        search_btn = st.button("Search")

    if not os.path.isdir(dataroot):
        st.error(f"Dataset path not found: {dataroot}")
        return

    nusc = load_nusc(version, dataroot)
    all_scenes = scene_name_list(version, dataroot)

    source_key = (version, dataroot)
    if st.session_state.get("scene_source") != source_key:
        st.session_state["scene_source"] = source_key
        st.session_state["scene_candidates"] = all_scenes

    if search_btn:
        key = search_text.strip().lower()
        matched = [s for s in all_scenes if key in s.lower()] if key else all_scenes
        st.session_state.scene_candidates = matched

    # Guard against stale/empty session state that can occur before manual refresh.
    if not st.session_state.get("scene_candidates"):
        st.session_state["scene_candidates"] = all_scenes

    candidates = st.session_state.scene_candidates
    if not candidates:
        st.warning("검색 결과가 없습니다.")
        return

    with st.sidebar:
        if "selected_scene" not in st.session_state or st.session_state["selected_scene"] not in candidates:
            st.session_state["selected_scene"] = candidates[0]
        scene_name = st.selectbox(
            "Scene List",
            candidates,
            index=candidates.index(st.session_state["selected_scene"]),
            key="selected_scene",
        )

    scene = get_scene_by_name(nusc, scene_name)
    if scene is None:
        st.error(f"Scene not found: {scene_name}")
        return

    st.markdown(
        """
        <style>
          .meta-wrap {
            border: 1px solid #dfe3ea;
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 12px;
            background: linear-gradient(180deg, #f8fafc 0%, #f2f6fb 100%);
          }
          .meta-grid {
            display: grid;
            grid-template-columns: 180px 180px 1fr;
            gap: 14px;
            align-items: start;
          }
          .meta-card {
            background: #ffffff;
            border: 1px solid #e6ecf4;
            border-radius: 10px;
            padding: 10px 12px;
          }
          .meta-title {
            color: #667085;
            font-size: 12px;
            margin-bottom: 6px;
            font-weight: 600;
            letter-spacing: 0.2px;
          }
          .meta-value {
            color: #101828;
            font-size: 17px;
            font-weight: 700;
          }
          .meta-desc {
            color: #344054;
            font-size: 14px;
            line-height: 1.45;
            white-space: pre-wrap;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="meta-wrap">
          <div class="meta-grid">
            <div class="meta-card">
              <div class="meta-title">Scene</div>
              <div class="meta-value">{scene_name}</div>
            </div>
            <div class="meta-card">
              <div class="meta-title">Samples</div>
              <div class="meta-value">{scene.get('nbr_samples', '-')}</div>
            </div>
            <div class="meta-card">
              <div class="meta-title">Description</div>
              <div class="meta-desc">{scene.get('description', '-')}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading scene frames into browser buffer..."):
        b64_frames = precompute_scene_b64_frames(version, dataroot, scene_name, max_frames=0)

    if not b64_frames:
        st.error("No frames to render.")
        return

    preload_key = (version, dataroot, scene_name)
    if st.session_state.get("preload_key") != preload_key:
        st.session_state["preload_key"] = preload_key
        if scene_name in all_scenes:
            idx = all_scenes.index(scene_name)
            lo = max(0, idx - PRELOAD_BEFORE)
            hi = min(len(all_scenes), idx + PRELOAD_AFTER + 1)
            around = [s for s in all_scenes[lo:hi] if s != scene_name]
            if around:
                t = threading.Thread(
                    target=_background_preload_worker,
                    args=(version, dataroot, around),
                    daemon=True,
                )
                t.start()

    sample_frame_indices = build_sample_frame_indices(version, dataroot, scene_name)
    sample_markers = build_sample_markers(version, dataroot, scene_name)
    render_player_html(
        b64_frames,
        sample_frame_indices=sample_frame_indices,
        sample_markers=sample_markers,
        fps_default=DEFAULT_FPS,
        width_px=TILE_W * 3,
    )

    with st.expander("Metadata", expanded=False):
        st.json(
            {
                "name": scene.get("name"),
                "token": scene.get("token"),
                "description": scene.get("description"),
                "nbr_samples": scene.get("nbr_samples"),
                "first_sample_token": scene.get("first_sample_token"),
                "last_sample_token": scene.get("last_sample_token"),
                "log_token": scene.get("log_token"),
            }
        )


if __name__ == "__main__":
    main()
