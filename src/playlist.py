import base64
import streamlit as st
from config import BASE_DIR

def video_to_base64(path):
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()

def render_playlist(detected, df):
            playlist        = []
            playlist_labels = []
            for kw in detected:
                row = df[df["keyword"] == kw]
                if not row.empty:
                    raw = row.iloc[0]["video_path"].strip()
                    vp  = (BASE_DIR / raw).resolve()
                    if vp.exists():
                        playlist.append(str(vp))
                        playlist_labels.append(kw)

            if playlist:
                sources_js = "[" + ",".join([f'"{video_to_base64(p)}"' for p in playlist]) + "]"
                labels_js  = "[" + ",".join([f'"{playlist_labels[i]}"' for i in range(len(playlist))]) + "]"

                html = f"""
                <div style="text-align:center;">
                    <div style="margin-bottom:8px;font-weight:bold;font-size:1.1em">
                        Now Signing: <span id="label" style="color:#2e7d32"></span>
                    </div>
                    <video id="player" width="480" autoplay playsinline
                        style="border-radius:12px;border:2px solid #2e7d32">
                    </video>
                    <div style="margin-top:8px;color:gray;font-size:0.9em">
                        Word <span id="cur">1</span> of <span id="tot"></span>
                    </div>
                </div>
                <script>
                    const sources = {sources_js};
                    const labels  = {labels_js};
                    const video   = document.getElementById("player");
                    const label   = document.getElementById("label");
                    const cur     = document.getElementById("cur");
                    const tot     = document.getElementById("tot");

                    let index = 0;
                    tot.textContent = sources.length;

                    function playNext() {{
                        if (index >= sources.length) index = 0;
                        label.textContent = labels[index];
                        cur.textContent   = index + 1;
                        video.src = "data:video/mp4;base64," + sources[index];
                        video.load();
                        video.play();
                        index++;
                    }}

                    video.addEventListener("ended", playNext);
                    playNext();
                </script>
                """
                st.components.v1.html(html, height=420)
            else:
                st.warning("No valid videos found for playlist.")
