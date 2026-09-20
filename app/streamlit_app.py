"""Dashboard live: types de nuages + segmentation sémantique."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
from PIL import Image

from src.cloud_types import CLASSES, overlay_types
from src.infer_types import legend_rows, load_type_model, run_type_inference
from src.ingest.sources import SOURCES, fetch_latest
from src.instances import annotate_instances, instances_table

st.set_page_config(page_title="Types de nuages live", layout="wide")


def _display_rgb(arr: np.ndarray) -> np.ndarray:
    """Contraste pour l'affichage (l'inference garde l'image brute)."""
    x = np.asarray(arr, dtype=np.float32)
    earth = x.max(axis=-1) > 8
    if not np.any(earth):
        return np.asarray(arr)
    lo, hi = np.percentile(x[earth], (2, 97))
    if hi <= lo + 1:
        return np.asarray(arr)
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    y = (np.power(y, 0.9) * 255.0).astype(np.uint8)
    y[~earth] = 0
    return y


@st.cache_resource
def get_type_model():
    return load_type_model()


@st.cache_data(ttl=180, show_spinner=False)
def cached_fetch(source: str) -> dict:
    payload = fetch_latest(source)
    return {
        "image": np.asarray(payload["image"]),
        "source": payload["source"],
        "provider": payload["provider"],
        "captured_at": str(payload["captured_at"]),
        "url": payload["url"],
        "domain_gap": payload["domain_gap"],
    }


def main() -> None:
    st.title("Types de nuages et segmentation")
    st.caption(
        "11 types + instances (composantes connexes par type). "
        "Source par defaut: Meteosat Europe (MET Norway)."
    )

    source = st.sidebar.selectbox(
        "Source satellite",
        options=list(SOURCES.keys()),
        format_func=lambda k: SOURCES[k],
        index=0,
    )
    min_area = st.sidebar.slider("Surface min instance (px)", 32, 2000, 200, 32)
    if st.sidebar.button("Rafraichir l'image", use_container_width=True):
        cached_fetch.clear()
        st.rerun()
    st.sidebar.markdown("**Legende**")
    for hexcol, label in legend_rows():
        st.sidebar.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
            f"<span style='width:16px;height:16px;background:{hexcol};display:inline-block;border-radius:3px'></span>"
            f"<span>{label}</span></div>",
            unsafe_allow_html=True,
        )

    status = st.status("Chargement…", expanded=True)
    try:
        with status:
            st.write("Modele")
            try:
                model, weights, device = get_type_model()
            except FileNotFoundError as exc:
                status.update(label="Checkpoint manquant", state="error")
                st.error(str(exc))
                return
            st.sidebar.write(f"Poids: `{weights}`")

            st.write("Image satellite")
            payload = cached_fetch(source)
            image = Image.fromarray(payload["image"])

            st.write("Segmentation des types")
            result = run_type_inference(
                image, model=model, device=device, min_instance_area=min_area
            )
        status.update(label="Pret", state="complete")
    except Exception as exc:
        status.update(label="Echec", state="error")
        st.error(f"Echec du flux {source}: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classe scene", result["scene_class"])
    c2.metric("Couverture nuages", f"{result['cloud_fraction'] * 100:.1f} %")
    c3.metric("Instances", str(result["n_instances"]))
    c4.metric("Acquisition", str(payload["captured_at"])[:25])

    fr = result["fractions"]
    ids = [i for i in CLASSES if i != 0]
    for start in range(0, len(ids), 5):
        chunk = ids[start : start + 5]
        cols = st.columns(len(chunk))
        for col, idx in zip(cols, chunk):
            key, label, _ = CLASSES[idx]
            col.metric(label.split(" (")[0], f"{fr.get(key, 0.0) * 100:.1f} %")

    shown = _display_rgb(np.asarray(result["image"]))
    semantic = overlay_types(shown, result["mask"])
    inst = annotate_instances(shown, result["mask"], result["instances"])

    left, mid, right = st.columns(3)
    left.image(shown, caption=f"Brut — {payload['provider']}", width="stretch")
    mid.image(semantic, caption="Segmentation semantique", width="stretch")
    right.image(inst, caption="Instances (boites + ids)", width="stretch")

    rows = instances_table(result["instances"])
    if rows:
        st.subheader("Instances detectees")
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("Aucune instance au-dessus du seuil de surface.")

    if source.startswith("europe") or source.startswith("meteosat"):
        st.info(payload["domain_gap"])
    elif source != "goes-19-cmi":
        st.warning(payload["domain_gap"])
    else:
        st.info(payload["domain_gap"])
    st.code(payload["url"])


if __name__ == "__main__":
    main()
