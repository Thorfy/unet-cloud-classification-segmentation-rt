"""Dashboard live: types de nuages + couches activables."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
from PIL import Image

from src.cloud_types import CLASSES, class_fractions, cloud_fraction, filter_mask, overlay_types
from src.infer_types import load_type_model, run_type_inference
from src.ingest.sources import SOURCES, fetch_latest
from src.instances import draw_instance_boxes, instances_table, mask_to_instances
from src.scene import classify_fraction

st.set_page_config(page_title="U-Net Cloud Classification and Segmentation RT", layout="wide")


def _display_rgb(arr: np.ndarray) -> np.ndarray:
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


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _layer_key(key: str) -> str:
    return f"layer_{key}"


def _init_layers() -> None:
    for idx, (key, _label, _rgb) in CLASSES.items():
        if idx != 0 and _layer_key(key) not in st.session_state:
            st.session_state[_layer_key(key)] = True


def _set_all_layers(value: bool) -> None:
    for idx, (key, _label, _rgb) in CLASSES.items():
        if idx != 0:
            st.session_state[_layer_key(key)] = value


def _rel_weights(path: str | Path) -> str:
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return raw.name


def main() -> None:
    _init_layers()
    st.title("Classification et segmentation de nuages")
    st.caption("Coche les types dans la barre laterale pour les afficher ou les masquer sur l'image.")

    source = st.sidebar.selectbox(
        "Source satellite",
        options=list(SOURCES.keys()),
        format_func=lambda k: SOURCES[k],
        index=0,
    )
    min_area = st.sidebar.slider("Surface min instance (px)", 32, 2000, 200, 32)
    if st.sidebar.button("Rafraichir l'image", use_container_width=True):
        cached_fetch.clear()
        st.session_state.pop(f"infer_{source}", None)
        st.rerun()

    st.sidebar.markdown("**Vues**")
    show_base = st.sidebar.checkbox("Image satellite", value=True)
    show_types = st.sidebar.checkbox("Masque des types", value=True)
    show_boxes = st.sidebar.checkbox("Boites d'instances", value=True)

    st.sidebar.markdown("**Types de nuages**")
    b1, b2 = st.sidebar.columns(2)
    b1.button("Tout", use_container_width=True, on_click=_set_all_layers, args=(True,))
    b2.button("Aucun", use_container_width=True, on_click=_set_all_layers, args=(False,))

    enabled: set[int] = set()
    for idx, (key, label, rgb) in CLASSES.items():
        if idx == 0:
            continue
        swatch, box = st.sidebar.columns([1, 8])
        swatch.markdown(
            f"<div style='width:16px;height:16px;margin-top:8px;border-radius:3px;background:{_hex(rgb)}'></div>",
            unsafe_allow_html=True,
        )
        if box.checkbox(label, key=_layer_key(key)):
            enabled.add(idx)

    cache_key = f"infer_{source}"
    if cache_key not in st.session_state:
        try:
            with st.status("Chargement image et inference…", expanded=True) as status:
                payload = cached_fetch(source)
                model, weights, device = get_type_model()
                result = run_type_inference(
                    Image.fromarray(payload["image"]),
                    model=model,
                    device=device,
                    min_instance_area=32,
                )
                st.session_state[cache_key] = {
                    "image": np.asarray(result["image"]),
                    "mask": np.asarray(result["mask"]),
                    "weights": weights,
                    "payload": payload,
                }
                status.update(label="Pret", state="complete")
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Echec du flux {source}: {exc}")
            return
    bundle = st.session_state[cache_key]

    st.sidebar.caption(f"Poids: `{_rel_weights(bundle['weights'])}`")
    payload = bundle["payload"]
    shown = _display_rgb(bundle["image"])
    visible = filter_mask(bundle["mask"], enabled)
    instances = mask_to_instances(visible, min_area=min_area)
    frac = cloud_fraction(visible)
    fr = class_fractions(visible)

    canvas = shown.copy()
    if show_types and enabled:
        canvas = overlay_types(canvas, visible, opacity=0.45)
    if show_boxes:
        canvas = draw_instance_boxes(canvas, instances)
    if not show_base and (show_types or show_boxes):
        # fond sombre pour ne garder que les couches
        dark = np.zeros_like(shown)
        if show_types and enabled:
            dark = overlay_types(dark, visible, opacity=0.85)
        if show_boxes:
            dark = draw_instance_boxes(dark, instances)
        canvas = dark

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classe scene", classify_fraction(frac))
    c2.metric("Couverture visible", f"{frac * 100:.1f} %")
    c3.metric("Instances", str(len(instances)))
    c4.metric("Acquisition", str(payload["captured_at"])[:25])

    ids = [i for i in CLASSES if i != 0]
    for start in range(0, len(ids), 5):
        chunk = ids[start : start + 5]
        cols = st.columns(len(chunk))
        for col, idx in zip(cols, chunk):
            key, label, _ = CLASSES[idx]
            on = idx in enabled
            col.metric(label.split(" (")[0], f"{fr.get(key, 0.0) * 100:.1f} %" if on else "off")

    st.image(canvas, caption=f"{payload['provider']} — couches actives", width="stretch")

    rows = instances_table(instances)
    if rows:
        st.subheader("Instances des couches actives")
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("Aucune instance pour les couches / seuil choisis.")

    if source.startswith("europe") or source.startswith("meteosat"):
        st.info(payload["domain_gap"])
    elif source != "goes-19-cmi":
        st.warning(payload["domain_gap"])
    else:
        st.info(payload["domain_gap"])
    st.code(payload["url"])


if __name__ == "__main__":
    main()
