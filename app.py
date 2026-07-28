"""
Streamlit dashboard for the KMC ward-map digitizer.

Upload a ward sheet (.png) and its QGIS Georeferencer control-point file
(.points); the app detects manholes and reconstructs the drainage network,
shows an overlay preview, and lets you download the georeferenced
shapefiles (EPSG:4326) as a zip.

Robustness for "any kind of PNG":
  * Colours are auto-detected per sheet (no hard-coded palette).
  * Detection self-calibrates to symbol size, so it works at any
    resolution and for both hollow-ring and solid-dot manhole styles.
  * Drain lines are matched by hue, catching multi-shade drain drawings.
  * When a sheet defeats auto-calibration, the sidebar lets you override
    the manhole / drain colours and sensitivity by hand.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io

import cv2
import numpy as np
import streamlit as st

import ward_digitizer as wd

st.set_page_config(page_title="Ward Map Digitizer", layout="wide")

st.title("KMC Ward Map Digitizer")
st.caption("Upload a ward `.png` + its `.points` file → get georeferenced manhole & drainage shapefiles.")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def _decode_image(png_bytes):
    # NOT cached on purpose: caching decoded ~70-megapixel images kept every
    # uploaded sheet resident in RAM across runs, which pushed the hosted
    # free tier over its memory limit. Decoding is fast, so just redo it.
    arr = np.frombuffer(png_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
col_png, col_pts = st.columns(2)
with col_png:
    png_file = st.file_uploader("Ward raster (.png)", type=["png", "jpg", "jpeg", "tif", "tiff"])
with col_pts:
    pts_file = st.file_uploader("GCP file (.points)", type=["points", "txt", "csv"])

ward_name = st.text_input("Output name", value="ward").strip() or "ward"

st.sidebar.header("Detection settings")

style_label = st.sidebar.selectbox(
    "Manhole symbol style",
    ["Filled / coloured symbol", "Outlined circle (marks inside)"],
    help="Most sheets draw manholes as solid coloured circles/dots → use the first. "
         "If manholes are black-ring circles with +/triangle/hatch marks inside "
         "(e.g. Ward 52), pick the second.")
manhole_style = "outlined" if style_label.startswith("Outlined") else "filled"

auto_mode = st.sidebar.toggle("Auto-detect colours", value=True,
                              help="Turn off to pick the manhole / drain colours by hand "
                                   "when a sheet confuses auto-detection.")

manhole_override = drain_override = None
if not auto_mode:
    default_mh = "#00ff00" if manhole_style == "outlined" else "#3f00ff"
    mh_label = "Circle colour (outline or fill)" if manhole_style == "outlined" else "Manhole colour"
    mh_help = ("The colour that defines the circle: the ring/outline colour "
               "(e.g. Ward 8 cyan) or the interior fill colour (e.g. Ward 52 green). "
               "Set this if auto-detection picks the wrong one.") if manhole_style == "outlined" else None
    mh_hex = st.sidebar.color_picker(mh_label, value=default_mh, help=mh_help)
    dr_hex = st.sidebar.color_picker("Drain-line colour", value="#ff0000")
    manhole_override = _hex_to_rgb(mh_hex)
    drain_override = _hex_to_rgb(dr_hex)

drain_sensitivity = st.sidebar.slider(
    "Drain sensitivity", 0.30, 0.90, 0.60, 0.05,
    help="Lower = connect more manhole pairs into drainage lines (more permissive).")
connect_k = st.sidebar.slider("Neighbours tested per manhole", 3, 10, 6, 1)
include_secondary = st.sidebar.checkbox(
    "Detect labelled (2nd colour) manholes", value=True,
    help="Also find a second manhole population, e.g. large labelled 'MH-n' circles.")

run = st.button("Run digitization", type="primary", disabled=not (png_file and pts_file))


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
if run and png_file and pts_file:
    img = _decode_image(png_file.getvalue())
    if img is None:
        st.error("Could not read the image file.")
        st.stop()

    try:
        pts_text = pts_file.getvalue().decode("utf-8", errors="replace")
        gcps = wd.parse_points_text(pts_text)
        crs_wkt = wd.parse_points_crs(pts_text)  # write the output .prj in the GCPs' own CRS
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not parse the .points file: {e}")
        st.stop()
    if len(gcps) < 3:
        st.error(f"Only {len(gcps)} usable GCPs found — need at least 3 to fit a transform.")
        st.stop()

    logs = []
    try:
        with st.spinner("Detecting manholes and drainage…"):
            result = wd.digitize_arrays(
                img, gcps,
                manhole_style=manhole_style,
                manhole_colour=manhole_override,
                drain_colour=drain_override,
                include_secondary=include_secondary,
                drain_sensitivity=drain_sensitivity,
                connect_k=connect_k,
                crs_wkt=crs_wkt,
                log=logs.append,
            )
    except Exception as e:  # noqa: BLE001
        st.error(f"Digitization failed: {e}")
        st.stop()

    st.session_state["result"] = result
    st.session_state["img"] = img
    st.session_state["logs"] = logs
    st.session_state["ward_name"] = ward_name


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
if "result" in st.session_state:
    result = st.session_state["result"]
    img = st.session_state["img"]
    ward_name = st.session_state.get("ward_name", "ward")

    w, h = result["image_size"]
    n_mh = len(result["manholes"])
    n_ed = len(result["edges"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Manholes", n_mh)
    m2.metric("Drainage lines", n_ed)
    m3.metric("Image", f"{w}×{h}")
    m4.metric("GCP fit RMSE", f"{result['rmse_px']:.0f} px")

    c1, c2 = st.columns([3, 1])
    with c1:
        with st.spinner("Rendering overlay…"):
            overlay = wd.render_overlay(img, result)
        st.image(overlay, caption="Green = manholes · magenta ring = labelled manholes · blue = drainage",
                 use_container_width=True)
    with c2:
        st.subheader("Detected colours")
        st.color_picker("Manhole", value=_rgb_to_hex(result["manhole_colour"]),
                        disabled=True, key="show_mh")
        if result["drain_colour"]:
            st.color_picker("Drain line", value=_rgb_to_hex(result["drain_colour"]),
                            disabled=True, key="show_dr")
        if result["secondary"]:
            st.caption("Secondary manhole colours:")
            for s in result["secondary"]:
                st.write(f"• {tuple(s['colour'])} — {s['n']} circles ({s['on_drain_frac']*100:.0f}% on drain)")
        if result["drain_colour"] is None:
            st.warning("No drain colour found — drainage layer will be empty. "
                       "Try manual override in the sidebar.")

    zip_bytes = wd.build_shapefile_zip(result, ward_name=ward_name)
    st.download_button(
        "Download shapefiles (.zip)", data=zip_bytes,
        file_name=f"{ward_name}_digitized_shapefiles.zip", mime="application/zip",
        type="primary")
    st.caption("Zip contains Potholes_*.shp (manholes) and Drainage_*.shp (lines) in EPSG:4326, "
               "plus a pixel-coordinate JSON for QA.")

    with st.expander("Run log"):
        st.code("\n".join(st.session_state.get("logs", [])) or "—")

    st.info("Tip: if manholes are missed, over-counted (many dots inside one circle), or the "
            "drainage looks sparse — try switching **Manhole symbol style** in the sidebar, turn "
            "off **Auto-detect colours** to set them by eye, or lower **Drain sensitivity**.")
else:
    st.info("Upload both files and click **Run digitization** to begin.")
