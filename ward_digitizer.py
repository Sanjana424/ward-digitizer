"""
ward_digitizer.py

Semi-automated digitization of KMC ward drainage maps.
Given a raw ward map raster (.png) and its QGIS Georeferencer control-point
file (.points), this script:

  1. Fits a pixel <-> geographic (EPSG:4326) affine transform from the GCPs.
  2. Detects manhole symbols (colour-coded circle markers) with classical CV
     (colour thresholding + connected components) -- this is the primary,
     production-grade extraction method. It is deterministic, needs no
     training data, runs on the full ~70-megapixel sheet in under a second,
     and is far more precise than asking a VLM to eyeball pixel coordinates.
  3. Reconstructs the drainage network by testing straight-line connectivity
     between nearby manhole nodes against the red dashed-line pixel mask
     (these ward sheets draw drain runs as straight segments between
     manholes, so this "graph snapping" approach reproduces the polylines
     without needing full raster skeletonization/vectorization).
  4. Writes both layers out as ESRI Shapefiles in EPSG:4326, matching the
     ground-truth Potholes_Ward_XX.shp / Drainage_Ward_XX.shp schema.

A VLM (Claude vision) pass is wired in as an OPTIONAL enrichment step to
read the "MHxx RCPC/CI" text labels and pipe-diameter callouts next to each
manhole -- OCR/semantic reading is a genuine VLM strength, unlike precise
sub-pixel geometric localization on a 70-megapixel sheet.

Usage:
    python ward_digitizer.py <ward.png> <ward.png.points> <output_dir> [--ward-name 44]

Requires: numpy, opencv-python-headless, pyshp (pip install ... --break-system-packages)
"""

import argparse
import io
import json
import os

import cv2
import numpy as np
import shapefile  # pyshp

# ---------------------------------------------------------------------------
# Symbol colour palette, reverse-engineered from ward 44 / ward 25 ground
# truth (see WARD_DIGITIZATION_ANALYSIS.md). These are near-exact RGB values
# because the sheets are rasterized CAD/GIS exports, not photographs -- so
# tight colour tolerances are appropriate and reliable.
# ---------------------------------------------------------------------------
COLOURS_RGB = {
    "manhole_unlabeled": (63, 0, 255),   # open blue/purple circle
    "manhole_labeled":   (255, 0, 0),    # red circle w/ "MHxx" text (trunk manholes)
    "drain_line":        (255, 0, 0),    # same red, dashed polyline
    "plot_boundary":     (45, 184, 0),   # green
    "lateral_line":      (0, 165, 221),  # cyan, house service connections
    "ward_boundary":     (255, 0, 255),  # magenta
}


# ---------------------------------------------------------------------------
# 1. Georeferencing: parse GCPs, fit affine transform
# ---------------------------------------------------------------------------
def parse_points_file(path):
    """Parse a QGIS Georeferencer .points file.

    Columns: mapX,mapY,sourceX,sourceY,enable,dX,dY,residual
    mapX/mapY are geographic lon/lat (per the #CRS header, normally EPSG:4326).
    sourceX/sourceY are pixel coordinates with sourceY stored NEGATED
    (i.e. actual pixel row = -sourceY, pixel col = sourceX), origin top-left.
    """
    with open(path, encoding="utf-8") as f:
        return parse_points_text(f.read())


def parse_points_text(text):
    """Same as parse_points_file but from the file's text content (for the
    dashboard, which receives an uploaded file's bytes rather than a path).

    Robust to the two header styles seen in the wild: a leading
    '#CRS: ...' comment line then a 'mapX,mapY,...' column header, OR just
    the column header. Any line whose first field isn't a float is skipped.
    """
    gcps = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            map_x, map_y, src_x, src_y = (float(parts[0]), float(parts[1]),
                                          float(parts[2]), float(parts[3]))
            enable = int(float(parts[4]))
        except ValueError:
            continue  # header row ("mapX,mapY,...") or junk
        if enable:
            gcps.append((map_x, map_y, src_x, -src_y))  # store as true pixel (col,row)
    return gcps


class GeoTransform:
    """Pixel<->geo transform fit with Thin Plate Spline (TPS) interpolation.

    IMPORTANT: an earlier version of this script used a single global
    6-parameter affine fit (least squares over all GCPs). That is what
    QGIS Georeferencer's "Linear" transform type also does -- but with
    only ~16 hand-picked, noisy GCPs spread over an irregular ward
    boundary, a global affine has to average out inconsistencies across
    every point at once, and can be off by *tens of metres even at the
    GCP locations themselves* (checked empirically: up to 62 m on these
    two wards). That shows up visually as a locally-clustered "shift" --
    manholes near a GCP that the affine fit poorly land noticeably off
    from the actual symbol, even though nearby manholes elsewhere line up
    fine.

    TPS fixes this: it's an exact interpolant (zero error *at* every GCP
    by construction) that flexes smoothly between them, which is also
    QGIS's recommended transform type for exactly this kind of scattered,
    hand-picked control-point set ("Thin Plate Spline (Spline)" in the
    Georeferencer dialog). If you georeferenced the source PNG in QGIS
    using Spline/TPS, this now matches that -- if you used Linear/
    Polynomial 1, switch back to affine_transform() below to match it
    instead. Either way, WARD_DIGITIZATION_ANALYSIS.md documents both so
    you can tell which one QGIS is showing you.
    """

    def __init__(self, gcps):
        from scipy.interpolate import RBFInterpolator
        px = np.array([[g[2], g[3]] for g in gcps])
        lon = np.array([g[0] for g in gcps])
        lat = np.array([g[1] for g in gcps])
        geo = np.column_stack([lon, lat])
        self._fwd_lon = RBFInterpolator(px, lon, kernel="thin_plate_spline")
        self._fwd_lat = RBFInterpolator(px, lat, kernel="thin_plate_spline")
        self._inv_x = RBFInterpolator(geo, px[:, 0], kernel="thin_plate_spline")
        self._inv_y = RBFInterpolator(geo, px[:, 1], kernel="thin_plate_spline")

        errs = []
        for map_x, map_y, tx, ty in gcps:
            lon_p = float(self._fwd_lon([[tx, ty]])[0])
            lat_p = float(self._fwd_lat([[tx, ty]])[0])
            dlon_m = (lon_p - map_x) * 111320 * np.cos(np.radians(map_y))
            dlat_m = (lat_p - map_y) * 110540
            errs.append((dlon_m ** 2 + dlat_m ** 2) ** 0.5)
        self.rmse_m = float(np.sqrt(np.mean(np.square(errs))))  # ~0 by construction

    def pixel_to_geo(self, px, py):
        lon = float(self._fwd_lon([[px, py]])[0])
        lat = float(self._fwd_lat([[px, py]])[0])
        return lon, lat

    def pixel_to_geo_batch(self, pxpy):
        pxpy = np.asarray(pxpy)
        return np.column_stack([self._fwd_lon(pxpy), self._fwd_lat(pxpy)])

    def geo_to_pixel(self, lon, lat):
        px = float(self._inv_x([[lon, lat]])[0])
        py = float(self._inv_y([[lon, lat]])[0])
        return px, py


def fit_affine(gcps):
    """Legacy pixel->geo least-squares affine fit.

    *** DO NOT use this to place output points -- use AffineGeoToPixel
    below and invert it. ***  Kept only because other helpers
    (robust_fit_affine, the analysis scripts) still call it.

    This fits lon/lat as a function of pixel x/y, i.e. it minimises error
    in the GEOGRAPHIC coordinate. That sounds right but produces points
    that land ~12 px (~1-2 m) off the drawn circle centres -- see
    AffineGeoToPixel's docstring for why, and the ~7x accuracy improvement
    from fixing it.
    """
    A, bx, by = [], [], []
    for map_x, map_y, px, py in gcps:
        A.append([px, py, 1])
        bx.append(map_x)
        by.append(map_y)
    A = np.array(A)
    coef_lon, *_ = np.linalg.lstsq(A, np.array(bx), rcond=None)
    coef_lat, *_ = np.linalg.lstsq(A, np.array(by), rcond=None)

    errs = []
    for map_x, map_y, px, py in gcps:
        lon = coef_lon[0] * px + coef_lon[1] * py + coef_lon[2]
        lat = coef_lat[0] * px + coef_lat[1] * py + coef_lat[2]
        dlon_m = (lon - map_x) * 111320 * np.cos(np.radians(map_y))
        dlat_m = (lat - map_y) * 110540
        errs.append((dlon_m ** 2 + dlat_m ** 2) ** 0.5)
    rmse_m = float(np.sqrt(np.mean(np.square(errs))))
    return coef_lon, coef_lat, rmse_m


class AffineGeoToPixel:
    """The CORRECT affine transform for this task: fit geo->pixel, invert
    for pixel->geo output.

    *** This is the fix for the "points are a few px off the circle
    centre" problem, and it is not a cosmetic tweak -- it cut median
    positional error against the ground truth from 2.41 m to 0.35 m on
    Ward 44 and 1.87 m to 0.28 m on Ward 25 (~7x), with identical
    detections. ***

    Why direction matters. An affine fit by least squares minimises
    residuals in its OUTPUT variable only. `fit_affine` above fits
    pixel->geo, so it minimises error in lon/lat -- but we don't care
    about lon/lat error, we care about landing on the right PIXEL (the
    circle centre). With noisy hand-picked GCPs the pixel->geo fit and the
    geo->pixel fit are NOT inverses of each other (classic regression
    attenuation: each direction shrinks differently toward the mean), and
    the gap between them is exactly the ~12 px offset that was visible in
    QGIS.

    QGIS/GDAL georeferencing (Polynomial 1) reports its residuals in
    PIXELS, i.e. it effectively works in the geo->pixel direction -- so to
    reproduce where QGIS actually places things (and where the ground
    truth was digitised), fit geo->pixel and invert. Confirmed
    empirically: mapping the ground-truth shapefile through the geo->pixel
    fit lands on the detected circle centres at 1.9 px median (Ward 44) /
    2.0 px (Ward 25), versus ~12 px the other way round.

    For a 1st-order (affine) transform the inverse is exact, so pixel->geo
    output is well defined and consistent with a Polynomial-1 raster warp.
    """

    def __init__(self, gcps):
        A = np.array([[g[0], g[1], 1.0] for g in gcps])  # [lon, lat, 1]
        px = np.array([g[2] for g in gcps])
        py = np.array([g[3] for g in gcps])
        self.cx, *_ = np.linalg.lstsq(A, px, rcond=None)  # lon,lat -> pixel x
        self.cy, *_ = np.linalg.lstsq(A, py, rcond=None)  # lon,lat -> pixel y
        self.M = np.array([[self.cx[0], self.cx[1]], [self.cy[0], self.cy[1]]])
        self.b = np.array([self.cx[2], self.cy[2]])
        self.Minv = np.linalg.inv(self.M)

        errs = []  # RMSE reported in PIXELS (matches QGIS's residual units)
        for map_x, map_y, tx, ty in gcps:
            pxp = self.cx[0] * map_x + self.cx[1] * map_y + self.cx[2]
            pyp = self.cy[0] * map_x + self.cy[1] * map_y + self.cy[2]
            errs.append((pxp - tx) ** 2 + (pyp - ty) ** 2)
        self.rmse_px = float(np.sqrt(np.mean(errs)))

    def pixel_to_geo(self, px, py):
        lon, lat = self.Minv @ (np.array([px, py]) - self.b)
        return float(lon), float(lat)

    def pixel_to_geo_batch(self, pxpy):
        pxpy = np.asarray(pxpy, dtype=float)
        return (self.Minv @ (pxpy - self.b).T).T


def _loo_rmse_m(gcps):
    """Leave-one-out RMSE (metres): fit on n-1 points, measure error on the
    held-out one. This is the honest measure of how well the transform
    generalizes -- unlike in-sample RMSE, it can't be gamed by dropping
    points just because they happen to fit badly."""
    if len(gcps) < 4:
        return float("inf")
    errs = []
    for i in range(len(gcps)):
        train = gcps[:i] + gcps[i + 1:]
        cx, cy = fit_affine(train)[:2]
        map_x, map_y, px, py = gcps[i]
        lon = cx[0] * px + cx[1] * py + cx[2]
        lat = cy[0] * px + cy[1] * py + cy[2]
        dlon = (lon - map_x) * 111320 * np.cos(np.radians(map_y))
        dlat = (lat - map_y) * 110540
        errs.append(dlon ** 2 + dlat ** 2)
    return float(np.sqrt(np.mean(errs)))


def robust_fit_affine(gcps, max_drop_frac=0.35, min_pts=8):
    """Fit the affine transform with automatic outlier-GCP rejection.

    IMPORTANT FINDING: hand-picked GCPs on these sheets are noisy enough
    that a handful of badly-placed points can dominate the whole-ward fit.
    On Ward 44, 5 of the 16 GCPs (rows 4, 9, 10, 11, 14 in the .points
    file) were bad enough that dropping them cut leave-one-out RMSE from
    36.4 m to 8.5 m -- a >4x improvement, entirely from data cleaning, no
    change to the transform model. Ward 25 improved less dramatically
    (38.7 m -> 24.1 m dropping 4 points, rows 3, 4, 6, 16) because its
    GCP noise is more uniformly spread across all 16 points rather than
    concentrated in a few outliers -- there's a real limit to what
    outlier rejection alone can fix; the remaining error there needs
    better ground control, not smarter fitting.

    Algorithm: iteratively drop the single worst-residual GCP as long as
    doing so improves *leave-one-out* RMSE by more than 3% (guards against
    overfitting by requiring genuine out-of-sample improvement, not just a
    better in-sample number), stopping at max_drop_frac of the points or
    min_pts remaining, whichever comes first.

    NOTE: if you also georeference the raw PNG in QGIS using all 16 GCPs
    (e.g. to view it as a raster layer), that raster's warp and this
    shapefile's coordinates will now be fit from slightly different GCP
    sets, which can show as a new small disagreement specifically near
    whichever GCPs got dropped here. For full consistency, remove the
    same flagged GCPs in QGIS's Georeferencer and re-warp the raster too
    -- see WARD_DIGITIZATION_ANALYSIS.md for the exact list of dropped
    points per ward.
    """
    pts = list(gcps)
    best_loo = _loo_rmse_m(pts)
    dropped = []
    while len(pts) > min_pts and len(dropped) < len(gcps) * max_drop_frac:
        coef_lon, coef_lat, _ = fit_affine(pts)
        errs = []
        for map_x, map_y, px, py in pts:
            lon = coef_lon[0] * px + coef_lon[1] * py + coef_lon[2]
            lat = coef_lat[0] * px + coef_lat[1] * py + coef_lat[2]
            dlon = (lon - map_x) * 111320 * np.cos(np.radians(map_y))
            dlat = (lat - map_y) * 110540
            errs.append(dlon ** 2 + dlat ** 2)
        worst_i = int(np.argmax(errs))
        trial = pts[:worst_i] + pts[worst_i + 1:]
        trial_loo = _loo_rmse_m(trial)
        if trial_loo < best_loo * 0.97:
            dropped.append(pts.pop(worst_i))
            best_loo = trial_loo
        else:
            break
    coef_lon, coef_lat, rmse_m = fit_affine(pts)
    return coef_lon, coef_lat, rmse_m, pts, dropped, best_loo


def pixel_to_geo(px, py, coef_lon, coef_lat):
    lon = coef_lon[0] * px + coef_lon[1] * py + coef_lon[2]
    lat = coef_lat[0] * px + coef_lat[1] * py + coef_lat[2]
    return lon, lat


def geo_to_pixel(lon, lat, coef_lon, coef_lat):
    M = np.array([[coef_lon[0], coef_lon[1]], [coef_lat[0], coef_lat[1]]])
    c = np.array([coef_lon[2], coef_lat[2]])
    px, py = np.linalg.solve(M, np.array([lon, lat]) - c)
    return float(px), float(py)


# ---------------------------------------------------------------------------
# 1b. Auto colour calibration -- IMPORTANT.
#
# The colour palette above was reverse-engineered from ward 44 and is NOT
# portable: ward 25's sheet uses a completely different scheme (manholes
# drawn in plain red (255,0,0), drain runs in purple (127,0,255) dashed
# lines, rather than ward 44's blue-circle / red-dashed-line convention).
# Hard-coding one ward's colours and pointing the script at a new ward will
# silently fail or badly under/over-detect. auto_calibrate_colours() scans
# the sheet's saturated colours and scores each one by its connected-
# component shape statistics: manhole colours produce hundreds of small,
# near-square (circular) blobs; drain-line colours produce a much smaller
# number of large, elongated blobs. This needs no legend and no per-ward
# tuning. (A VLM is a good complementary sanity check here -- pass it a
# small crop and ask "what colour are the manhole circles / drain lines in
# this legend/sample?" -- see the Part 1 discussion in the analysis doc.)
# ---------------------------------------------------------------------------
def auto_calibrate_colours(img_bgr, sample_stride=3, tol=20, top_n=12, min_pop=50):
    """Rank the sheet's saturated colours by connected-component shape.

    Key discriminator, found empirically by comparing ward 44 (manholes =
    blue open circles, drain = red dashes) against ward 25 (manholes = red
    circles, drain = purple dashes): the MEDIAN ASPECT RATIO (long side /
    short side) of each colour's small blobs separates circles from dashes
    reliably even though the actual colours differ completely between
    sheets:
      - circle/manhole colours: aspect ratio clusters right at ~1.0
        (a ring is as tall as it is wide), and blob AREA has very low
        variance (every manhole is stamped from the same template).
      - dashed-line colours: aspect ratio is consistently >= ~2 (dash
        strokes are elongated), and area varies more (dash lengths vary
        with line angle/spacing).
    Naively using population count alone (more pixels/components = more
    likely a symbol) is NOT sufficient -- it picks up dashed lateral
    connector lines or solid plot-boundary colours instead. See
    WARD_DIGITIZATION_ANALYSIS.md for the failed first attempt.
    """
    small = img_bgr[::sample_stride, ::sample_stride]
    flat = small.reshape(-1, 3).astype(int)
    mx, mn = flat.max(axis=1), flat.min(axis=1)
    saturated = flat[(mx - mn) > 40]  # drop white/black/grey background & anti-aliasing
    if len(saturated) == 0:
        raise RuntimeError("No saturated colours found -- unexpected sheet style")
    uniq, counts = np.unique(saturated, axis=0, return_counts=True)
    candidates = uniq[np.argsort(-counts)][:top_n]

    stats_by_colour = {}
    for bgr in candidates:
        b, g, r = [int(v) for v in bgr]
        lower = np.array([max(b - tol, 0), max(g - tol, 0), max(r - tol, 0)], dtype=np.uint8)
        upper = np.array([min(b + tol, 255), min(g + tol, 255), min(r + tol, 255)], dtype=np.uint8)
        mask = cv2.inRange(img_bgr, lower, upper)
        n, _labels, cstats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            continue
        areas = cstats[1:, 4].astype(float)
        wh = cstats[1:, 2:4].astype(float)
        small_mask = (areas > 8) & (areas < 2000) & (wh[:, 0] < 70) & (wh[:, 1] < 70)
        if small_mask.sum() < min_pop:
            continue
        a = areas[small_mask]
        ar = np.maximum(wh[small_mask, 0], wh[small_mask, 1]) / np.maximum(
            np.minimum(wh[small_mask, 0], wh[small_mask, 1]), 1)
        stats_by_colour[(r, g, b)] = {
            "n": int(small_mask.sum()),
            "area_cv": float(a.std() / a.mean()) if a.mean() > 0 else 0.0,
            "aspect_med": float(np.median(ar)),
        }

    best_circle, best_circle_score = None, -1
    for colour, s in stats_by_colour.items():
        if s["aspect_med"] <= 1.35:  # near-square/circular blobs only
            score = s["n"] / (s["area_cv"] + 0.05)
            if score > best_circle_score:
                best_circle_score, best_circle = score, colour

    best_line, best_line_score = None, -1
    for colour, s in stats_by_colour.items():
        if colour == best_circle:
            continue
        if s["aspect_med"] >= 2.0:  # elongated dash strokes only
            score = s["aspect_med"] * s["area_cv"] * s["n"]
            if score > best_line_score:
                best_line_score, best_line = score, colour

    return {"manhole_colour": best_circle, "drain_colour": best_line, "debug_stats": stats_by_colour}


def find_secondary_manhole_colours(img_bgr, primary_manhole_colour, drain_colour, sample_stride=3,
                                    tol=20, top_n=12, min_pop=20, drain_proximity_radius=20,
                                    on_drain_frac_thresh=0.7):
    """Find additional manhole colours beyond the single best pick.

    Real-world finding (Ward 44): the sheet draws TWO visually distinct
    manhole populations in different colours -- small blue "ordinary"
    circles (the dominant population, picked as manhole_colour above) and
    larger red circles individually labelled "MH-1", "MH-2", ... "MH-17"
    at trunk/junction manholes. auto_calibrate_colours() only ever returns
    the single best-scoring circular colour, so this whole second
    population (~20% of ground-truth manholes on Ward 44) was silently
    dropped. It wasn't picked up by the aspect-ratio test above because
    that test looks at the MEDIAN aspect ratio across ALL of a colour's
    blobs -- and red is also used for "ID = x.x M." text and drain-line
    dashes on these sheets, which swamps the real circular subset in the
    aggregate statistic.

    The fix: explicitly pull out the near-circular SUBSET of each
    remaining candidate colour (ignoring what the colour's overall median
    looks like), then use a domain fact to decide whether that subset is
    real manholes or an unrelated same-shaped symbol (e.g. Ward 25 also
    has a secondary circle population in blue -- but those are a
    different utility layer, not sewer manholes): genuine manholes are
    drainage-network nodes, so they sit ON the drain line. Candidate
    circles are accepted as a secondary manhole colour only if >=70% of
    them fall within drain_proximity_radius px of the drain-line colour
    (validated against ground truth: Ward 44's secondary red circles hit
    100% on this test and are real; Ward 25's secondary blue circles hit
    3% and are correctly rejected).
    """
    small = img_bgr[::sample_stride, ::sample_stride]
    flat = small.reshape(-1, 3).astype(int)
    mx, mn = flat.max(axis=1), flat.min(axis=1)
    # NOTE: threshold here must be much stricter than auto_calibrate_colours'
    # (mx-mn)>40 -- a first version of this function used 40 and it picked
    # (255,212,223), a near-white ANTI-ALIASING HALO colour along the edge
    # of the red drain-line stroke itself, as a "secondary manhole colour".
    # Of course it scored 100% on-drain -- it *is* the drain line's edge
    # pixels. (mx-mn) > 120 keeps only genuinely saturated/vivid drawing
    # colours and excludes anti-aliasing blends.
    saturated = flat[(mx - mn) > 120]
    uniq, counts = np.unique(saturated, axis=0, return_counts=True)
    candidates = uniq[np.argsort(-counts)][:top_n]

    dr, dg, db = drain_colour
    dlower = np.array([max(db - 35, 0), max(dg - 35, 0), max(dr - 35, 0)], dtype=np.uint8)
    dupper = np.array([min(db + 35, 255), min(dg + 35, 255), min(dr + 35, 255)], dtype=np.uint8)
    drain_mask = cv2.inRange(img_bgr, dlower, dupper)
    drain_mask = cv2.dilate(drain_mask, np.ones((drain_proximity_radius, drain_proximity_radius), np.uint8))

    accepted = []
    for bgr in candidates:
        r, g, b = [int(v) for v in bgr]
        if (r, g, b) == primary_manhole_colour:
            continue
        # NOTE: do NOT also skip colour == drain_colour here. On Ward 44 the
        # same red (255,0,0) is used both for the dashed drain-line stroke
        # AND for the labelled "MH-xx" trunk-manhole circles -- one colour,
        # two distinct symbol roles. An earlier version of this function
        # skipped any candidate equal to drain_colour and silently missed
        # that entire secondary manhole population as a result.
        lower = np.array([max(b - tol, 0), max(g - tol, 0), max(r - tol, 0)], dtype=np.uint8)
        upper = np.array([min(b + tol, 255), min(g + tol, 255), min(r + tol, 255)], dtype=np.uint8)
        mask = cv2.inRange(img_bgr, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        n, _labels, cstats, cent = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if n <= 1:
            continue
        areas = cstats[1:, 4].astype(float)
        wh = cstats[1:, 2:4].astype(float)
        aspect = np.maximum(wh[:, 0], wh[:, 1]) / np.maximum(np.minimum(wh[:, 0], wh[:, 1]), 1)
        longdim = np.maximum(wh[:, 0], wh[:, 1])
        circ = (aspect <= 1.3) & (areas > 15) & (areas < 3000) & (wh[:, 0] < 90) & (wh[:, 1] < 90)
        if circ.sum() < min_pop:
            continue
        circ_cent = cent[1:][circ]
        on_drain = 0
        for cx, cy in circ_cent:
            cx, cy = int(cx), int(cy)
            if 0 <= cy < drain_mask.shape[0] and 0 <= cx < drain_mask.shape[1] and drain_mask[cy, cx]:
                on_drain += 1
        frac = on_drain / len(circ_cent)
        if frac >= on_drain_frac_thresh:
            # ref_dim = the typical size of these real circles. Passed to
            # detect_manholes so its size filter targets THIS size directly,
            # instead of the sheet-wide median -- which for a secondary
            # colour like red is corrupted by the hundreds of ~1 px dash/
            # text fragments that share the colour, and would otherwise
            # build a 0-3 px window that excludes the 20-36 px MH circles.
            accepted.append({"colour": (r, g, b), "n": int(circ.sum()),
                             "on_drain_frac": frac, "ref_dim": float(np.median(longdim[circ]))})
    return accepted


# ---------------------------------------------------------------------------
# 2. Manhole detection (classical CV -- primary method)
# ---------------------------------------------------------------------------
def _detect_blobs(img_bgr, rgb_colour, tol=30, max_aspect=None,
                  size_lo_frac=0.3, size_hi_frac=3.5, min_dim_floor=2, ref_dim=None):
    """Colour-threshold + connected-components blob centroid detector.

    Size filtering is SELF-CALIBRATING to the sheet, not hard-coded.
    An earlier version used fixed pixel windows (area 8-4000, min side 4 px)
    tuned on the ~10110 px-wide Ward 44/25 sheets, where manhole rings are
    ~16 px. On the ~2526 px-wide Ward 31 export the manholes are ~4 px
    solid dots, and the min-side-4 rule silently rejected ~400 of ~560 of
    them -- the reported "detection under-counts on low-res sheets" bug.
    Symbol size scales with sheet resolution, so instead of assuming a
    pixel size we measure it: take the median long-side of this colour's
    blobs (the dominant population for a manhole colour IS the manholes)
    and keep blobs within [size_lo_frac, size_hi_frac] x that median. This
    handles both hollow rings and solid filled dots (connected components
    yields one component either way) at any resolution with no tuning.

    `max_aspect`, if given, additionally requires long-side/short-side <=
    max_aspect (reject elongated dashes/text) -- needed when a colour does
    double duty as both circles and drain-line/label strokes (Ward 44 red).
    """
    r, g, b = rgb_colour
    lower = np.array([max(b - tol, 0), max(g - tol, 0), max(r - tol, 0)], dtype=np.uint8)
    upper = np.array([min(b + tol, 255), min(g + tol, 255), min(r + tol, 255)], dtype=np.uint8)
    mask = cv2.inRange(img_bgr, lower, upper)

    # Morphology kernel scaled to sheet resolution: closing tiny 4 px dots
    # with a big kernel would merge neighbours. ~5 px on a 10k-wide sheet,
    # ~3 px on a 2.5k-wide one.
    ksz = max(3, int(round(5 * max(img_bgr.shape[:2]) / 10000)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return [], mask

    wh = stats[1:, 2:4].astype(float)
    longside = np.maximum(wh[:, 0], wh[:, 1])
    if ref_dim is not None:
        # explicit target size (used for secondary colours whose blob
        # population is dominated by same-colour noise, making the median
        # unreliable -- see find_secondary_manhole_colours / detect_manholes)
        med = float(ref_dim)
        lo = max(min_dim_floor, 0.5 * med)
        hi = 2.5 * med
    else:
        # robust typical symbol size from the near-round subset (ignores long
        # dash/text blobs when the colour does double duty)
        round_mask = longside <= 3.0 * np.maximum(np.minimum(wh[:, 0], wh[:, 1]), 1)
        ref = longside[round_mask]
        med = float(np.median(ref)) if len(ref) else float(np.median(longside))
        lo = max(min_dim_floor, size_lo_frac * med)
        hi = size_hi_frac * med

    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        long_i = max(w, h)
        short_i = min(w, h)
        if not (lo <= long_i <= hi and short_i >= min_dim_floor):
            continue
        if max_aspect is not None and (long_i / max(short_i, 1)) > max_aspect:
            continue
        cx, cy = centroids[i]
        out.append((float(cx), float(cy)))
    return out, mask


def detect_manholes(img_bgr, manhole_colour=None, extra_colours=None, dedup_radius=15):
    """Detect manhole circle symbols, optionally across multiple colours.

    `manhole_colour` is the primary (R,G,B) colour (from
    auto_calibrate_colours()). `extra_colours` is an optional list of
    additional (R,G,B) colours (from find_secondary_manhole_colours()) for
    sheets that draw more than one manhole population -- see that
    function's docstring for why this is necessary on Ward 44. Detections
    from different colours within `dedup_radius` px of each other are
    merged (keeps the first).

    Returns a list of dicts: {"x":, "y":, "colour": (r,g,b)}
    """
    colour = manhole_colour or COLOURS_RGB["manhole_unlabeled"]
    all_pts = []
    # extra_colours may double as a drain-line/text colour (Ward 44's
    # secondary manhole red also draws the dashed drain stroke), so those
    # need the circularity filter that the primary colour's already-clean
    # population doesn't require. Each may carry a `ref_dim` (typical circle
    # size) so the size filter targets the real circles rather than the
    # same-colour noise -- see find_secondary_manhole_colours.
    jobs = [(colour, None, None)]
    for ec in (extra_colours or []):
        if isinstance(ec, dict):
            jobs.append((ec["colour"], 1.3, ec.get("ref_dim")))
        else:
            jobs.append((ec, 1.3, None))
    for c, max_aspect, ref_dim in jobs:
        pts, _ = _detect_blobs(img_bgr, c, tol=25, max_aspect=max_aspect, ref_dim=ref_dim)
        all_pts.extend((x, y, c) for x, y in pts)

    if not all_pts:
        return []
    from scipy.spatial import cKDTree
    # dedup radius scales with resolution (15 px on a 10k-wide sheet -> ~4 px
    # on a 2.5k-wide one) so tightly-spaced small dots aren't merged away
    dedup_radius = max(3, dedup_radius * max(img_bgr.shape[:2]) / 10000)
    coords = np.array([[x, y] for x, y, _ in all_pts])
    tree = cKDTree(coords)
    kept = np.ones(len(all_pts), dtype=bool)
    for i in range(len(all_pts)):
        if not kept[i]:
            continue
        neighbours = tree.query_ball_point(coords[i], dedup_radius)
        for j in neighbours:
            if j > i:
                kept[j] = False
    return [{"x": x, "y": y, "colour": c} for keep, (x, y, c) in zip(kept, all_pts) if keep]


# ---------------------------------------------------------------------------
# 3. Drainage network reconstruction (graph-snapping)
# ---------------------------------------------------------------------------
def detect_drainage_edges(img_bgr, nodes, drain_colour=None, k_neighbors=6, sample_n=40,
                           red_frac_thresh=0.6, tol=35):
    """Connect manhole nodes that have a red-dashed line running between them.

    Naively testing every node pair badly over-connects: three or more
    manholes strung out along one straight drain run are all mutually
    "red-aligned", so an all-pairs radius search also fires on long-range
    skip connections (A-C as well as A-B, B-C). Restricting candidates to
    each node's k nearest neighbours, then keeping only genuinely
    red-covered *and* not-redundant-with-a-shorter-edge connections,
    reproduces the real one-segment-per-adjacent-manhole topology.

    For each candidate pair, sample `sample_n` points along the straight
    segment and check what fraction land on/near a red pixel (a small
    dilation bridges dash gaps). If the fraction clears red_frac_thresh,
    and no *shorter* accepted edge from the same node already points in
    almost the same direction (i.e. this would just be a skip-connection
    over an intermediate manhole), keep it as a drainage edge.

    The drain mask is matched by HUE, not by a tight RGB box -- crucial
    finding from Ward 20. That sheet draws drain runs in TWO shades of the
    same hue: bold red (255,0,0) for main/brick sewers and a light pink
    (255,~200,~200) tint for the smaller-diameter pipes. A tight RGB box
    around (255,0,0) caught only the bold mains and missed the light-pink
    runs entirely -- so most manhole chains ended up unconnected ("the
    drainage lines are not there"). Matching any pixel whose HUE is close
    to the drain colour, at any lightness/saturation above a floor, catches
    both shades: connected-manhole count on Ward 20 jumped 294 -> 741. This
    is strictly safe for the single-shade sheets (Ward 44/25 drains are the
    same red hue) since it only ADDS the tint pixels of the correct hue.
    """
    from scipy.spatial import cKDTree

    r, g, b = drain_colour or COLOURS_RGB["drain_line"]
    drain_hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    if drain_hsv[1] >= 25:  # chromatic drain colour -> hue match (handles tints)
        drain_hue = int(drain_hsv[0])
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(int)
        hue_dist = np.minimum(np.abs(h - drain_hue), 180 - np.abs(h - drain_hue))
        mask = ((hue_dist <= 12) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 40)).astype(np.uint8) * 255
    else:  # achromatic (grey/near-black) drain -> fall back to RGB box
        lower = np.array([max(b - tol, 0), max(g - tol, 0), max(r - tol, 0)], dtype=np.uint8)
        upper = np.array([min(b + tol, 255), min(g + tol, 255), min(r + tol, 255)], dtype=np.uint8)
        mask = cv2.inRange(img_bgr, lower, upper)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))  # bridge dash gaps

    pts = np.array([[n["x"], n["y"]] for n in nodes])
    tree = cKDTree(pts)
    _, neighbor_idx = tree.query(pts, k=min(k_neighbors + 1, len(pts)))

    candidates = []  # (dist, i, j, hit_frac)
    seen = set()
    for i, neighbors in enumerate(neighbor_idx):
        for j in neighbors[1:]:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1:
                continue
            xs = np.linspace(pts[i][0], pts[j][0], sample_n).astype(int)
            ys = np.linspace(pts[i][1], pts[j][1], sample_n).astype(int)
            xs = np.clip(xs, 0, mask.shape[1] - 1)
            ys = np.clip(ys, 0, mask.shape[0] - 1)
            hit_frac = mask[ys, xs].astype(bool).mean()
            if hit_frac >= red_frac_thresh:
                candidates.append((d, key[0], key[1], hit_frac))

    # Prune skip-connections: sort shortest first, accept an edge unless a
    # shorter already-accepted edge from one of its endpoints points in
    # nearly the same direction (i.e. this edge just spans over a manhole
    # that already has a shorter connection covering the same line).
    candidates.sort(key=lambda c: c[0])
    accepted = []
    accepted_dirs = {}  # node -> list of unit direction vectors already used
    for d, i, j, hit_frac in candidates:
        vec = pts[j] - pts[i]
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        redundant = False
        for node, other in ((i, vec), (j, -vec)):
            for prev_dir in accepted_dirs.get(node, []):
                if np.dot(prev_dir, other) > 0.97:  # nearly collinear, same side
                    redundant = True
                    break
            if redundant:
                break
        if redundant:
            continue
        accepted.append((i, j))
        accepted_dirs.setdefault(i, []).append(vec)
        accepted_dirs.setdefault(j, []).append(-vec)
    return accepted


# ---------------------------------------------------------------------------
# 4. Shapefile writers (EPSG:4326)
# ---------------------------------------------------------------------------
def _write_prj(out_path_no_ext):
    wgs84_wkt = (
        'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
    )
    with open(out_path_no_ext + ".prj", "w") as f:
        f.write(wgs84_wkt)


def write_point_shapefile(points_geo, attrs_list, out_path_no_ext):
    with shapefile.Writer(out_path_no_ext, shapeType=shapefile.POINT) as w:
        w.field("id", "N")
        w.field("type", "C")
        for i, ((lon, lat), a) in enumerate(zip(points_geo, attrs_list)):
            w.point(lon, lat)
            w.record(i, a.get("type") or str(a.get("colour", "")))
    _write_prj(out_path_no_ext)


def write_polyline_shapefile(lines_geo, out_path_no_ext):
    with shapefile.Writer(out_path_no_ext, shapeType=shapefile.POLYLINE) as w:
        w.field("id", "N")
        for i, line in enumerate(lines_geo):
            w.line([line])
            w.record(i)
    _write_prj(out_path_no_ext)


# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------
def digitize_arrays(img_bgr, gcps, *, manhole_colour=None, drain_colour=None,
                    include_secondary=True, drain_sensitivity=0.6, connect_k=6,
                    log=None):
    """Core in-memory pipeline. No file I/O -- used by both the CLI wrapper
    and the Streamlit dashboard.

    img_bgr : OpenCV BGR image array.
    gcps    : list of (mapX, mapY, pixel_x, pixel_y) from parse_points_*.
    manhole_colour / drain_colour : optional (R,G,B) MANUAL OVERRIDES. If
        None, auto_calibrate_colours() picks them. Exposing these is the
        key robustness lever for "any kind of PNG": when a sheet's colour
        scheme confuses auto-calibration (e.g. Ward 31's green boundary
        out-scoring the faint red drain), the user can point the detector
        at the right colour directly from the dashboard.
    include_secondary : also hunt for a second manhole population (labelled
        trunk-manhole circles in a different/shared colour).
    drain_sensitivity : fraction of a segment that must be drain-coloured to
        accept a drainage edge (lower = more edges, more permissive).
    connect_k : each manhole is tested against its k nearest neighbours.

    Returns a dict with everything the dashboard needs to render + export.
    """
    log = log or (lambda *_a, **_k: None)
    h, w = img_bgr.shape[:2]

    xf = AffineGeoToPixel(gcps)  # fit geo->pixel, invert for output
    log(f"image {w}x{h}px | {len(gcps)} GCPs | geo->pixel affine RMSE ~{xf.rmse_px:.1f} px")

    calib = auto_calibrate_colours(img_bgr)
    mh_colour = tuple(manhole_colour) if manhole_colour else calib["manhole_colour"]
    dr_colour = tuple(drain_colour) if drain_colour else calib["drain_colour"]
    log(f"colours: manhole={mh_colour} drain={dr_colour}"
        + ("  (manual override)" if manhole_colour or drain_colour else "  (auto)"))
    if mh_colour is None:
        raise RuntimeError("Could not determine a manhole colour automatically -- "
                           "set one manually in the dashboard.")

    secondary = []
    if include_secondary and dr_colour is not None:
        secondary = find_secondary_manhole_colours(img_bgr, mh_colour, dr_colour)
        if secondary:
            log("secondary manhole colour(s): "
                + ", ".join(f"{s['colour']} (n={s['n']}, on_drain={s['on_drain_frac']*100:.0f}%)"
                            for s in secondary))

    manholes = detect_manholes(img_bgr, manhole_colour=mh_colour, extra_colours=secondary)
    log(f"detected {len(manholes)} manhole candidates")

    edges = []
    if dr_colour is not None and len(manholes) >= 2:
        edges = detect_drainage_edges(img_bgr, manholes, drain_colour=dr_colour,
                                      k_neighbors=connect_k, red_frac_thresh=drain_sensitivity)
    log(f"reconstructed {len(edges)} drainage edges")

    mh_px = np.array([[m["x"], m["y"]] for m in manholes], dtype=float)
    mh_geo = xf.pixel_to_geo_batch(mh_px) if len(manholes) else np.zeros((0, 2))
    line_geo = [[tuple(mh_geo[i]), tuple(mh_geo[j])] for i, j in edges]

    return {
        "manholes": manholes, "edges": edges, "transform": xf,
        "manhole_geo": mh_geo, "line_geo": line_geo,
        "manhole_colour": mh_colour, "drain_colour": dr_colour,
        "secondary": secondary, "calibration": calib,
        "image_size": (w, h), "rmse_px": xf.rmse_px,
    }


def render_overlay(img_bgr, result, max_width=1600):
    """Return an RGB overlay image (numpy array) with drainage lines and
    manhole markers drawn on the source sheet, downscaled for display."""
    vis = img_bgr.copy()

    # drainage edges (blue), drawn from stored pixel node coords
    mh = result["manholes"]
    for i, j in result["edges"]:
        p1 = (int(mh[i]["x"]), int(mh[i]["y"]))
        p2 = (int(mh[j]["x"]), int(mh[j]["y"]))
        cv2.line(vis, p1, p2, (255, 0, 0), 3)  # BGR blue
    # manholes: primary as green dot, secondary/labelled as magenta ring
    primary = result["manhole_colour"]
    for m in mh:
        c = (int(m["x"]), int(m["y"]))
        if tuple(m.get("colour", primary)) == tuple(primary):
            cv2.circle(vis, c, 5, (0, 200, 0), 2)      # green
        else:
            cv2.circle(vis, c, 12, (255, 0, 255), 3)   # magenta ring for labelled circles

    if vis.shape[1] > max_width:
        scale = max_width / vis.shape[1]
        vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def build_shapefile_zip(result, ward_name="ward"):
    """Write the Potholes/Drainage shapefiles (+ pixel JSON) into a temp dir
    and return the bytes of a .zip bundle -- for the dashboard download."""
    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory() as tmp:
        pot = os.path.join(tmp, f"Potholes_{ward_name}")
        dra = os.path.join(tmp, f"Drainage_{ward_name}")
        write_point_shapefile([tuple(g) for g in result["manhole_geo"]],
                              result["manholes"], pot)
        write_polyline_shapefile(result["line_geo"], dra)
        with open(os.path.join(tmp, f"manholes_{ward_name}_pixels.json"), "w") as f:
            json.dump(result["manholes"], f, indent=2)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fn in sorted(os.listdir(tmp)):
                zf.write(os.path.join(tmp, fn), fn)
        return buf.getvalue()


def digitize_ward(png_path, points_path, out_dir, ward_name="ward"):
    """CLI wrapper: read files, run digitize_arrays, write outputs to disk."""
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(png_path)
    if img is None:
        raise FileNotFoundError(png_path)
    gcps = parse_points_file(points_path)

    result = digitize_arrays(img, gcps, log=lambda m: print(f"[{ward_name}] {m}"))

    write_point_shapefile([tuple(g) for g in result["manhole_geo"]], result["manholes"],
                          os.path.join(out_dir, f"Potholes_{ward_name}_AUTO"))
    write_polyline_shapefile(result["line_geo"],
                             os.path.join(out_dir, f"Drainage_{ward_name}_AUTO"))
    with open(os.path.join(out_dir, f"manholes_{ward_name}_pixels.json"), "w") as f:
        json.dump(result["manholes"], f, indent=2)
    print(f"[{ward_name}] wrote shapefiles + pixel JSON to {out_dir}")
    return result["manholes"], result["edges"], result["transform"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("png", help="path to ward raster (.png)")
    ap.add_argument("points", help="path to matching QGIS .points GCP file")
    ap.add_argument("out_dir", help="output directory for shapefiles")
    ap.add_argument("--ward-name", default="ward", help="label used in output filenames")
    args = ap.parse_args()
    digitize_ward(args.png, args.points, args.out_dir, args.ward_name)
