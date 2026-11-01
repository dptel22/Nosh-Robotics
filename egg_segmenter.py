"""How this works:
1. Denoises the BGR frame using Gaussian or bilateral filtering.
2. Generates an initial rough foreground mask using thresholding or GrabCut.
3. Removes skin-colored pixels, cleans binary noise with morphological ops, and fills holes.
4. Identifies connected components and filters candidate egg blobs by geometric score/solidity.
5. Renders selected blobs with optional shape refinement (hull, ellipse, or GrabCut boundary polish).
"""

import cv2
import numpy as np
import scipy.ndimage as ndi

DEFAULTS = dict(
    denoise="gauss",
    mode="otsu_gray",
    s_max=65,
    v_min=150,
    l_min=140,
    ab_tol=22,
    skin=False,
    morph=False,
    k_open=5,
    k_close=15,
    select="largest",
    min_area=0.005,
    max_area=0.98,
    min_solid=0.9,
    refine=None,
    gc_margin=0.08,
    gc_iters=4,
)


def kernel(size):
    """Elliptical structuring element for morphological operations."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def largest_contour(blob):
    """Return the largest external contour of a binary mask."""
    contours = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return max(contours, key=cv2.contourArea)


def solidity(blob):
    """Blob area divided by the area of its convex hull. A smooth egg is close to 1."""
    contour = largest_contour(blob)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return cv2.contourArea(contour) / hull_area if hull_area else 0.0


def egg_score(blob):
    """How egg-like is this blob? Convex, ellipse-shaped, near the centre and big all push the score up."""
    contour = largest_contour(blob)
    area = cv2.contourArea(contour)
    height, width = blob.shape
    if area < 1 or len(contour) < 5:
        return 0.0
    (cx, cy), (ma, ma_major), _ = cv2.fitEllipse(contour)
    ellipse_area = np.pi * ma * ma_major / 4
    ellipse_fit = min(area, ellipse_area) / max(area, ellipse_area, 1)
    off_centre = np.hypot(cx - width / 2, cy - height / 2) / np.hypot(width / 2, height / 2)
    return solidity(blob) * ellipse_fit * (1 - 0.5 * off_centre) * np.sqrt(area / (height * width))


def grabcut_refine(img, mask, iterations=3, ring=15):
    """Pull the edge onto the real boundary: inside = sure egg, a ring around it = unsure, outside = sure background."""
    state = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    state[cv2.dilate(mask, kernel(2 * ring + 1)) > 0] = cv2.GC_PR_BGD
    state[mask > 0] = cv2.GC_PR_FGD
    state[cv2.erode(mask, kernel(2 * ring + 1)) > 0] = cv2.GC_FGD
    cv2.grabCut(img, state, None, np.zeros((1, 65)), np.zeros((1, 65)), iterations, cv2.GC_INIT_WITH_MASK)
    return np.where((state == cv2.GC_FGD) | (state == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def denoise(bgr, kind):
    """Smooth the image to reduce camera sensor noise and speckles."""
    if kind == "gauss":
        return cv2.GaussianBlur(bgr, (5, 5), 0)
    if kind == "bilateral":
        return cv2.bilateralFilter(bgr, 9, 50, 50)
    return bgr


def rough_mask(img, cfg):
    """Step 1: a first black/white guess (white = maybe egg)."""
    mode = cfg["mode"]
    if mode == "otsu_gray":  # brightness only
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if mode == "grabcut":  # no colour rule: assume the egg is big and roughly central
        height, width = img.shape[:2]
        margin = int(cfg["gc_margin"] * min(height, width))
        state = np.zeros((height, width), np.uint8)
        box = (margin, margin, width - 2 * margin, height - 2 * margin)
        cv2.grabCut(img, state, box, np.zeros((1, 65)), np.zeros((1, 65)), cfg["gc_iters"], cv2.GC_INIT_WITH_RECT)
        return np.where((state == cv2.GC_FGD) | (state == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # "hsv" / "hsv_lab": a white egg is bright (high V) and colourless (low S)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, cfg["v_min"]), (180, cfg["s_max"], 255))
    if mode == "hsv_lab":  # also require neutral a*, b* in LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        low = (cfg["l_min"], 128 - cfg["ab_tol"], 128 - cfg["ab_tol"])
        high = (255, 128 + cfg["ab_tol"], 128 + cfg["ab_tol"])
        mask &= cv2.inRange(lab, low, high)
    return mask


def pick_blobs(mask, cfg):
    """Step 3: from all white regions, keep the one(s) that should be the egg."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = stats[:, cv2.CC_STAT_AREA]
    blob = lambda k: (labels == k).astype(np.uint8)
    size_ok = [k for k in range(1, count) if cfg["min_area"] <= areas[k] / mask.size <= cfg["max_area"]]

    if cfg["select"] == "largest":
        chosen = [1 + int(areas[1:].argmax())] if count > 1 else []
    elif cfg["select"] == "score":  # the single most egg-like blob
        scores = {k: egg_score(blob(k)) for k in size_ok}
        chosen = [max(scores, key=scores.get)] if scores else []
    else:  # "shape": every smooth, convex blob (handles several eggs)
        chosen = [k for k in size_ok if solidity(blob(k)) >= cfg["min_solid"]]
    return [blob(k) for k in chosen]


def draw_blob(canvas, blob, method):
    """Step 4: copy the blob onto the canvas, optionally replacing it by its convex hull or a fitted ellipse."""
    hull = cv2.convexHull(largest_contour(blob))
    if method == "hull":
        cv2.fillPoly(canvas, [hull], 255)
    elif method == "ellipse" and len(hull) >= 5:
        cv2.ellipse(canvas, cv2.fitEllipse(hull), 255, -1)
    else:
        canvas[blob > 0] = 255


def seg(bgr, cfg):
    """The frozen Task 2 algorithm. Returns a 0/255 mask."""
    cv2.setRNGSeed(0)  # GrabCut starts from random k-means centres; seeding makes every run give the same mask
    img = denoise(bgr, cfg["denoise"])
    mask = rough_mask(img, cfg)

    if cfg["skin"]:  # remove skin-coloured pixels (YCrCb range)
        skin = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb), (0, 133, 77), (255, 173, 127))
        mask[skin > 0] = 0
    if cfg["morph"]:  # open = remove specks, close = fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(cfg["k_open"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(cfg["k_close"]))
    mask = (ndi.binary_fill_holes(mask > 0) * 255).astype(np.uint8)

    result = np.zeros_like(mask)
    for blob in pick_blobs(mask, cfg):
        draw_blob(result, blob, cfg["refine"])
    if cfg["refine"] == "grabcut" and result.any():
        result = grabcut_refine(img, result)
    return result
