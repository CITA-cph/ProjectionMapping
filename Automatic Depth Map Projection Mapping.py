#!/usr/bin/env python3
"""
Shahriar Akbari 25.02.2026 

Live Depth analysis projection using Zivid internal projector.

- Uses 3D capture settings loaded from a Zivid .yml file (your preset).
- Captures initial 3Dscan as reference plane.

- Then loops:
    * capture 3D
    * compute height relative to that plane
    * build a height colormap
    * project it for 2 seconds with the internal projector
    * repeat

You can move/deform the object between cycles and see updated height analysis.
"""

import pathlib
import time

import cv2
import numpy as np
import zivid

# Path to your Zivid 3D settings exported from Zivid Studio
# e.g. File -> Export Capture Settings
SETTINGS_FILE = r"C:\Users\sakb\OneDrive - Det Kongelige Akademi\Desktop\reflective.yml"   # <-- CHANGE THIS to your .yml setting filename


# ----------------- Geometry helpers ----------------- #

def fit_plane_least_squares(points: np.ndarray):
    """
    Fit a plane n·X + d = 0 in least-squares sense.

    points: (N,3) array of XYZ points (meters), NaNs filtered out beforehand.

    Returns:
        normal: (3,) unit normal vector
        d:      float
    """
    if points.size == 0:
        raise ValueError("No valid points for plane fitting")

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    # Normal is eigenvector corresponding to smallest singular value
    normal = vh[-1, :]
    normal /= np.linalg.norm(normal)

    d = -np.dot(normal, centroid)
    return normal, d


def compute_heights(xyz: np.ndarray, normal: np.ndarray, d: float):
    """
    Compute signed distance from each XYZ point to plane n·X + d = 0.

    xyz:    (H,W,3)
    normal: (3,)
    d:      float

    Returns:
        heights: (H,W) array
    """
    return (
        xyz[..., 0] * normal[0]
        + xyz[..., 1] * normal[1]
        + xyz[..., 2] * normal[2]
        + d
    )


def heights_to_colormap_bgra(
    heights: np.ndarray,
    valid_mask: np.ndarray,
    min_h: float = None,
    max_h: float = None,
):
    """
    Map heights to a BGRA color image using OpenCV's JET colormap.

    heights:     (H,W)
    valid_mask:  (H,W) boolean mask of valid 3D points
    min_h, max_h: height range for colormap; if None use percentiles of valid values

    Returns:
        colors_bgra: (H,W,4) uint8
    """
    h_valid = heights[valid_mask]

    if h_valid.size == 0:
        raise ValueError("No valid heights for colormap")

    if min_h is None:
        min_h = np.percentile(h_valid, 5.0)
    if max_h is None:
        max_h = np.percentile(h_valid, 95.0)

    # Avoid degenerate range
    if np.isclose(max_h, min_h):
        max_h = min_h + 1e-6

    # Normalize to 0..1
    norm = np.zeros_like(heights, dtype=np.float32)
    clipped = np.clip(heights, min_h, max_h)
    norm[valid_mask] = (clipped[valid_mask] - min_h) / (max_h - min_h)

    # Convert to 0..255 and JET colormap
    norm_8u = (norm * 255.0).astype(np.uint8)
    colors_bgr = cv2.applyColorMap(norm_8u, cv2.COLORMAP_JET)  # (H,W,3), BGR

    # Alpha: 255 for valid, 0 for invalid
    alpha = np.zeros(heights.shape + (1,), dtype=np.uint8)
    alpha[valid_mask] = 255

    colors_bgra = np.concatenate([colors_bgr, alpha], axis=2)
    return colors_bgra


# ----------------- Main ----------------- #

def main():
    app = zivid.Application()
    print("Connecting to camera")
    camera = app.connect_camera()

    # ---- Load 3D settings from your .yml file ----
    print(f"Loading 3D settings from file: {SETTINGS_FILE}")
    settings_3d = zivid.Settings.load(SETTINGS_FILE)

    # ---- Capture reference frame, fit plane ----
    print("Capturing reference frame for plane (empty / flat support)")
    frame_ref = camera.capture(settings_3d)
    pc_ref = frame_ref.point_cloud()
    xyz_ref = pc_ref.copy_data("xyz")       # (H,W,3)
    valid_ref = np.isfinite(xyz_ref).all(axis=2)

    if not np.any(valid_ref):
        raise RuntimeError("No valid points in reference frame")

    normal, d = fit_plane_least_squares(xyz_ref[valid_ref].reshape(-1, 3))
    print(f"Plane normal: {normal}, d: {d:.6f} (n·X + d = 0)")

    # ---- Prepare projector image buffer ----
    proj_h, proj_w = zivid.projection.projector_resolution(camera)
    background_color = (0, 0, 0, 255)
    projector_image = np.full((proj_h, proj_w, 4), background_color, dtype=np.uint8)

    # ---- Output folder for saving projector images ----
    out_dir = pathlib.Path(".")
    out_dir.mkdir(exist_ok=True)

    print("Entering live loop.")
    print("Each cycle: capture -> compute height -> project for 2 seconds -> repeat")
    print("Press Ctrl+C in the terminal to stop.\n")

    frame_counter = 0

    try:
        while True:
            frame_counter += 1
            t0 = time.time()

            # --- Capture with your preset settings ---
            frame = camera.capture(settings_3d)
            pc = frame.point_cloud()
            xyz = pc.copy_data("xyz")       # (H,W,3)
            H, W, _ = xyz.shape
            valid = np.isfinite(xyz).all(axis=2)

            if not np.any(valid):
                print("Warning: no valid points this frame")
                continue

            # --- Height computation ---
            heights = compute_heights(xyz, normal, d)
            colors_bgra_full = heights_to_colormap_bgra(heights, valid)

            # --- Map valid 3D points to projector pixels ---
            pts_cam = xyz[valid]  # (N,3)
            proj_pixels = zivid.projection.pixels_from_3d_points(camera, pts_cam)

            # Clear projector buffer
            projector_image[...] = background_color

            valid_indices = np.argwhere(valid)     # (N,2) row, col
            colors_valid = colors_bgra_full[valid] # (N,4)

            for (row, col), (u_proj, v_proj), color in zip(
                valid_indices, proj_pixels, colors_valid
            ):
                u = int(round(u_proj))
                v = int(round(v_proj))
                if 0 <= v < proj_h and 0 <= u < proj_w:
                    projector_image[v, u, :] = color

            # Slight blur to fill holes / smooth
            projector_image[:] = cv2.GaussianBlur(projector_image, (3, 3), 0)

            # --- Save projector image for debugging / logging ---
            out_path = out_dir / f"height_map_frame_{frame_counter:04d}.png"
            cv2.imwrite(str(out_path), projector_image)
            print(f"[Frame {frame_counter}] Saved {out_path}")

            # --- Show on internal projector for 2 seconds ---
            # Projection is active only inside this context.
            with zivid.projection.show_image_bgra(camera, projector_image):
                print(f"[Frame {frame_counter}] Projecting height map for 2 seconds...")
                time.sleep(120.0)

            t1 = time.time()
            print(f"[Frame {frame_counter}] Cycle time: {(t1 - t0)*1000:.1f} ms\n")

            # After this, loop repeats: next capture / update / projection

    except KeyboardInterrupt:
        print("Stopping live loop")

    print("Done")


if __name__ == "__main__":
    main()
