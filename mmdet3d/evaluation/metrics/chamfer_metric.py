import numpy as np
from scipy.spatial import cKDTree

def chamfer_distance_per_range(p1, p2, bins=[40]):
    """
    Compute Chamfer Distance for full point clouds and per origin-distance bin.

    Args:
        p1: (N, 3) array of points
        p2: (M, 3) array of points
        bins: list of distance bin edges (e.g. [2.0, 5.0])

    Returns:
        dict: {
            'total': overall CD,
            '0.0-2.0': CD for bin 0,
            '2.0-5.0': CD for bin 1,
            '5.0-inf': CD for bin 2
        }
    """
    results = {}

    # ---- Total Chamfer Distance ----
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    dists1, _ = tree2.query(p1)
    dists2, _ = tree1.query(p2)
    cd_total = np.mean(dists1**2) + np.mean(dists2**2)
    results["total"] = cd_total

    # ---- Bin-Based Chamfer Distance ----
    dist1 = np.linalg.norm(p1, axis=1)
    dist2 = np.linalg.norm(p2, axis=1)

    labels1 = np.digitize(dist1, bins)
    labels2 = np.digitize(dist2, bins)

    for i in range(len(bins) + 1):
        bin_name = f"{bins[i-1] if i > 0 else 0.0}-{bins[i] if i < len(bins) else 'inf'}"
        points1 = p1[labels1 == i]
        points2 = p2[labels2 == i]

        if len(points1) == 0 or len(points2) == 0:
            results[bin_name] = 0
            continue

        tree1_bin = cKDTree(points1)
        tree2_bin = cKDTree(points2)
        d1_bin, _ = tree2_bin.query(points1)
        d2_bin, _ = tree1_bin.query(points2)
        cd_bin = np.mean(d1_bin**2) + np.mean(d2_bin**2)

        results[bin_name] = cd_bin

    return results
