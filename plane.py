from __future__ import annotations

import os, time, argparse
import numpy as np
from collections import Counter 
from PIL import Image            

import open3d as o3d

def now_s():
    return time.time()

def fmt_secs(x):
    if x is None: return ""
    x = float(x)
    if x < 60: return f"{x:0.1f}s"
    if x < 3600: return f"{x/60:0.1f}m"
    return f"{x/3600:0.1f}h"

class ETALogger:
    def __init__(self, total, window=200):
        self.total = max(1, int(total))
        self.window = int(window)
        self.t0 = now_s()
        self.hist = []

    def step(self, i):
        t = now_s()
        self.hist.append((i, t))
        if len(self.hist) > self.window:
            self.hist = self.hist[-self.window:]

    def eta(self, i):
        if len(self.hist) < 2: return None
        (i0, t0), (i1, t1) = self.hist[0], self.hist[-1]
        if i1 <= i0: return None
        rate = (t1 - t0) / (i1 - i0 + 1e-9)
        remain = self.total - i
        if remain <= 0: return 0.0
        return rate * remain

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    import struct
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary(path_to_model_file):
    xyzs = []
    rgbs = []
    errors = []
    point3D_ids = []

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point3D_id = read_next_bytes(fid, 8, "Q")[0]
            xyz = np.array(read_next_bytes(fid, 24, "ddd"))
            rgb = np.array(read_next_bytes(fid, 3, "BBB"))
            error = read_next_bytes(fid, 8, "d")[0]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(8 * int(track_length), 1)

            xyzs.append(xyz)
            rgbs.append(rgb)
            errors.append(error)
            point3D_ids.append(point3D_id)

    return (
        np.array(point3D_ids, dtype=np.uint64),
        np.array(xyzs, dtype=np.float64),
        np.array(rgbs, dtype=np.uint8),
        np.array(errors, dtype=np.float64),
    )

def read_images_binary(path_to_model_file):
    images = {}
    import struct
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            image_id = read_next_bytes(fid, 4, "I")[0]
            qvec = np.array(read_next_bytes(fid, 32, "dddd"))
            tvec = np.array(read_next_bytes(fid, 24, "ddd"))
            camera_id = read_next_bytes(fid, 4, "I")[0]

            # read image name
            name = ""
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name += c.decode("utf-8")

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            xys = []
            p3d_ids = []
            for _j in range(num_points2D):
                x, y = read_next_bytes(fid, 16, "dd")
                p3d_id = read_next_bytes(fid, 8, "q")[0]
                xys.append((x, y))
                p3d_ids.append(p3d_id)

            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "xys": np.array(xys, dtype=np.float64),
                "point3D_ids": np.array(p3d_ids, dtype=np.int64),
            }

    return images

# Segmentation
class SegPNG:
    def __init__(self, seg_dir, seg_ext):
        self.seg_dir = seg_dir
        self.seg_ext = seg_ext
        self.cache = {}

    def path_for(self, image_name):
        base = os.path.splitext(os.path.basename(image_name))[0]
        return os.path.join(self.seg_dir, base + self.seg_ext)

    def load(self, path):
        if path in self.cache:
            return self.cache[path]
        arr = np.array(Image.open(path))
        self.cache[path] = arr
        return arr

    def read_label_patch(self, image_name, u, v):
        path = self.path_for(image_name)
        if not os.path.isfile(path):
            return None
        arr = self.load(path)
        H, W = arr.shape[:2]
        uu = int(np.clip(round(u), 0, W - 1))
        vv = int(np.clip(round(v), 0, H - 1))
        u0, u1 = max(0, uu - 1), min(W - 1, uu + 1)
        v0, v1 = max(0, vv - 1), min(H - 1, vv + 1)
        patch = arr[v0 : v1 + 1, u0 : u1 + 1].reshape(-1)
        if patch.size == 0:
            return None
        val, _ = Counter(patch.tolist()).most_common(1)[0]
        return int(val)

def transfer_semantic_labels_from_images(  # NEW
    xyz,
    p3d_ids,
    images,
    seg_dir,
    seg_ext,
    *,
    max_obs=6,
    unlabeled_value=-1,
):
    seg = SegPNG(seg_dir, seg_ext)
    id2idx = {int(pid): i for i, pid in enumerate(np.asarray(p3d_ids).tolist())}

    labels = np.full((len(xyz),), int(unlabeled_value), dtype=np.int32)
    votes = [Counter() for _ in range(len(xyz))]

    for img in images.values():
        name = img["name"]
        xys = img["xys"]
        pids = img["point3D_ids"]

        if len(pids) == 0:
            continue
        for (u, v), pid in zip(xys, pids):
            pid = int(pid)
            if pid < 0:
                continue
            j = id2idx.get(pid, None)
            if j is None:
                continue
            if sum(votes[j].values()) >= max_obs:
                continue

            lab = seg.read_label_patch(name, float(u), float(v))
            if lab is None:
                continue
            votes[j][int(lab)] += 1

    for i, c in enumerate(votes):
        if c:
            labels[i] = int(c.most_common(1)[0][0])

    return labels

def fit_plane_tls(P):
    c = np.mean(P, axis=0)
    Q = P - c
    C = Q.T @ Q
    w, V = np.linalg.eigh(C)
    n = V[:, 0]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(n @ c)
    return n, d

def mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    m = float(np.median(x))
    return float(np.median(np.abs(x - m)))

def epsilon_plane_from_residuals(residuals: np.ndarray, *, scale: float = 2.0) -> float:
    residuals = np.asarray(residuals, dtype=np.float64)
    return float(scale * mad(residuals) + 1e-12)

def quat_shortest_from_z_to_vec(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    c = float(np.clip(z @ n, -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if c < -1.0 + 1e-10:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    axis = np.cross(z, n)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    angle = float(np.arccos(c))
    s = np.sin(angle / 2.0)
    q = np.array([np.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    return q

def connected_components_plane_graph(
    xyz: np.ndarray,
    normals: np.ndarray,
    *,
    tau_d: float,
    tau_theta_deg: float,
) -> list[np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    kdt = o3d.geometry.KDTreeFlann(pcd)

    N = xyz.shape[0]
    visited = np.zeros((N,), dtype=bool)
    cos_th = float(np.cos(np.radians(tau_theta_deg)))
    comps: list[np.ndarray] = []

    for s in range(N):
        if visited[s]:
            continue
        stack = [s]
        visited[s] = True
        comp = []
        while stack:
            i = stack.pop()
            comp.append(i)
            cnt, nn, _ = kdt.search_radius_vector_3d(xyz[i], float(tau_d))
            if cnt <= 1:
                continue
            ni = normals[i]
            for j in nn:
                if visited[j]:
                    continue
                if abs(float(ni @ normals[j])) < cos_th:
                    continue
                visited[j] = True
                stack.append(int(j))

        if len(comp) > 0:
            comps.append(np.asarray(comp, dtype=np.int64))

    return comps


def iterative_plane_extraction_tls_mad(
    xyz: np.ndarray,
    normals: np.ndarray,
    idxs: np.ndarray,
    *,
    tau_theta_deg: float,
    max_planes: int = 10,
) -> list[dict]:
    xyz = np.asarray(xyz, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    idxs = np.asarray(idxs, dtype=np.int64)

    if idxs.size < 3:
        return []

    cos_th = float(np.cos(np.radians(tau_theta_deg)))
    remain = idxs.copy()
    R0 = int(idxs.size)
    planes: list[dict] = []

    for _ in range(int(max_planes)):
        if remain.size < 3:
            break
        P = xyz[remain]
        n, d = fit_plane_tls(P)
        if n[2] < 0:
            n = -n
            d = -d

        residuals = np.abs(P @ n + d)
        eps_plane = epsilon_plane_from_residuals(residuals, scale=2.0)
        ang_ok = np.abs(normals[remain] @ n) >= cos_th
        inl_mask = (residuals <= eps_plane) & ang_ok
        inliers = remain[inl_mask]
        if inliers.size < 3:
            break

        planes.append({
            "n": n,
            "d": float(d),
            "inliers": inliers,
            "eps_plane": float(eps_plane),
        })

        residual_set = remain[~inl_mask]
        if residual_set.size < max(20, int(np.ceil(0.2 * R0))):
            break
        remain = residual_set

    return planes

# Merge coplanar planes (normal + distance threshold)
from collections import defaultdict

def merge_coplanar(planes, xyz, angle_deg, d_th, max_center_dist=None):
    if not planes:
        return []

    parent = list(range(len(planes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    cos_th = np.cos(np.radians(angle_deg))

    if max_center_dist is not None:
        centroids = np.zeros((len(planes), 3), dtype=np.float64)
        for i, pl in enumerate(planes):
            inl = np.asarray(pl["inliers"], dtype=np.int64)
            centroids[i] = np.mean(xyz[inl], axis=0) if inl.size else 0.0

    for i in range(len(planes)):
        ni, di = planes[i]["n"], planes[i]["d"]
        for j in range(i + 1, len(planes)):
            nj, dj = planes[j]["n"], planes[j]["d"]

            if abs(ni @ nj) >= cos_th and abs(di - dj) <= d_th:
                unite(i, j)

    groups = defaultdict(list)
    for i in range(len(planes)):
        groups[find(i)].append(i)

    merged = []
    for _, idxs in groups.items():
        inl = np.unique(np.concatenate([planes[k]["inliers"] for k in idxs]))
        n, d = fit_plane_tls(xyz[inl])
        merged.append({"n": n, "d": d, "inliers": inl, "label": None})

    return merged

# Save plane-colored pointcloud PLY
def _plane_color_u8(pid: int, *, seed: int = 0) -> np.ndarray:
    # Use a stable hash-like mix so colors don't depend on n_planes / max pid.
    x = (int(pid) + 1) ^ (int(seed) * 0x9E3779B1)
    rng = np.random.default_rng(x & 0xFFFFFFFF)
    return rng.integers(low=0, high=256, size=3, dtype=np.uint8)

def _save_plane_colored_pointcloud_ply(path, xyz, plane_id, *, seed=0, unassigned_rgb=(180, 180, 180)):
    plane_id = np.asarray(plane_id, dtype=np.int32)
    N = xyz.shape[0]

    colors = np.empty((N, 3), dtype=np.uint8)
    colors[:] = np.array(unassigned_rgb, dtype=np.uint8)[None, :]

    valid = np.where(plane_id >= 0)[0]
    if valid.size:
        for pid in np.unique(plane_id[valid]):
            colors[plane_id == pid] = _plane_color_u8(int(pid), seed=int(seed))

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int plane_id\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b), pid in zip(xyz, colors, plane_id):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {int(pid)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points3d_bin")
    ap.add_argument("--images_bin", required=True)
    ap.add_argument("--seg_dir", required=True)
    ap.add_argument("--seg_ext", default="_pan2sem.png")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--out_ply", default=None, help="optional: save raw pointcloud as PLY")
    ap.add_argument("--out_planes_pointcloud_ply", default=None, help="optional: in-plane points colored by plane_id")

    ap.add_argument("--tau_d", type=float, default=0.02, help="spatial distance threshold τ_d")
    ap.add_argument("--tau_theta", type=float, default=20.0, help="angular threshold τ_θ (deg)")
    ap.add_argument("--merge_tau_d", type=float, default=None, help="plane spatial merge threshold (default: 1.1*τ_d)")
    ap.add_argument("--merge_tau_theta", type=float, default=None, help="plane angular merge threshold (default: 1.1*τ_θ)")

    args = ap.parse_args()

    t0 = now_s()

    # Load point cloud
    print(f"[{fmt_secs(now_s()-t0):>8}] loading COLMAP points3D.bin ...")
    p3d_ids, xyz, rgb, err = read_points3D_binary(args.points3d_bin)
    xyz = xyz.astype(np.float64)

    # Load images (for label transfer)
    print(f"[{fmt_secs(now_s()-t0):>8}] loading COLMAP images.bin ...")
    images = read_images_binary(args.images_bin)

    print(f"[{fmt_secs(now_s()-t0):>8}] transferring semantic labels (seg PNG vote) ...")
    labels = transfer_semantic_labels_from_images(
        xyz,
        p3d_ids,
        images,
        args.seg_dir,
        args.seg_ext,
        max_obs=6,
        unlabeled_value=-1,
    )
    n_unl = int(np.sum(labels < 0))
    print(f"[INFO] labels done: labeled={len(labels)-n_unl}/{len(labels)} unlabeled={n_unl}")

    #Normal estimation (seg-aware weighted PCA as described)
    print(f"[{fmt_secs(now_s()-t0):>8}] estimating normals (distance/label-weighted PCA) ...")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    kdt = o3d.geometry.KDTreeFlann(pcd)
    normals = np.zeros_like(xyz)

    for i in range(len(xyz)):
        cnt, idxs, d2 = kdt.search_knn_vector_3d(xyz[i], 24 + 1)
        if cnt <= 3:
            normals[i] = np.array([0, 0, 1.0], float)
            continue

        idxs = np.array([j for j in idxs if j != i], dtype=int)
        dj = np.sqrt(np.maximum(0.0, np.array(d2, float)))[1:len(idxs)+1]
        s_i = np.median(dj) + 1e-9
        w_dist = np.exp(-0.5 * (dj / (1.5 * s_i))**2)
        w_lab  = np.where(labels[idxs] == labels[i], 1.0, 0.3)
        w = w_dist * w_lab
        w = w / (np.sum(w) + 1e-9)

        P = xyz[idxs]
        c = np.sum(P * w[:, None], axis=0)
        Q = (P - c) * np.sqrt(w[:, None])
        C = Q.T @ Q
        _, V = np.linalg.eigh(C)
        normals[i] = V[:, 0] / (np.linalg.norm(V[:, 0]) + 1e-12)

    #Plane graph connected components (Eq. 1)
    print(f"[{fmt_secs(now_s()-t0):>8}] building plane graph components (tau_d/tau_theta) ...")
    comps = connected_components_plane_graph(
        xyz,
        normals,
        tau_d=float(args.tau_d),
        tau_theta_deg=float(args.tau_theta),
    )
    comps = [c for c in comps if c.size >= 20]
    print(f"[{fmt_secs(now_s()-t0):>8}] components: {len(comps)} (kept >=20 pts)")
    init_planes: list[dict] = []
    eta_c = ETALogger(total=len(comps), window=200)
    for ci, idxs in enumerate(comps, 1):
        sub = iterative_plane_extraction_tls_mad(
            xyz,
            normals,
            idxs,
            tau_theta_deg=float(args.tau_theta),
            max_planes=10,
        )
        init_planes.extend(sub)

        if (ci % 200) == 0:
            eta_c.step(ci)
            eta = eta_c.eta(ci)
            print(
                f"[{fmt_secs(now_s()-t0):>8}] components {ci}/{len(comps)} | planes {len(init_planes)}"
                + (f" | ETA {fmt_secs(eta)}" if eta else "")
            )

    #Merge planes
    merge_tau_d = float(args.merge_tau_d) if args.merge_tau_d is not None else float(1.1 * args.tau_d)
    merge_tau_theta = float(args.merge_tau_theta) if args.merge_tau_theta is not None else float(1.1 * args.tau_theta)
    print(f"[INFO] planes before merge: {len(init_planes)}")
    planes = merge_coplanar(
        init_planes,
        xyz,
        angle_deg=float(merge_tau_theta),
        d_th=float(merge_tau_d),
        max_center_dist=None,
    )
    print(f"[INFO] planes after  merge: {len(planes)}")

    #Assign each point to the best plane
    plane_id = -np.ones((len(xyz),), dtype=np.int32)
    best_dist = np.full((len(xyz),), np.inf, dtype=np.float64)
    for pid, pl in enumerate(planes):
        n = np.asarray(pl["n"], dtype=np.float64)
        d = float(pl["d"])
        dist = np.abs(xyz @ n + d)
        better = dist < best_dist
        best_dist[better] = dist[better]
        plane_id[better] = int(pid)
    eps_per_plane = np.array([float(pl.get("eps_plane", np.nan)) for pl in planes], dtype=np.float64)
    global_eps = float(np.nanmedian(eps_per_plane)) if np.isfinite(eps_per_plane).any() else 0.0
    for pid in range(len(planes)):
        eps = float(eps_per_plane[pid]) if np.isfinite(eps_per_plane[pid]) else global_eps
        mask = plane_id == pid
        too_far = mask & (best_dist > eps)
        plane_id[too_far] = -1

    # Quaternion initialization
    quats = np.zeros((len(xyz), 4), dtype=np.float64)
    for pid, pl in enumerate(planes):
        q = quat_shortest_from_z_to_vec(pl["n"])
        quats[plane_id == pid] = q[None, :]
    quats[plane_id < 0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)[None, :]

    #Save outputs
    print(f"[{fmt_secs(now_s()-t0):>8}] saving npz: {args.out_npz}")
    np.savez(
        args.out_npz,
        xyz=xyz.astype(np.float32),
        rgb=rgb.astype(np.uint8),
        quat=quats.astype(np.float32),
        plane_id=plane_id.astype(np.int32),
    )

    if args.out_ply is not None:
        print(f"[{fmt_secs(now_s()-t0):>8}] saving ply: {args.out_ply}")
        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(xyz)
        pcd_out.colors = o3d.utility.Vector3dVector((rgb.astype(np.float32) / 255.0))
        o3d.io.write_point_cloud(args.out_ply, pcd_out)

    if args.out_planes_pointcloud_ply is not None:
        print(f"[{fmt_secs(now_s()-t0):>8}] saving plane-colored pointcloud ply: {args.out_planes_pointcloud_ply}")
        _save_plane_colored_pointcloud_ply(
            args.out_planes_pointcloud_ply,
            xyz,
            plane_id,
            seed=0,
            unassigned_rgb=(180, 180, 180),
        )

    print(f"[DONE] total time: {fmt_secs(now_s()-t0)}")


if __name__ == "__main__":
    main()
