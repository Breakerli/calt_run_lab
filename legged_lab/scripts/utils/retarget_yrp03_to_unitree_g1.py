"""yrp03-v2 → unitree_g1 (G1 29 DoF) AMP expert / motion visualization 的 FK/IK 重定向。

参考 legged_lab/scripts/utils/retarget_ultra_to_ryp01_v2.py (方案 B)，把 yrp03_v2 的
AMP expert txt (42 维) 与 motion visualization txt (38 维 eric-zhao schema) 重定向为
G1 的同格式数据，替换 unitree_g1 env 端逐字节复制的 yrp03_v2 占位数据。

核心逻辑:
1. 通用 URDF 解析 (跳过 floating base 关节), 构建运动链, 向量化 FK (整条轨迹批处理)。
2. 源 yrp03 FK 提取关键点: 髋(hip_pitch link) / 膝 / 踝(ankle_pitch link) / 肩 / 手(root 系,
   手 = 肩 + R_shoulder @ SRC_ARM_VEC, 与 yrp03 env get_amp_obs_for_expert_trans 口径一致)。
3. 目标 G1 腿 6-DoF 阻尼最小二乘(DLS) IK: 拟合按腿长比缩放后的踝位置 + 大腿方向,
   初值同源映射并逐帧热启动, ankle_roll 无源对应 -> 正则回 0 (同 yrp03 的零填充设计);
   肩 1-DoF IK 拟合手位置 (手 = 肩 + R_shoulder @ G1_ARM_VEC, 与 unitree_g1 env 口径一致);
   腰 yaw 直接映射。G1 肩的父链经过 waist_roll/pitch (子链 IK 中置 0 一并建模)。
4. np.gradient 求关节速度; 目标 FK 重算 ee (足端 = ankle_roll_link, 与 unitree_g1 env
   feet_body_ids 一致), 写出同格式 42 维 expert txt。
   vis 文件: 腿 5-DoF IK 写回 [6:16], root z 与 lin_vel 按站立高度比 (0.793/1.20) 缩放。

运行示例:
  python legged_lab/scripts/utils/retarget_yrp03_to_unitree_g1.py
  python legged_lab/scripts/utils/retarget_yrp03_to_unitree_g1.py --expert_files stand_ultra_yaw2.txt --ik_iters 60
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# --- Constants & Config ---

YRP03_URDF = "legged_lab/assets/yrp03_v2/urdf/yrp03-v2.urdf"
G1_URDF = "legged_lab/assets/unitree_g1/urdf/g1_custom_collision_29dof.urdf"
SRC_EXPERT_DIR = "legged_lab/envs/yrp03_v2/datasets/motion_amp_expert"
OUT_EXPERT_DIR = "legged_lab/envs/unitree_g1/datasets/motion_amp_expert"
SRC_VIS_DIR = "legged_lab/envs/yrp03_v2/datasets/motion_visualization"
OUT_VIS_DIR = "legged_lab/envs/unitree_g1/datasets/motion_visualization"
DEFAULT_EXPERT_FILES = [
    "stand_ultra_yaw2.txt",
    "walk_ultra_yaw2.txt",
    "run_17200_9mps.txt",
    "collected_20260602_160257/env000_vxp15p00_vyp0p00_wp0p00.txt",
]
DEFAULT_VIS_FILES = ["run_14mps_play_newultra.txt"]

# AMP expert 42 维帧的关节排布 (pos/vel 同序, 源 yrp03 与目标 G1 语义一致):
# [0:15]  pos: shoulder_r, shoulder_l, 右腿(yaw,roll,pitch,knee,ankle_pitch,ankle_roll),
#          左腿(...6...), waist_yaw; [15:30] vel 同序; [30:42] ee(root 系): lh, rh, lf, rf
AMP_POS_END = 15
AMP_DIM = 42

# 站立 pelvis 高度 (yrp03_v2 cfg init 1.20 / g1_mocap xml 0.793), 用于 root z 与 vel 缩放
SRC_STAND_HEIGHT = 1.20
TGT_STAND_HEIGHT = 0.793

SRC_ARM_VEC = np.array([0.0, 0.0, -0.4])      # yrp03 env left_arm_local_vec
G1_ARM_VEC = np.array([0.24, 0.0, -0.10])     # unitree_g1 env left_arm_local_vec

FD_H = 1e-5        # 数值雅可比步长 (rad)
DLS_LAM = 1e-2     # DLS 阻尼
MAX_STEP = 0.3     # 单迭代最大关节增量 (rad)
DIR_W = 0.1        # 大腿方向残差权重 (等效米)
Q_W = 0.02         # 关节正则化权重: 约束冗余自由度停留在同源映射初值附近 (等效米/rad)
IK_ITERS = 60
IK_TOL = 2e-3      # 踝位置残差收敛阈值 (m)
SH_ITERS = 10

# 机器人相关的名字映射: yrp03 用 *_l/_r 后缀, G1 用 left_*/right_* 前缀
ROBOT_NAMES = {
    "src": {
        "j_hip_yaw": "hip_yaw_{s}_joint", "j_hip_roll": "hip_roll_{s}_joint",
        "j_hip_pitch": "hip_pitch_{s}_joint", "j_knee": "knee_pitch_{s}_joint",
        "j_ankle_pitch": "ankle_pitch_{s}_joint", "j_ankle_roll": "ankle_roll_{s}_joint",
        "j_shoulder": "shoulder_pitch_{s}_joint", "j_waist": "waist_yaw_joint",
        "l_hip_pitch": "hip_pitch_{s}_link", "l_knee": "knee_pitch_{s}_link",
        "l_ankle_pitch": "ankle_pitch_{s}_link", "l_shoulder": "shoulder_pitch_{s}_link",
    },
    "tgt": {
        "j_hip_yaw": "{p}hip_yaw_joint", "j_hip_roll": "{p}hip_roll_joint",
        "j_hip_pitch": "{p}hip_pitch_joint", "j_knee": "{p}knee_joint",
        "j_ankle_pitch": "{p}ankle_pitch_joint", "j_ankle_roll": "{p}ankle_roll_joint",
        "j_shoulder": "{p}shoulder_pitch_joint", "j_waist": "waist_yaw_joint",
        "l_hip_pitch": "{p}hip_pitch_link", "l_knee": "{p}knee_link",
        "l_ankle_pitch": "{p}ankle_pitch_link", "l_shoulder": "{p}shoulder_pitch_link",
    },
}
LEG_KEYS = ("j_hip_yaw", "j_hip_roll", "j_hip_pitch", "j_knee", "j_ankle_pitch", "j_ankle_roll")

# 手臂整体前摆偏置 (rad): yrp03 源手臂自然下垂, G1 参考轨迹要求手臂整体再前摆该角度。
# 绕 root 系 Y 轴 (横向) 旋转目标手臂向量, 使下垂手臂向 +X (前) 摆。
ARM_SWING_FORWARD = np.pi / 2
_c, _s = np.cos(ARM_SWING_FORWARD), np.sin(ARM_SWING_FORWARD)
ARM_BIAS_ROT = np.array([[_c, 0.0, -_s], [0.0, 1.0, 0.0], [_s, 0.0, _c]])  # RotY(-θ)


def N(robot: str, key: str, s: str) -> str:
    """按机器人取关节/连杆名, s in {"l", "r"}。"""
    fmt = ROBOT_NAMES[robot][key]
    return fmt.format(s=s, p="left_" if s == "l" else "right_")


# AMP 序关节名 (语义序: shoulder_r, shoulder_l, right_leg 6, left_leg 6, waist)
def AMP_JOINT_NAMES(robot: str) -> tuple[str, ...]:
    return (
        N(robot, "j_shoulder", "r"), N(robot, "j_shoulder", "l"),
        *[N(robot, k, "r") for k in LEG_KEYS],
        *[N(robot, k, "l") for k in LEG_KEYS],
        N(robot, "j_waist", "l"),
    )


# --- URDF 解析与向量化 FK ---

def parse_urdf(urdf_path: str) -> dict:
    """解析 URDF 为运动链: 拓扑序关节列表 + 根 link。仅支持单树 revolute/fixed;
    floating/planar 等基座关节跳过 (warning), 此时根 link 为pelvis/base 等实体连杆。"""
    path = Path(urdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"URDF 不存在: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    joints = {}
    skipped = []
    for j in root.iter(f"{ns}joint"):
        name = j.get("name")
        jtype = j.get("type")
        if jtype not in ("revolute", "fixed"):
            skipped.append((name, jtype))
            continue
        origin = j.find(f"{ns}origin")
        axis = j.find(f"{ns}axis")
        limit = j.find(f"{ns}limit")
        if j.find(f"{ns}parent") is None or j.find(f"{ns}child") is None:
            skipped.append((name, jtype))
            continue
        joints[name] = {
            "name": name,
            "type": jtype,
            "parent": j.find(f"{ns}parent").get("link"),
            "child": j.find(f"{ns}child").get("link"),
            "xyz": np.array((origin.get("xyz") or "0 0 0").split(), dtype=np.float64) if origin is not None else np.zeros(3),
            "rpy": rpy_mat(np.array((origin.get("rpy") or "0 0 0").split(), dtype=np.float64)) if origin is not None else np.eye(3),
            "axis": np.array((axis.get("xyz") if axis is not None else "1 0 0").split(), dtype=np.float64),
            "lower": float(limit.get("lower")) if limit is not None else -np.inf,
            "upper": float(limit.get("upper")) if limit is not None else np.inf,
        }
    if skipped:
        print(f"[URDF] {path.name}: 跳过非 revolute/fixed 关节: {skipped}")
    if not joints:
        raise ValueError(f"URDF 中未找到关节: {urdf_path}")
    children = {j["child"] for j in joints.values()}
    roots = [j["parent"] for j in joints.values() if j["parent"] not in children]
    if len(set(roots)) != 1:
        raise ValueError(f"{urdf_path} 根 link 不唯一: {set(roots)}")
    root_link = roots[0]
    # 拓扑排序(BFS), 单树假设
    by_parent: dict[str, list[str]] = {}
    for j in joints.values():
        by_parent.setdefault(j["parent"], []).append(j["name"])
    order, queue = [], [root_link]
    while queue:
        link = queue.pop(0)
        for name in by_parent.get(link, []):
            order.append(joints[name])
            queue.append(joints[name]["child"])
    if len(order) != len(joints):
        raise ValueError(f"{urdf_path} 关节图不是单树, 无法拓扑排序")
    for i, j in enumerate(order):
        j["qidx"] = i if j["type"] == "revolute" else -1
    rev_count = sum(j["type"] == "revolute" for j in order)
    print(f"[URDF] {path.name}: root={root_link}, joints={len(order)} (revolute={rev_count})")
    return {"root": root_link, "joints": order}


def rpy_mat(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cx, sx, cy, sy, cz, sz = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])


def axis_angle_mat(axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """轴角旋转, ang (N,) -> (N,3,3)"""
    a = axis / np.linalg.norm(axis)
    c, s = np.cos(ang), np.sin(ang)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return (np.eye(3)[None] * c[:, None, None] + np.outer(a, a)[None] * (1 - c[:, None, None])
            + K[None] * s[:, None, None])


def fk_batch(chain: dict, q: np.ndarray) -> tuple[dict, dict]:
    """正向运动学: q (N, n_revolute)(拓扑序) -> ({link: (N,3) 位置}, {link: (N,3,3) 姿态})。
    按父 link 查表累积, 支持多分支树; 拓扑序保证父先于子处理。"""
    n = q.shape[0]
    pos = {chain["root"]: np.zeros((n, 3))}
    rot = {chain["root"]: np.broadcast_to(np.eye(3), (n, 3, 3))}
    for j in chain["joints"]:
        R0, p0 = rot[j["parent"]], pos[j["parent"]]
        p = p0 + np.einsum("nij,j->ni", R0, j["xyz"])
        R = np.einsum("nij,jk->nik", R0, j["rpy"])
        if j["qidx"] >= 0:
            R = np.einsum("nij,njk->nik", R, axis_angle_mat(j["axis"], q[:, j["qidx"]]))
        pos[j["child"]], rot[j["child"]] = p, R
    return pos, rot


def subchain(chain: dict, keep_names: list[str]) -> dict:
    """取子链: 关节仍按拓扑序保存 (FK 要求父先于子), 但 qidx 按 keep_names
    的语义位置赋值, 使 q 向量列序与 keep_names 一致 (G1 腿拓扑序 pitch→roll→yaw
    与 AMP 语义序 yaw→roll→pitch 不同, 错位会导致 yaw/pitch 互换)。"""
    name_set, joints = set(keep_names), []
    qidx_of = {name: i for i, name in enumerate(keep_names)}
    for j in chain["joints"]:
        if j["name"] in name_set:
            jj = dict(j)
            jj["qidx"] = qidx_of[jj["name"]]
            joints.append(jj)
    if len(joints) != len(keep_names):
        missing = name_set - {j["name"] for j in joints}
        raise KeyError(f"子链关节在 URDF 中缺失: {missing}")
    return {"root": chain["root"], "joints": joints}


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("零向量归一化")
    return v / n


# --- DLS 数值 IK ---

def dls_ik(resid_fn, q0: np.ndarray, lo: np.ndarray, hi: np.ndarray,
           iters: int, tol: float) -> tuple[np.ndarray, np.ndarray, float]:
    """阻尼最小二乘 IK, 全程向量化 (N, dof)。resid_fn(q) -> (residual (N,R), aux)。"""
    q = q0.copy()
    r, aux = resid_fn(q)
    fin = 0.0
    for _ in range(iters):
        n_dof = q.shape[1]
        J = np.zeros((q.shape[0], r.shape[1], n_dof))
        for j in range(n_dof):
            qp = q.copy()
            qp[:, j] += FD_H
            rp, _ = resid_fn(qp)
            J[:, :, j] = (rp - r) / FD_H
        dq = np.linalg.solve(
            J.transpose(0, 2, 1) @ J + (DLS_LAM ** 2) * np.eye(n_dof),
            J.transpose(0, 2, 1) @ r[..., None],
        )[..., 0]
        q = np.clip(q - np.clip(dq, -MAX_STEP, MAX_STEP), lo, hi)
        r, aux = resid_fn(q)
        fin = float(np.max(np.linalg.norm(r[:, :3], axis=1)))
        if fin < tol:
            break
    return q, r, fin


# --- AMP expert txt 读写 ---

def load_amp_txt(path: Path, min_cols: int = AMP_DIM) -> tuple[np.ndarray, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"expert 文件不存在: {path}")
    with path.open(encoding="utf-8") as f:
        js = json.load(f)
    frames = np.asarray(js["Frames"], dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] < min_cols:
        raise ValueError(f"{path} 帧维度异常: {frames.shape}, 期望 (N,>={min_cols})")
    meta = {k: js[k] for k in ("LoopMode", "FrameDuration", "EnableCycleOffsetPosition",
                               "EnableCycleOffsetRotation", "MotionWeight") if k in js}
    if "FrameDuration" not in meta or meta["FrameDuration"] <= 0:
        raise ValueError(f"{path} FrameDuration 缺失或非法")
    return frames, meta


def save_amp_txt(path: Path, frames: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'"LoopMode": "{meta.get("LoopMode", "Wrap")}",\n')
        f.write(f'"FrameDuration": {float(meta["FrameDuration"]):.6f},\n')
        f.write(f'"EnableCycleOffsetPosition": {str(meta.get("EnableCycleOffsetPosition", True)).lower()},\n')
        f.write(f'"EnableCycleOffsetRotation": {str(meta.get("EnableCycleOffsetRotation", True)).lower()},\n')
        f.write(f'"MotionWeight": {float(meta.get("MotionWeight", 0.5)):.6f},\n\n')
        f.write('"Frames":\n[\n')
        for i, row in enumerate(frames):
            tail = "]," if i < len(frames) - 1 else "]"
            f.write("  [" + ", ".join(f"{v:.6f}" for v in row) + tail + "\n")
        f.write("]\n}")


# --- 重定向核心 ---

def build_robot(urdf_path: str, tag: str) -> dict:
    chain = parse_urdf(urdf_path)
    names = {j["name"] for j in chain["joints"] if j["type"] == "revolute"}
    missing = [n for n in AMP_JOINT_NAMES(tag) if n not in names]
    if missing:
        raise KeyError(f"{tag} URDF 缺少 AMP 关节: {missing}")
    robot = {"chain": chain, "name2j": {j["name"]: j for j in chain["joints"]}}
    # 腿子链: 目标 6-DoF (含 ankle_roll), 源取 5-DoF (ankle_roll 在 ankle_pitch 下游, 不影响踝点)
    n_leg = 6 if tag == "tgt" else 5
    robot["leg_sc"] = {s: subchain(chain, [N(tag, k, s) for k in LEG_KEYS[:n_leg]]) for s in ("l", "r")}
    return robot


def leg_lengths(chain: dict, tag: str, s: str) -> float:
    """零位构型下大腿+小腿长度 (FK 常值原点间距, hip_pitch -> knee -> ankle_pitch)。"""
    names = [N(tag, k, s) for k in LEG_KEYS[:5]]
    pos, _ = fk_batch(subchain(chain, names), np.zeros((1, len(names))))
    hip = pos[N(tag, "l_hip_pitch", s)]
    knee = pos[N(tag, "l_knee", s)]
    ank = pos[N(tag, "l_ankle_pitch", s)]
    return float(np.linalg.norm(knee - hip) + np.linalg.norm(ank - knee))


def solve_leg_ik(tgt: dict, s: str, keys: dict, q0: np.ndarray, n_leg: int,
                 iters: int, tol: float, tag: str = "tgt") -> tuple[np.ndarray, np.ndarray]:
    """目标腿 DLS IK: 拟合缩放踝位置 + 大腿方向, 冗余自由度 (ankle_roll) 正则回初值。
    返回 (q (N,n_leg), 踝位置残差 (N,3))。"""
    sc = tgt["leg_sc"][s]
    hip_lnk, knee_lnk = N(tag, "l_hip_pitch", s), N(tag, "l_knee", s)
    ank_lnk = N(tag, "l_ankle_pitch", s) if n_leg == 5 else N(tag, "j_ankle_roll", s).replace("_joint", "_link")
    by_qidx = sorted(sc["joints"], key=lambda j: j["qidx"])  # q 向量按语义列序
    lo = np.array([by_qidx[i]["lower"] for i in range(n_leg)])
    hi = np.array([by_qidx[i]["upper"] for i in range(n_leg)])
    k = keys[s]

    def resid(q):
        pos, _ = fk_batch(sc, q)
        p_hip, p_knee, p_ank = pos[hip_lnk], pos[knee_lnk], pos[ank_lnk]
        r = np.concatenate([p_ank - (p_hip + k["ank_rel"]),
                            (unit(p_knee - p_hip) - k["thigh"]) * DIR_W,
                            (q - q0) * Q_W], axis=1)
        return r, pos

    q, r, _ = dls_ik(resid, q0, lo, hi, iters, tol)
    return q, r[:, :3]


def solve_leg_ik_hot(tgt: dict, s: str, keys: dict, q0: np.ndarray, n_leg: int,
                     iters: int, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """整条轨迹腿 IK: 先同源映射初值向量化求解, 再对残差大的帧逐帧热启动修复
    (深屈位姿下同源映射初值会落入错误局部极小, 前一帧解是可靠初值)。"""
    q, r = solve_leg_ik(tgt, s, keys, q0, n_leg, iters, tol)
    err = np.linalg.norm(r, axis=1)
    bad_thr = max(5 * tol, 1e-2)
    bad = np.where(err > bad_thr)[0]
    n_repair = 0
    for i in bad:
        q_prev = q[i - 1:i] if i > 0 else q0[i:i + 1]
        keys_i = {s: {k: v[i:i + 1] for k, v in keys[s].items()}}
        qi, ri = solve_leg_ik(tgt, s, keys_i, q_prev, n_leg, iters * 3, tol)
        q[i] = qi[0]
        err[i] = np.linalg.norm(ri[0])
        n_repair += 1
    if n_repair:
        print(f"  leg {s}: 热启动修复 {n_repair}/{len(err)} 帧 (残差>{bad_thr:.4f} m)")
    return q, err


def solve_shoulder_ik(tgt: dict, idx_map: dict, q_waist: np.ndarray,
                      sh_pos_s: np.ndarray, hand_s: np.ndarray, arm_scale: float,
                      s: str, n: int) -> np.ndarray:
    """目标肩 1-DoF IK: 拟合缩放手位置。idx_map: {"waist": 0, s: col}。
    G1 肩父链经 waist_roll/pitch, 子链含全部腰关节 (roll/pitch 置 0)。"""
    waist_chain = [N("tgt", "j_waist", "l"), "waist_roll_joint", "waist_pitch_joint"]
    sc_sh = subchain(tgt["chain"], waist_chain + [N("tgt", "j_shoulder", s)])
    i_sh = len(waist_chain)  # 肩在 keep_names 中的语义列 (== 其 qidx)
    j_sh = next(j for j in sc_sh["joints"] if j["qidx"] == i_sh)
    lo = np.array([j_sh["lower"]])
    hi = np.array([j_sh["upper"]])
    lnk = j_sh["child"]

    def resid(q):
        qq = np.zeros((n, len(waist_chain) + 1))
        qq[:, 0] = q_waist
        qq[:, i_sh] = q[:, 0]
        pos, rot = fk_batch(sc_sh, qq)
        hand = pos[lnk] + np.einsum("nij,j->ni", rot[lnk], G1_ARM_VEC)
        # 目标手臂向量 = 源手臂向量缩放后整体前摆 ARM_SWING_FORWARD
        off = np.einsum("ij,nj->ni", ARM_BIAS_ROT, (hand_s - sh_pos_s) * arm_scale)
        return (hand - (pos[lnk] + off)), None

    q, _, _ = dls_ik(resid, np.zeros((n, 1)), lo, hi, SH_ITERS, 1e-4)
    return q[:, 0]


def retarget_expert_file(src: dict, tgt: dict, in_path: Path, out_path: Path,
                         iters: int, tol: float) -> None:
    frames, meta = load_amp_txt(in_path)
    n = frames.shape[0]
    dt = float(meta["FrameDuration"])
    names_src = AMP_JOINT_NAMES("src")
    q_amp = frames[:, :AMP_POS_END]  # AMP 语义序, ankle_roll 为 yrp03 零填充
    if np.isnan(q_amp).any():
        raise ValueError(f"{in_path} 关节角含 NaN")

    # --- 源 FK 关键点 (root = pelvis 单位阵) ---
    q_src = np.zeros((n, len(src["chain"]["joints"])))
    for i, name in enumerate(names_src):
        q_src[:, src["name2j"][name]["qidx"]] = q_amp[:, i]
    spos, srot = fk_batch(src["chain"], q_src)

    def src_key(s: str) -> dict:
        hip = spos[N("src", "l_hip_pitch", s)]
        knee = spos[N("src", "l_knee", s)]
        ank = spos[N("src", "l_ankle_pitch", s)]
        return {"hip": hip, "ank_rel": (ank - hip) * (tgt["scale"][s] / src["scale"][s]),
                "thigh": unit(knee - hip)}

    keys = {s: src_key(s) for s in ("l", "r")}
    sh = {s: spos[N("src", "l_shoulder", s)] for s in ("l", "r")}
    hand = {s: sh[s] + np.einsum("nij,j->ni", srot[N("src", "l_shoulder", s)], SRC_ARM_VEC)
            for s in ("l", "r")}

    # --- 目标腿 6-DoF IK (初值 = 同源映射, ankle_roll=0) ---
    q_leg = {}
    for s in ("l", "r"):
        i_off = 2 + (0 if s == "r" else 6)  # q_amp 中本腿起始列 (shoulder_r, shoulder_l 之后)
        q0 = np.stack([q_amp[:, i_off + i] for i in range(5)] + [np.zeros(n)], axis=1)
        q_leg[s], err = solve_leg_ik_hot(tgt, s, keys, q0, 6, iters, tol)
        print(f"  leg {s}: IK pos err max={err.max():.4f} m mean={err.mean():.4f} m "
              f"(收敛阈值 {tol:g} m)")

    # --- 腰 yaw 直接映射 + 肩 1-DoF IK ---
    q_waist = q_amp[:, names_src.index(N("src", "j_waist", "l"))]
    arm_scale = np.linalg.norm(G1_ARM_VEC) / np.linalg.norm(SRC_ARM_VEC)
    i_l = names_src.index(N("src", "j_shoulder", "l"))
    i_r = names_src.index(N("src", "j_shoulder", "r"))
    # 肩 IK 初值 = 源肩角
    q_sh = {}
    for s, i_src in (("l", i_l), ("r", i_r)):
        q_sh[s] = solve_shoulder_ik(tgt, None, q_waist, sh[s], hand[s], arm_scale, s, n)
        print(f"  shoulder {s}: |dq| vs src mean={np.abs(q_sh[s] - q_amp[:, i_src]).mean():.4f} rad")

    # --- 组装目标 AMP 序关节角 ---
    names_tgt = AMP_JOINT_NAMES("tgt")
    q_tgt = np.zeros((n, len(tgt["chain"]["joints"])))
    for s in ("l", "r"):
        for i, k in enumerate(LEG_KEYS):
            q_tgt[:, tgt["name2j"][N("tgt", k, s)]["qidx"]] = q_leg[s][:, i]
    q_tgt[:, tgt["name2j"][N("tgt", "j_waist", "l")]["qidx"]] = q_waist
    q_tgt[:, tgt["name2j"][N("tgt", "j_shoulder", "l")]["qidx"]] = q_sh["l"]
    q_tgt[:, tgt["name2j"][N("tgt", "j_shoulder", "r")]["qidx"]] = q_sh["r"]

    # --- 目标 FK 重算 ee (与 unitree_g1 env get_amp_obs_for_expert_trans 口径一致) ---
    tpos, trot = fk_batch(tgt["chain"], q_tgt)
    ee_lh = tpos[N("tgt", "l_shoulder", "l")] + np.einsum(
        "nij,j->ni", trot[N("tgt", "l_shoulder", "l")], G1_ARM_VEC)
    ee_rh = tpos[N("tgt", "l_shoulder", "r")] + np.einsum(
        "nij,j->ni", trot[N("tgt", "l_shoulder", "r")], G1_ARM_VEC)
    ee_lf = tpos[N("tgt", "j_ankle_roll", "l").replace("_joint", "_link")]
    ee_rf = tpos[N("tgt", "j_ankle_roll", "r").replace("_joint", "_link")]

    # --- 关节速度 + 42 维帧 ---
    q_out = np.stack([q_tgt[:, tgt["name2j"][nm]["qidx"]] for nm in names_tgt], axis=1)
    dq_out = np.gradient(q_out, dt, axis=0, edge_order=2)
    out = np.concatenate([q_out, dq_out, ee_lh, ee_rh, ee_lf, ee_rf], axis=1)
    if out.shape != (n, AMP_DIM):
        raise AssertionError(f"输出维度异常: {out.shape}")
    if np.isnan(out).any():
        raise ValueError("输出含 NaN")

    save_amp_txt(out_path, out, meta)
    limit_viol = int(sum(np.sum(q_out[:, i] < tgt["name2j"][nm]["lower"] - 1e-9) +
                         np.sum(q_out[:, i] > tgt["name2j"][nm]["upper"] + 1e-9)
                         for i, nm in enumerate(names_tgt)))
    print(f"  -> {out_path}  frames={n} fps={1/dt:.1f}  关节限位越界帧数={limit_viol}")
    print(f"  足端 z 范围: L [{ee_lf[:, 2].min():.3f},{ee_lf[:, 2].max():.3f}] "
          f"R [{ee_rf[:, 2].min():.3f},{ee_rf[:, 2].max():.3f}] (root 系)")


# --- motion visualization (38 维 eric-zhao schema) 重定向 ---
# [0:3] root pos, [3:6] euler, [6:11] L leg pos, [11:16] R leg pos, [16] waist,
# [17] L arm, [18] R arm, [19:22] lin vel, [22:25] ang vel, [25:30] L leg vel,
# [30:35] R leg vel, [35] waist vel, [36] L arm vel, [37] R arm vel
VIS_DIM = 38


def retarget_vis_file(src: dict, tgt: dict, in_path: Path, out_path: Path,
                      iters: int, tol: float) -> None:
    frames, meta = load_amp_txt(in_path, VIS_DIM)
    if frames.shape[1] != VIS_DIM:
        raise ValueError(f"{in_path} 是 {frames.shape[1]} 列, 仅支持 {VIS_DIM} 列 eric-zhao schema")
    n = frames.shape[0]
    dt = float(meta["FrameDuration"])
    height_ratio = TGT_STAND_HEIGHT / SRC_STAND_HEIGHT

    out = frames.copy()
    # root z 与线速度按站立高度比缩放
    out[:, 2] = frames[:, 2] * height_ratio
    out[:, 19:22] = frames[:, 19:22] * height_ratio

    # --- 源 FK (root = pelvis 单位阵; vis 腿只有 5-DoF, ankle_roll 缺省 0) ---
    names_src = AMP_JOINT_NAMES("src")
    q_src = np.zeros((n, len(src["chain"]["joints"])))
    for s, sl in (("l", slice(6, 11)), ("r", slice(11, 16))):
        for i, k in enumerate(LEG_KEYS[:5]):
            q_src[:, src["name2j"][N("src", k, s)]["qidx"]] = frames[:, sl][:, i]
    q_src[:, src["name2j"][N("src", "j_waist", "l")]["qidx"]] = frames[:, 16]
    q_src[:, src["name2j"][N("src", "j_shoulder", "l")]["qidx"]] = frames[:, 17]
    q_src[:, src["name2j"][N("src", "j_shoulder", "r")]["qidx"]] = frames[:, 18]
    spos, srot = fk_batch(src["chain"], q_src)

    def src_key(s: str, sl: slice) -> dict:
        hip = spos[N("src", "l_hip_pitch", s)]
        knee = spos[N("src", "l_knee", s)]
        ank = spos[N("src", "l_ankle_pitch", s)]
        return {"hip": hip, "ank_rel": (ank - hip) * (tgt["scale"][s] / src["scale"][s]),
                "thigh": unit(knee - hip)}

    keys = {s: src_key(s, sl) for s, sl in (("l", slice(6, 11)), ("r", slice(11, 16)))}
    sh = {s: spos[N("src", "l_shoulder", s)] for s in ("l", "r")}
    hand = {s: sh[s] + np.einsum("nij,j->ni", srot[N("src", "l_shoulder", s)], SRC_ARM_VEC)
            for s in ("l", "r")}

    # --- 目标腿 5-DoF IK (G1 子链取 yaw/roll/pitch/knee/ankle_pitch), 初值同源映射 ---
    q_leg, q_leg_sc = {}, {s: subchain(tgt["chain"], [N("tgt", k, s) for k in LEG_KEYS[:5]])
                           for s in ("l", "r")}
    saved = tgt["leg_sc"]
    tgt["leg_sc"] = q_leg_sc  # solve_leg_ik 用 5-DoF 子链
    for s, sl in (("l", slice(6, 11)), ("r", slice(11, 16))):
        q0 = frames[:, sl]
        q_leg[s], err = solve_leg_ik_hot(tgt, s, keys, q0, 5, iters, tol)
        print(f"  vis leg {s}: IK pos err max={err.max():.4f} m mean={err.mean():.4f} m")
    tgt["leg_sc"] = saved

    # --- 腰 yaw 映射 + 肩 1-DoF IK ---
    q_waist = frames[:, 16]
    arm_scale = np.linalg.norm(G1_ARM_VEC) / np.linalg.norm(SRC_ARM_VEC)
    q_sh = {s: solve_shoulder_ik(tgt, None, q_waist, sh[s], hand[s], arm_scale, s, n)
            for s in ("l", "r")}

    # --- 写回关节块 + 速度重算 ---
    out[:, 6:11] = q_leg["l"]
    out[:, 11:16] = q_leg["r"]
    out[:, 16] = q_waist
    out[:, 17] = q_sh["l"]
    out[:, 18] = q_sh["r"]
    dq = np.gradient(out[:, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]], dt, axis=0, edge_order=2)
    out[:, 25:38] = dq
    if np.isnan(out).any():
        raise ValueError("输出含 NaN")

    save_amp_txt(out_path, out, meta)
    print(f"  -> {out_path}  frames={n} fps={1/dt:.1f}  (root z 缩放比 {height_ratio:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="yrp03-v2 → unitree_g1 AMP expert / vis FK/IK 重定向")
    parser.add_argument("--src_expert_dir", type=str, default=SRC_EXPERT_DIR, help="yrp03 expert 目录")
    parser.add_argument("--out_expert_dir", type=str, default=OUT_EXPERT_DIR, help="G1 expert 输出目录")
    parser.add_argument("--src_vis_dir", type=str, default=SRC_VIS_DIR, help="yrp03 vis 目录")
    parser.add_argument("--out_vis_dir", type=str, default=OUT_VIS_DIR, help="G1 vis 输出目录")
    parser.add_argument("--expert_files", type=str, nargs="*", default=None,
                        help="相对 src_expert_dir 的 expert 文件 (默认 unitree_g1 cfg 引用的 4 个)")
    parser.add_argument("--vis_files", type=str, nargs="*", default=None,
                        help="相对 src_vis_dir 的 vis 文件 (默认 run_14mps_play_newultra.txt)")
    parser.add_argument("--ik_iters", type=int, default=IK_ITERS, help="腿 IK 最大迭代数")
    parser.add_argument("--ik_tol", type=float, default=IK_TOL, help="腿 IK 收敛阈值 (m)")
    args = parser.parse_args()

    sys.setrecursionlimit(10000)
    src = build_robot(YRP03_URDF, "src")
    tgt = build_robot(G1_URDF, "tgt")
    tgt["scale"], src["scale"] = {}, {}
    for s in ("l", "r"):
        src["scale"][s] = leg_lengths(src["chain"], "src", s)
        tgt["scale"][s] = leg_lengths(tgt["chain"], "tgt", s)
    print(f"[scale] 腿长比 L={tgt['scale']['l']/src['scale']['l']:.4f} "
          f"R={tgt['scale']['r']/src['scale']['r']:.4f}  "
          f"身高比 {TGT_STAND_HEIGHT/SRC_STAND_HEIGHT:.4f}")

    for rel in (args.expert_files if args.expert_files is not None else DEFAULT_EXPERT_FILES):
        in_path = Path(args.src_expert_dir) / rel
        out_path = Path(args.out_expert_dir) / rel
        print(f"[retarget expert] {in_path} -> {out_path}")
        retarget_expert_file(src, tgt, in_path, out_path, args.ik_iters, args.ik_tol)

    for rel in (args.vis_files if args.vis_files is not None else DEFAULT_VIS_FILES):
        in_path = Path(args.src_vis_dir) / rel
        out_path = Path(args.out_vis_dir) / rel
        print(f"[retarget vis] {in_path} -> {out_path}")
        retarget_vis_file(src, tgt, in_path, out_path, args.ik_iters, args.ik_tol)

    print("全部完成。")


if __name__ == "__main__":
    main()
