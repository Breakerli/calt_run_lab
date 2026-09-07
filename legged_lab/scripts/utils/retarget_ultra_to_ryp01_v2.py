"""ultra_game_yaw_0617 → 01RYP-v2 AMP expert 的 FK/IK 重定向（方案 B）。

把 ultra_game_yaw_0617 的 AMP expert txt（38 维 DeepMimic JSON，root 系）重定向为
01RYP-v2 的同格式 expert，替换 ryp01_v2 端逐字节复制的临时烟测数据。

核心逻辑:
1. 通用 URDF 解析，构建运动链，向量化 FK（整条轨迹一次批处理）。
2. 源机器人 FK 提取关键点: 髋(hip_pitch link) / 膝 / 踝 / 肩 / 手(root 系)。
3. 目标腿 5-DoF 阻尼最小二乘(DLS)数值 IK: 拟合按腿长比缩放后的踝位置 + 大腿方向,
   初值用同源映射关节角并逐帧热启动; 肩 1-DoF IK 拟合手位置; 腰 yaw 直接映射。
4. np.gradient 求关节速度; 目标 FK 重算 ee(手 = 肩位置 + 肩姿态旋转本地臂向量,
   与 ryp01_motion_run_env.get_amp_obs_for_expert_trans 口径一致), 写出同格式 38 维 txt。

运行示例:
  python legged_lab/scripts/utils/retarget_ultra_to_ryp01_v2.py
  python legged_lab/scripts/utils/retarget_ultra_to_ryp01_v2.py \
      --files stand_ultra_yaw2.txt --ik_iters 60
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# --- Constants & Config ---

ULTRA_URDF = "legged_lab/assets/ultra_GameYaw_0617/urdf/ultra_game_yaw_0617.urdf"
RYP_URDF = "legged_lab/assets/ryp01-v2/urdf/01RYP-v2.urdf"
DEFAULT_INPUT_DIR = "legged_lab/envs/ultra_game_yaw/datasets/motion_amp_expert"
DEFAULT_OUTPUT_DIR = "legged_lab/envs/ryp01_v2/datasets/motion_amp_expert/retargeted"
DEFAULT_FILES = [
    "stand_ultra_yaw2.txt",
    "walk_ultra_yaw2.txt",
    "run_17200_9mps.txt",
    "collected_20260602_160257/env000_vxp15p00_vyp0p00_wp0p00.txt",
]

# AMP expert 38 维帧的关节排布(两台机器人一致, ryp01_v2 端见 ryp01_motion_run_env.AMP_JOINT_NAMES):
# [0:13]  pos: shoulder_r, shoulder_l, 右腿(yaw,roll,pitch,knee,ankle), 左腿(...), waist_yaw
# [13:26] vel 同序; [26:38] ee(root 系): left_hand, right_hand, left_foot, right_foot
AMP_JOINT_NAMES = (
    "shoulder_pitch_r_joint", "shoulder_pitch_l_joint",
    "hip_yaw_r_joint", "hip_roll_r_joint", "hip_pitch_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint",
    "hip_yaw_l_joint", "hip_roll_l_joint", "hip_pitch_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint",
    "waist_yaw_joint",
)
AMP_POS_END = 13
AMP_VEL_END = 26
AMP_DIM = 38

# 腿链(拓扑序): 源 ultra 是 yaw→roll→pitch, 目标 ryp01_v2 是 roll→yaw→pitch
LEG_SEGS_SRC = ("hip_yaw_{s}_joint", "hip_roll_{s}_joint", "hip_pitch_{s}_joint",
                "knee_pitch_{s}_joint", "ankle_pitch_{s}_joint")
LEG_SEGS_TGT = ("hip_roll_{s}_joint", "hip_yaw_{s}_joint", "hip_pitch_{s}_joint",
                "knee_pitch_{s}_joint", "ankle_pitch_{s}_joint")

ULTRA_ARM_LEN = 0.4    # ultra env 手 ee 本地臂向量 [0,0,-0.4]
RYP_ARM_LEN = 0.464    # ryp01_v2 env 手 ee 本地臂向量 [0,0,-0.464]

FD_H = 1e-5        # 数值雅可比步长 (rad)
DLS_LAM = 1e-2     # DLS 阻尼
MAX_STEP = 0.3     # 单迭代最大关节增量 (rad)
DIR_W = 0.1        # 大腿方向残差权重 (等效米)
Q_W = 0.02         # 关节正则化权重: 约束冗余自由度停留在同源映射初值附近 (等效米/rad)
IK_ITERS = 60
IK_TOL = 2e-3      # 踝位置残差收敛阈值 (m)
SH_ITERS = 10


# --- URDF 解析与向量化 FK ---

def parse_urdf(urdf_path: str) -> dict:
    """解析 URDF 为运动链: 拓扑序关节列表 + 根 link。仅支持单树 revolute/fixed。"""
    path = Path(urdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"URDF 不存在: {urdf_path}")
    root = ET.parse(path).getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    joints = {}
    for j in root.iter(f"{ns}joint"):
        name = j.get("name")
        origin = j.find(f"{ns}origin")
        axis = j.find(f"{ns}axis")
        limit = j.find(f"{ns}limit")
        joints[name] = {
            "name": name,
            "type": j.get("type"),
            "parent": j.find(f"{ns}parent").get("link"),
            "child": j.find(f"{ns}child").get("link"),
            "xyz": np.array((origin.get("xyz") or "0 0 0").split(), dtype=np.float64),
            "rpy": rpy_mat(np.array((origin.get("rpy") or "0 0 0").split(), dtype=np.float64)),
            "axis": np.array((axis.get("xyz") if axis is not None else "1 0 0").split(), dtype=np.float64),
            "lower": float(limit.get("lower")) if limit is not None else -np.inf,
            "upper": float(limit.get("upper")) if limit is not None else np.inf,
        }
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
    """取子链(仅保留指定关节, 保持拓扑序, 重排 qidx)。"""
    name_set, joints = set(keep_names), []
    for j in chain["joints"]:
        if j["name"] in name_set:
            joints.append(dict(j))
    if len(joints) != len(keep_names):
        missing = name_set - {j["name"] for j in joints}
        raise KeyError(f"子链关节在 URDF 中缺失: {missing}")
    for i, j in enumerate(joints):
        j["qidx"] = i
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

def load_amp_txt(path: Path) -> tuple[np.ndarray, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"expert 文件不存在: {path}")
    with path.open(encoding="utf-8") as f:
        js = json.load(f)
    frames = np.asarray(js["Frames"], dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] < AMP_DIM:
        raise ValueError(f"{path} 帧维度异常: {frames.shape}, 期望 (N,>={AMP_DIM})")
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
    missing = [n for n in AMP_JOINT_NAMES if n not in names]
    if missing:
        raise KeyError(f"{tag} URDF 缺少 AMP 关节: {missing}")
    robot = {"chain": chain, "name2j": {j["name"]: j for j in chain["joints"]}}
    robot["leg_sc"] = {s: subchain(chain, [t.format(s=s) for t in LEG_SEGS_TGT]) for s in ("l", "r")}
    return robot


def leg_lengths(chain: dict, s: str, segs: tuple[str, ...]) -> float:
    """零位构型下大腿+小腿长度 (FK 常值原点间距)。"""
    names = [t.format(s=s) for t in segs]
    pos, _ = fk_batch(subchain(chain, names), np.zeros((1, len(names))))
    hip, knee, ank = (pos[nm.replace("_joint", "_link")] for nm in names[2:5])
    return float(np.linalg.norm(knee - hip) + np.linalg.norm(ank - knee))


def retarget_file(src: dict, tgt: dict, in_path: Path, out_path: Path,
                  iters: int, tol: float) -> None:
    frames, meta = load_amp_txt(in_path)
    n = frames.shape[0]
    dt = float(meta["FrameDuration"])
    q_amp = frames[:, :AMP_POS_END]  # AMP 排布, 两台机器人关节名一致
    if np.isnan(q_amp).any():
        raise ValueError(f"{in_path} 关节角含 NaN")

    # --- 源 FK 关键点 (root = base_link 单位阵) ---
    q_src = np.zeros((n, len(src["chain"]["joints"])))
    for i, name in enumerate(AMP_JOINT_NAMES):
        q_src[:, src["name2j"][name]["qidx"]] = q_amp[:, i]
    spos, srot = fk_batch(src["chain"], q_src)

    def src_key(s: str) -> dict:
        seg = [t.format(s=s) for t in LEG_SEGS_SRC]
        hip = spos[seg[2].replace("_joint", "_link")]
        knee = spos[seg[3].replace("_joint", "_link")]
        ank = spos[seg[4].replace("_joint", "_link")]
        return {"hip": hip, "ank_rel": (ank - hip) * (tgt["scale"][s] / src["scale"][s]),
                "thigh": unit(knee - hip)}

    keys = {s: src_key(s) for s in ("l", "r")}
    sh_l, sh_r = spos["shoulder_pitch_l_link"], spos["shoulder_pitch_r_link"]
    hand_l = sh_l + np.einsum("nij,j->ni", srot["shoulder_pitch_l_link"], np.array([0, 0, -ULTRA_ARM_LEN]))
    hand_r = sh_r + np.einsum("nij,j->ni", srot["shoulder_pitch_r_link"], np.array([0, 0, -ULTRA_ARM_LEN]))

    # --- 目标腿 IK ---
    def solve_leg(s: str) -> np.ndarray:
        sc, names = tgt["leg_sc"][s], [t.format(s=s) for t in LEG_SEGS_TGT]
        hip_lnk, knee_lnk, ank_lnk = (nm.replace("_joint", "_link") for nm in names[2:5])
        lo = np.array([sc["joints"][i]["lower"] for i in range(5)])
        hi = np.array([sc["joints"][i]["upper"] for i in range(5)])
        k = keys[s]
        # 初值: 同源映射 (yaw,roll,pitch,knee,ankle) -> (roll,yaw,pitch,knee,ankle)
        q0 = np.stack([q_amp[:, AMP_JOINT_NAMES.index(f"hip_roll_{s}_joint")],
                       q_amp[:, AMP_JOINT_NAMES.index(f"hip_yaw_{s}_joint")],
                       q_amp[:, AMP_JOINT_NAMES.index(f"hip_pitch_{s}_joint")],
                       q_amp[:, AMP_JOINT_NAMES.index(f"knee_pitch_{s}_joint")],
                       q_amp[:, AMP_JOINT_NAMES.index(f"ankle_pitch_{s}_joint")]], axis=1)

        def resid(q):
            pos, _ = fk_batch(sc, q)
            p_hip, p_knee, p_ank = pos[hip_lnk], pos[knee_lnk], pos[ank_lnk]
            r = np.concatenate([p_ank - (p_hip + k["ank_rel"]),
                                (unit(p_knee - p_hip) - k["thigh"]) * DIR_W,
                                (q - q0) * Q_W], axis=1)
            return r, pos

        q, r, fin = dls_ik(resid, q0, lo, hi, iters, tol)
        err = np.linalg.norm(r[:, :3], axis=1)
        print(f"  leg {s}: IK pos err max={err.max():.4f} m mean={err.mean():.4f} m "
              f"(iters 收敛阈值 {tol:g} m)")
        return q

    q_leg = {s: solve_leg(s) for s in ("l", "r")}

    # --- 腰 yaw 直接映射 + 肩 1-DoF IK ---
    q_waist = q_amp[:, AMP_JOINT_NAMES.index("waist_yaw_joint")]
    sc_sh = subchain(tgt["chain"], ["waist_yaw_joint", "shoulder_pitch_l_joint", "shoulder_pitch_r_joint"])

    def solve_sh(idx: int, sh_pos_s: np.ndarray, hand_s: np.ndarray) -> np.ndarray:
        arm_scale = RYP_ARM_LEN / ULTRA_ARM_LEN
        lo = np.array([sc_sh["joints"][idx]["lower"]])
        hi = np.array([sc_sh["joints"][idx]["upper"]])
        q0 = q_amp[:, [AMP_JOINT_NAMES.index(
            "shoulder_pitch_l_joint" if idx == 1 else "shoulder_pitch_r_joint")]]
        lnk = sc_sh["joints"][idx]["child"]

        def resid(q):
            qq = np.zeros((n, 3))
            qq[:, 0] = q_waist
            qq[:, idx] = q[:, 0]
            pos, rot = fk_batch(sc_sh, qq)
            hand = pos[lnk] + np.einsum("nij,j->ni", rot[lnk], np.array([0, 0, -RYP_ARM_LEN]))
            return (hand - (pos[lnk] + (hand_s - sh_pos_s) * arm_scale)), None

        q, r, _ = dls_ik(resid, q0, lo, hi, SH_ITERS, 1e-4)
        return q[:, 0]

    q_sh_l = solve_sh(1, sh_l, hand_l)
    q_sh_r = solve_sh(2, sh_r, hand_r)

    # --- 组装目标 AMP 序关节角 ---
    q_tgt = np.zeros((n, len(tgt["chain"]["joints"])))
    for s in ("l", "r"):
        for i, nm in enumerate(LEG_SEGS_TGT):
            q_tgt[:, tgt["name2j"][nm.format(s=s)]["qidx"]] = q_leg[s][:, i]
    q_tgt[:, tgt["name2j"]["waist_yaw_joint"]["qidx"]] = q_waist
    q_tgt[:, tgt["name2j"]["shoulder_pitch_l_joint"]["qidx"]] = q_sh_l
    q_tgt[:, tgt["name2j"]["shoulder_pitch_r_joint"]["qidx"]] = q_sh_r

    # --- 目标 FK 重算 ee (与 ryp01 env get_amp_obs_for_expert_trans 口径一致) ---
    tpos, trot = fk_batch(tgt["chain"], q_tgt)
    ee_lh = tpos["shoulder_pitch_l_link"] + np.einsum("nij,j->ni", trot["shoulder_pitch_l_link"], np.array([0, 0, -RYP_ARM_LEN]))
    ee_rh = tpos["shoulder_pitch_r_link"] + np.einsum("nij,j->ni", trot["shoulder_pitch_r_link"], np.array([0, 0, -RYP_ARM_LEN]))
    ee_lf, ee_rf = tpos["ankle_pitch_l_link"], tpos["ankle_pitch_r_link"]

    # --- 关节速度 + 38 维帧 ---
    q_out = np.stack([q_tgt[:, tgt["name2j"][nm]["qidx"]] for nm in AMP_JOINT_NAMES], axis=1)
    dq_out = np.gradient(q_out, dt, axis=0, edge_order=2)
    out = np.concatenate([q_out, dq_out, ee_lh, ee_rh, ee_lf, ee_rf], axis=1)
    if out.shape != (n, AMP_DIM):
        raise AssertionError(f"输出维度异常: {out.shape}")
    if np.isnan(out).any():
        raise ValueError("输出含 NaN")

    save_amp_txt(out_path, out, meta)
    limit_viol = int(sum(np.sum(q_out[:, i] < tgt["name2j"][nm]["lower"] - 1e-9) +
                         np.sum(q_out[:, i] > tgt["name2j"][nm]["upper"] + 1e-9)
                         for i, nm in enumerate(AMP_JOINT_NAMES)))
    print(f"  -> {out_path}  frames={n} fps={1/dt:.1f}  关节限位越界帧数={limit_viol}")
    print(f"  足端 z 范围: L [{ee_lf[:,2].min():.3f},{ee_lf[:,2].max():.3f}] "
          f"R [{ee_rf[:,2].min():.3f},{ee_rf[:,2].max():.3f}] (root 系)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ultra_game_yaw_0617 → 01RYP-v2 AMP expert FK/IK 重定向")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR, help="ultra expert 目录")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="重定向输出目录")
    parser.add_argument("--source_urdf", type=str, default=ULTRA_URDF, help="源机器人 URDF")
    parser.add_argument("--target_urdf", type=str, default=RYP_URDF, help="目标机器人 URDF")
    parser.add_argument("--files", type=str, nargs="*", default=None, help="相对 input_dir 的文件列表(默认 ryp01_v2 cfg 引用的 4 个)")
    parser.add_argument("--ik_iters", type=int, default=IK_ITERS, help="腿 IK 最大迭代数")
    parser.add_argument("--ik_tol", type=float, default=IK_TOL, help="腿 IK 收敛阈值 (m)")
    args = parser.parse_args()

    files = args.files or DEFAULT_FILES
    src = build_robot(args.source_urdf, "src")
    tgt = build_robot(args.target_urdf, "tgt")
    tgt["scale"], src["scale"] = {}, {}
    for s in ("l", "r"):
        src["scale"][s] = leg_lengths(src["chain"], s, LEG_SEGS_SRC)
        tgt["scale"][s] = leg_lengths(tgt["chain"], s, LEG_SEGS_TGT)
    print(f"[scale] 腿长比 L={tgt['scale']['l']/src['scale']['l']:.4f} R={tgt['scale']['r']/src['scale']['r']:.4f}")

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel in files:
        in_path = in_dir / rel
        out_path = out_dir / in_path.relative_to(in_dir).with_name(in_path.stem + "_ryp01_v2.txt")
        print(f"[retarget] {in_path} -> {out_path}")
        retarget_file(src, tgt, in_path, out_path, args.ik_iters, args.ik_tol)
    print("全部完成。")


if __name__ == "__main__":
    main()
