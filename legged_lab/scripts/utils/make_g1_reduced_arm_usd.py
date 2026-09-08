"""从 g1_29dof_rev_1_0.usd 生成减臂自由度的 G1 USD (默认锁肩 yaw/肘/腕, 臂仅留 pitch+roll)。

在 wrapper 根层打 override, 将指定 arm 关节 prim 的 typeName 改为 PhysicsFixedJoint
(PhysX 会把固定关节两侧连杆合并为一个刚体, 关节不再产生自由度); 几何/物理/传感器
仍共享 configuration/g1_29dof_rev_1_0_* 各层, 新文件只存 override, 体积小。

运行示例 (需 isaacsim python.sh 提供 pxr 环境):
  /home/robot02/isaacsim/python.sh --headless --enable-camera \
      legged_lab/scripts/utils/make_g1_reduced_arm_usd.py
  /home/robot02/isaacsim/python.sh ... --lock shoulder_yaw elbow wrist_roll wrist_pitch wrist_yaw
  /home/robot02/isaacsim/python.sh ... --dst .../usd/g1_19dof_rev_1_0.usd
"""

import argparse
from pathlib import Path

USD_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "unitree_g1" / "usd"
SRC = USD_DIR / "g1_29dof_rev_1_0.usd"
DST = USD_DIR / "g1_19dof_rev_1_0.usd"
# 默认锁死的关节 (每侧 prefix left_/right_): 肩只剩 pitch+roll 两个自由度
DEFAULT_LOCK = ("shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")


def parse_args():
    parser = argparse.ArgumentParser(description="生成减臂自由度的 G1 USD (肩 pitch+roll)")
    parser.add_argument("--src", type=str, default=str(SRC), help="源 USD (wrapper)")
    parser.add_argument("--dst", type=str, default=str(DST), help="输出 USD (wrapper, 只含 override)")
    parser.add_argument("--lock", type=str, nargs="*", default=list(DEFAULT_LOCK),
                        help="要锁死为 fixed 的关节名后缀 (不含 left_/right_ 前缀)")
    return parser.parse_args()


def main():
    from pxr import Usd, UsdPhysics

    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if not src.is_file():
        raise FileNotFoundError(f"源 USD 不存在: {src}")
    if not hasattr(UsdPhysics, "FixedJoint"):
        raise RuntimeError("当前 pxr 版本无 UsdPhysics.FixedJoint schema")

    stage = Usd.Stage.Open(str(src))
    stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))

    lock_names = {f"{p}_{j}_joint" for p in ("left", "right") for j in args.lock}
    n_rev, n_fixed = 0, 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        n_rev += 1
        if prim.GetName() not in lock_names:
            continue
        # override typeName 到根层; PhysX 将两侧连杆合并, 关节不再有自由度
        prim.SetTypeName("PhysicsFixedJoint")
        n_fixed += 1

    missing = lock_names - {p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)}
    if missing:
        raise KeyError(f"以下关节在 USD 中未找到或未成功转换: {sorted(missing)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(dst))
    print(f"[OK] revolute 关节总数={n_rev}, 已锁死={n_fixed}, 保留自由度={n_rev - n_fixed}")
    print(f"[OK] {src.name} -> {dst}")


if __name__ == "__main__":
    main()
