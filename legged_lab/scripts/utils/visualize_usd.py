"""在 Isaac Sim 中加载并可视化一个 USD 模型 (默认天工2plus)。

新建空场景, 添加地面/光照, 将目标 USD 作为 reference 挂到 /World/robot,
自动根据模型包围盒把相机对准模型, 然后进入交互式仿真循环 (GUI 模式)。

运行示例:
  /home/robot02/isaacsim/python.sh legged_lab/scripts/utils/visualize_usd.py
  /home/robot02/isaacsim/python.sh legged_lab/scripts/utils/visualize_usd.py --headless  # 仅加载自检
  /home/robot02/isaacsim/python.sh legged_lab/scripts/utils/visualize_usd.py --usd /path/to/xxx.usd --eye 2.5,2.5,1.5
"""

import argparse

DEFAULT_USD = "legged_lab/assets/tiangong2plus/usd/tiangong2_plus_urdf.usd"


def parse_args():
    parser = argparse.ArgumentParser(description="在 Isaac Sim 中可视化 USD 模型")
    parser.add_argument("--usd", type=str, default=DEFAULT_USD, help="要可视化的 USD 文件路径")
    parser.add_argument("--prim_path", type=str, default="/World/robot", help="USD 在场景中的挂载 prim 路径")
    parser.add_argument("--eye", type=str, default=None, help="相机位置 'x,y,z' (默认按包围盒自动计算)")
    parser.add_argument("--lookat", type=str, default=None, help="相机注视点 'x,y,z' (默认包围盒中心)")
    parser.add_argument("--ground", action="store_true", default=True, help="添加地面平面 (默认开)")
    parser.add_argument("--no_ground", dest="ground", action="store_false", help="不添加地面平面")
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.02,
        help="地面间隙 (m): 按包围盒把模型抬到地板上方该高度; 设为负数关闭 (默认 0.02)",
    )
    parser.add_argument("--headless", action="store_true", help="无头加载自检, 不进入交互循环")
    return parser.parse_args()


def _parse_xyz(value, label):
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{label} 必须是 'x,y,z'")
    import numpy as np

    return np.array([float(p) for p in parts], dtype=float)


def main():
    args = parse_args()
    from pathlib import Path

    usd_path = Path(args.usd)
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD 文件不存在: {usd_path.resolve()}")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    import numpy as np
    import omni.usd
    from isaaclab.sim import SimulationContext
    from pxr import Usd, UsdGeom, Gf

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    if args.ground:
        sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    # 将目标 USD 以 reference 形式挂到场景
    robot_cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path.resolve()))
    robot_cfg.func(args.prim_path, robot_cfg)

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(args.prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"prim 未创建成功: {args.prim_path}")
    print("[VIEW] prim 已创建", flush=True)

    # 按包围盒把模型整体抬到地板上方, 避免零位网格穿地卡住 (URDF 系 USD 原点
    # 在 pelvis, 零位脚在 z<0; 如 tiangong2plus 脚到 -0.99, 不穿地直接陷入地面)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bbox = cache.ComputeWorldBound(robot_prim).ComputeAlignedRange()
    print("[VIEW] 包围盒已计算", flush=True)
    if bbox.IsEmpty():
        raise RuntimeError(f"模型包围盒为空: {args.prim_path}")
    if args.ground and args.clearance >= 0:
        lift = args.clearance - bbox.GetMin()[2]
        if abs(lift) > 1e-3:
            xform = UsdGeom.Xformable(robot_prim)
            trans_op = next((op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
            (trans_op or xform.AddTranslateOp()).Set(Gf.Vec3d(0.0, 0.0, lift))
            print(f"[VIEW] 抬升模型 z += {lift:.4f} (bbox_min_z={bbox.GetMin()[2]:.4f})", flush=True)
        bbox = cache.ComputeWorldBound(robot_prim).ComputeAlignedRange()

    sim.reset()
    print("[VIEW] sim.reset 完成", flush=True)

    center = np.array([(bbox.GetMin()[i] + bbox.GetMax()[i]) / 2 for i in range(3)])
    size = max(np.array([bbox.GetMax()[i] - bbox.GetMin()[i] for i in range(3)]).max(), 0.5)
    lookat = _parse_xyz(args.lookat, "--lookat") if args.lookat else center
    eye = _parse_xyz(args.eye, "--eye") if args.eye else center + np.array([1.2, 1.2, 0.5]) * size
    if not args.headless:
        sim.set_camera_view(eye, lookat)
    print(f"[VIEW] {usd_path.name}: bbox_min={np.array(bbox.GetMin())}, bbox_max={np.array(bbox.GetMax())}", flush=True)
    print(f"[VIEW] camera eye={eye}, lookat={lookat}", flush=True)

    if not args.headless:
        print("[VIEW] 进入交互循环, 关闭窗口退出")
        while simulation_app.is_running():
            sim.step(render=True)
        simulation_app.close()
    else:
        # headless 自检: Isaac kit teardown (simulation_app.close) 在此环境会空转,
        # 校验已完成, 直接硬退出
        print("[VIEW] OK (headless 自检完成)")
        import os

        os._exit(0)


if __name__ == "__main__":
    main()
