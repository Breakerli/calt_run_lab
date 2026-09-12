"""从 tiangong2_plus_urdf.urdf 自动生成 MuJoCo MJCF。

用 MuJoCo 内置 URDF loader 编译 URDF, 再 mj_saveLastXML 序列化回 MJCF,
保证运动学/惯量/碰撞/限位与 URDF 严格一致 (已核对 USD 与 URDF 一致:
关节名/轴向 roll=x,pitch=y,yaw=z/限位/effort/velocity 均相符)。

URDF mesh 路径为 package://tiangong2plus_urdf/meshes/*.STL, MuJoCo 不认识
package:// URI, 故在临时目录生成去掉 package 前缀的 URDF 副本 + meshes 符号链接,
编译完成后删除。输出 MJCF 的 meshdir 指向 ../meshes/。

URDF 无 transmission 标签, 生成的 MJCF 不含执行器 (后续按训练 cfg 补 position
actuator); MuJoCo URDF loader 默认把 root link 焊到 world, 脚本会包一层
<body name="pelvis"> + <freejoint name="root"> 使基座浮动。

运行: /home/robot02/miniconda3/envs/calt_run/bin/python legged_lab/scripts/utils/generate_tiangong2plus_mjcf.py
"""

import argparse
import os
import re
import shutil
import tempfile

import mujoco

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
ASSET_DIR = os.path.join(REPO_ROOT, "legged_lab/assets/tiangong2plus")
URDF_FILE = os.path.join(ASSET_DIR, "urdf/tiangong2_plus_urdf.urdf")
MESH_DIR = os.path.join(ASSET_DIR, "meshes")
OUT_DIR = os.path.join(ASSET_DIR, "xml")
OUT_XML = os.path.join(OUT_DIR, "tiangong2_plus.xml")
PACKAGE_PREFIX = "package://tiangong2plus_urdf/meshes/"


def parse_args():
    parser = argparse.ArgumentParser(description="从 tiangong2_plus_urdf.urdf 生成 MJCF")
    parser.add_argument("--urdf", type=str, default=URDF_FILE, help="源 URDF 路径")
    parser.add_argument("--out", type=str, default=OUT_XML, help="输出 MJCF 路径")
    return parser.parse_args()


# --- URDF 预处理 ---
def prepare_urdf_workdir(urdf_path):
    """临时目录: 去 package:// 前缀的 URDF 副本 + meshes 符号链接, 供 MuJoCo 解析。"""
    tmp = tempfile.mkdtemp(prefix="tiangong_urdf_")
    with open(urdf_path, encoding="utf-8") as f:
        text = f.read()
    text = text.replace(PACKAGE_PREFIX, "")
    stripped = os.path.join(tmp, os.path.basename(urdf_path))
    with open(stripped, "w", encoding="utf-8") as f:
        f.write(text)
    for fn in os.listdir(MESH_DIR):
        if fn.lower().endswith(".stl"):
            dst = os.path.join(tmp, fn)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(MESH_DIR, fn), dst)
    return tmp, stripped


# --- MJCF 生成 ---
def generate(urdf_path, out_path):
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF 不存在: {urdf_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp, stripped = prepare_urdf_workdir(urdf_path)
    try:
        model = mujoco.MjModel.from_xml_path(stripped)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mujoco.mj_saveLastXML(out_path, model)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # meshdir 指向 ../meshes/ (MJCF 在 xml/ 下); meshdir 是 <compiler> 的属性
    with open(out_path, encoding="utf-8") as f:
        xml = f.read()
    if "meshdir" in xml:
        xml = re.sub(r'meshdir="[^"]*"', 'meshdir="../meshes/"', xml)
    else:
        xml = re.sub(r"<compiler ", '<compiler meshdir="../meshes/" ', xml, count=1)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    _float_root(out_path)
    return model


# --- MJCF 后处理 ---
def _float_root(out_path):
    """MuJoCo URDF loader 把 root link (pelvis) 焊死到 worldbody;

    包一层 <body name="pelvis"> + <freejoint>, 让基座浮动 (nq = 6 + 关节数)。
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(out_path)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise RuntimeError("MJCF 中找不到 worldbody")
    body = ET.Element("body", {"name": "pelvis"})
    ET.SubElement(body, "freejoint", {"name": "root"})
    for child in list(worldbody):
        worldbody.remove(child)
        body.append(child)
    worldbody.append(body)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


# --- 校验 ---
def validate(out_path):
    model = mujoco.MjModel.from_xml_path(out_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    free = [j for j in joints if "pelvis" in str(j) or "root" in str(j) or "free" in str(j)]
    print(f"[OK] {out_path}")
    print(f"     nq={model.nq} nv={model.nv} nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} nu={model.nu}")
    print(f"     movable joints: {len([j for j in joints if j not in free])} (期望 21 revolute + fixed 被合并)")
    print(f"     root/free joints: {free} (期望含 pelvis freejoint)")
    assert model.nv == 6 + 21, f"nv 期望 27 (6 free + 21 revolute), 实际 {model.nv}"
    return model


def main():
    args = parse_args()
    generate(args.urdf, args.out)
    validate(args.out)


if __name__ == "__main__":
    main()
