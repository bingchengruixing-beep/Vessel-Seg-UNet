"""
数据集划分脚本（用户方案：外源测试 + 本地训练/验证）

划分口径：
  - 测试集 = DIAS 全部 50 张（外源，纯跨源测试）
  - 训练集/验证集 = 本地手动标注 177 张，按「病人」整块切 80/20
      dataset1: 41 病人 × 3 时相 -> 33 train + 8 val
      dataset2: 54 病人 × 1 时相 -> 43 train + 11 val

核心规则：病人是划分最小单位。
  dataset1 同一编号在 2~3s / 4s / 5~6s 三个时相目录里的图是「同一个病人的
  三个造影时刻」，必须整块进同一集合，绝不能拆到 train 和 val 两边。

固定随机种子（42）保证可复现。运行后生成划分清单 split_log.txt。
"""

import os
import random
import shutil

random.seed(42)

BASE = r"E:/Unet模型训练/数据集-标注完成"
DST = os.path.join(BASE, "跨源划分")

DS1 = os.path.join(BASE, "dataset1")            # 2~3s / 4s / 5~6s
DS2 = os.path.join(BASE, "dataset2（4s）")        # normal / masks
DIAS = os.path.join(BASE, "开源数据集；DIAS")     # train / val

DS1_PHASES = ["2~3s", "4s", "5~6s"]

# 划分数量（病人数）
DS1_TRAIN, DS1_VAL = 33, 8     # 41 = 33 + 8
DS2_TRAIN, DS2_VAL = 43, 11    # 54 = 43 + 11


def list_ids(img_dir):
    """返回图像目录下所有文件的 stem（编号），按数值排序。"""
    ids = []
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            ids.append(os.path.splitext(f)[0])
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def copy_pair(src_img, src_mask, dst_img_dir, dst_mask_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    shutil.copy2(src_img, os.path.join(dst_img_dir, os.path.basename(src_img)))
    shutil.copy2(src_mask, os.path.join(dst_mask_dir, os.path.basename(src_mask)))


def split_ids(ids, n_train):
    ids = list(ids)
    random.shuffle(ids)
    return ids[:n_train], ids[n_train:]


def main():
    # 清空旧划分，重建
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)

    log = []  # 划分记录

    # ── dataset1：按病人整块切（三个时相一起走）──
    ds1_ids = list_ids(os.path.join(DS1, DS1_PHASES[0], "normal"))  # 41 个
    ds1_train, ds1_val = split_ids(ds1_ids, DS1_TRAIN)
    log.append(f"[dataset1] 总 {len(ds1_ids)} 病人 -> train {len(ds1_train)} / val {len(ds1_val)}")
    log.append(f"  train 编号: {', '.join(ds1_train)}")
    log.append(f"  val   编号: {', '.join(ds1_val)}")

    for phase in DS1_PHASES:
        src_img = os.path.join(DS1, phase, "normal")
        src_mask = os.path.join(DS1, phase, "masks")
        dst_name = "ds1_" + phase
        for pid in ds1_train:
            copy_pair(
                os.path.join(src_img, pid + ".png"),
                os.path.join(src_mask, pid + ".png"),
                os.path.join(DST, "train", dst_name, "normal"),
                os.path.join(DST, "train", dst_name, "masks"),
            )
        for pid in ds1_val:
            copy_pair(
                os.path.join(src_img, pid + ".png"),
                os.path.join(src_mask, pid + ".png"),
                os.path.join(DST, "val", dst_name, "normal"),
                os.path.join(DST, "val", dst_name, "masks"),
            )

    # ── dataset2：按编号整块切 ──
    ds2_ids = list_ids(os.path.join(DS2, "normal"))  # 54 个
    ds2_train, ds2_val = split_ids(ds2_ids, DS2_TRAIN)
    log.append(f"[dataset2] 总 {len(ds2_ids)} 病人 -> train {len(ds2_train)} / val {len(ds2_val)}")
    log.append(f"  train 编号: {', '.join(ds2_train)}")
    log.append(f"  val   编号: {', '.join(ds2_val)}")

    for pid in ds2_train:
        copy_pair(
            os.path.join(DS2, "normal", pid + ".png"),
            os.path.join(DS2, "masks", pid + ".png"),
            os.path.join(DST, "train", "ds2_4s", "normal"),
            os.path.join(DST, "train", "ds2_4s", "masks"),
        )
    for pid in ds2_val:
        copy_pair(
            os.path.join(DS2, "normal", pid + ".png"),
            os.path.join(DS2, "masks", pid + ".png"),
            os.path.join(DST, "val", "ds2_4s", "normal"),
            os.path.join(DST, "val", "ds2_4s", "masks"),
        )

    # ── DIAS：全部 50 张合并为测试集，重命名 001~050 避免 train/val 同名冲突 ──
    dias_files = []
    for sub in ["train", "val"]:
        img_dir = os.path.join(DIAS, sub, "images")
        for f in sorted(list_ids(img_dir), key=lambda x: int(x)):
            dias_files.append((sub, f))
    log.append(f"[DIAS] 总 {len(dias_files)} 张 -> 全部做 test")

    for i, (sub, pid) in enumerate(dias_files, start=1):
        new_name = f"{i:03d}.png"
        copy_pair(
            os.path.join(DIAS, sub, "images", pid + ".png"),
            os.path.join(DIAS, sub, "masks", pid + ".png"),
            os.path.join(DST, "test", "images"),
            os.path.join(DST, "test", "masks"),
        )
        # 重命名为 001~050
        os.rename(
            os.path.join(DST, "test", "images", pid + ".png"),
            os.path.join(DST, "test", "images", new_name),
        )
        os.rename(
            os.path.join(DST, "test", "masks", pid + ".png"),
            os.path.join(DST, "test", "masks", new_name),
        )

    # ── 统计 ──
    def count(img_dir):
        return len([f for f in os.listdir(img_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])

    train_imgs = sum(count(os.path.join(DST, "train", d, "normal"))
                     for d in os.listdir(os.path.join(DST, "train")))
    val_imgs = sum(count(os.path.join(DST, "val", d, "normal"))
                   for d in os.listdir(os.path.join(DST, "val")))
    test_imgs = count(os.path.join(DST, "test", "images"))

    log.append("")
    log.append(f"=== 划分结果 ===")
    log.append(f"训练集: {train_imgs} 张")
    log.append(f"验证集: {val_imgs} 张")
    log.append(f"测试集: {test_imgs} 张")

    # 写日志
    with open(os.path.join(DST, "split_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")

    print("\n".join(log))


if __name__ == "__main__":
    main()
