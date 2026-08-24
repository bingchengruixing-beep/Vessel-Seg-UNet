"""
实验 14 划分脚本：从实验 13 的「跨源划分」派生，只移动 DIAS 的 train 30 张。

口径（用户方案）：
  - 训练集 = 本地 142（跨源划分/train 原样复制）+ DIAS-train 30（001~030）= 172
  - 验证集 = 本地 35（跨源划分/val 原样复制，不变）
  - 测试集 = DIAS-val 20（031~050）

关键：本地 142/35 的病人划分必须与实验 13 完全一致，故直接从实验 13 的
「跨源划分」目录派生（copy），不重新 shuffle。
"""
import os
import shutil

SRC = r"E:/Unet模型训练/数据集-标注完成/跨源划分"
DST = r"E:/Unet模型训练/数据集-标注完成/跨源划分14"

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def count(d):
    return len([f for f in os.listdir(d) if f.lower().endswith(IMG_EXT)])


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)

    # 1. 本地 train/val 原样复制（142 + 35）
    shutil.copytree(os.path.join(SRC, "train"), os.path.join(DST, "train"))
    shutil.copytree(os.path.join(SRC, "val"), os.path.join(DST, "val"))

    # 2. DIAS-train 30（001~030）挪回训练集 → train/dias_train/
    for i in range(1, 31):
        name = f"{i:03d}.png"
        copy_file(os.path.join(SRC, "test", "images", name),
                  os.path.join(DST, "train", "dias_train", "normal", name))
        copy_file(os.path.join(SRC, "test", "masks", name),
                  os.path.join(DST, "train", "dias_train", "masks", name))

    # 3. DIAS-val 20（031~050）→ test/
    for i in range(31, 51):
        name = f"{i:03d}.png"
        copy_file(os.path.join(SRC, "test", "images", name),
                  os.path.join(DST, "test", "images", name))
        copy_file(os.path.join(SRC, "test", "masks", name),
                  os.path.join(DST, "test", "masks", name))

    # 统计
    train_imgs = sum(count(os.path.join(DST, "train", d, "normal"))
                     for d in os.listdir(os.path.join(DST, "train")))
    val_imgs = sum(count(os.path.join(DST, "val", d, "normal"))
                   for d in os.listdir(os.path.join(DST, "val")))
    test_imgs = count(os.path.join(DST, "test", "images"))

    log = [
        "=== 实验14 划分结果 ===",
        f"训练集: {train_imgs} 张（本地 142 + DIAS-train 30）",
        f"验证集: {val_imgs} 张（本地 35，不变）",
        f"测试集: {test_imgs} 张（DIAS-val 20）",
    ]
    with open(os.path.join(DST, "split_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
