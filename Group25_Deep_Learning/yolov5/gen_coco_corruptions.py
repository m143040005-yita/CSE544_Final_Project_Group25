import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# ==== 這兩個路徑請依你實際情況調整 ====
# SRC_IMG_DIR = "datasets/coco/val2017"          # 原始 COCO val2017 圖片
SRC_IMG_DIR = "datasets/coco/images/val2017"          # 原始 COCO val2017 圖片
SRC_LABEL_DIR = "datasets/coco/labels/val2017" # 對應的 YOLO 格式標註
DST_ROOT = "datasets/coco_c"                   # 要放腐蝕圖的根目錄
# ======================================

os.makedirs(DST_ROOT, exist_ok=True)

# ---- 各種 corruption 的參數（severity = 3 大約中等強度） ----
def motion_blur(img, severity=3):
    # kernel size 隨 severity 增加
    k_list = {1: 5, 2: 9, 3: 13, 4: 17, 5: 21}
    k = k_list.get(severity, 13)
    kernel = np.zeros((k, k))
    kernel[int((k - 1) / 2), :] = np.ones(k)
    kernel = kernel / k
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img, severity=3):
    # sigma 與 severity 成正比
    h, w, c = img.shape
    sigma_list = {1: 8, 2: 16, 3: 25, 4: 40, 5: 60}
    sigma = sigma_list.get(severity, 25)
    noise = np.random.randn(h, w, c) * sigma
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def brightness(img, severity=3):
    # factor < 1 變暗，>1 變亮；這裡做變亮
    factor_list = {1: 1.1, 2: 1.25, 3: 1.4, 4: 1.6, 5: 1.8}
    factor = factor_list.get(severity, 1.4)
    bright = img.astype(np.float32) * factor
    bright = np.clip(bright, 0, 255).astype(np.uint8)
    return bright


def snow(img, severity=3):
    # 簡化版雪效果：白點 + 模糊
    h, w, c = img.shape
    snow_layer = np.random.randn(h, w)  # 高斯噪音當雪點
    snow_layer = (snow_layer > np.percentile(snow_layer, 100 - severity * 3)).astype(np.float32)

    snow_layer = cv2.GaussianBlur(snow_layer, (5,5), 0)
    snow_layer = np.repeat(snow_layer[:, :, None], 3, axis=2)

    alpha = 0.5  # 雪與原圖混合比例
    snowy = img.astype(np.float32) * (1 - alpha) + 255.0 * alpha * snow_layer
    snowy = np.clip(snowy, 0, 255).astype(np.uint8)
    return snowy


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def generate_for_one_corruption(name, func, severity=3):
    img_out_dir = os.path.join(DST_ROOT, "images", f"{name}_s{severity}")
    lbl_out_dir = os.path.join(DST_ROOT, "labels", f"{name}_s{severity}")

    ensure_dir(img_out_dir)
    ensure_dir(lbl_out_dir)

    # 1) 影像腐蝕
    img_files = [f for f in os.listdir(SRC_IMG_DIR) if f.lower().endswith(".jpg")]
    print(f"[{name}] generating corrupted images to: {img_out_dir}")
    for fname in tqdm(img_files):
        src_path = os.path.join(SRC_IMG_DIR, fname)
        dst_path = os.path.join(img_out_dir, fname)

        img = cv2.imread(src_path)
        if img is None:
            continue
        out = func(img, severity=severity)
        cv2.imwrite(dst_path, out)

    # 2) 標註檔直接 copy，檔名對應 .txt
    print(f"[{name}] copying labels to: {lbl_out_dir}")
    lbl_files = [f for f in os.listdir(SRC_LABEL_DIR) if f.lower().endswith(".txt")]
    for fname in tqdm(lbl_files):
        src_lbl = os.path.join(SRC_LABEL_DIR, fname)
        dst_lbl = os.path.join(lbl_out_dir, fname)
        shutil.copy2(src_lbl, dst_lbl)


def main():
    severity = 3  # 你可以改成 1~5

    corruptions = [
        ("motion_blur", motion_blur),
        ("gaussian_noise", gaussian_noise),
        ("snow", snow),
        ("brightness", brightness),
    ]

    for name, func in corruptions:
        generate_for_one_corruption(name, func, severity=severity)


if __name__ == "__main__":
    main()

