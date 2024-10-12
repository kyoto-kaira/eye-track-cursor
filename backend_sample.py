from backend.ax_gaze_estimation import calibrate, infer_gaze_position, draw_gaze_vector
import cv2
import os
import logging
from typing import Tuple
import numpy as np

# ======================
# 使用例
# ======================


if __name__ == '__main__':
    # キャリブレーション画像の読み込み（例として4つの画像を使用）
    calibration_images = []
    screen_positions = []  # それぞれの画像に対応するスクリーン上の位置

    # 例として画像を読み込む（実際にはフロントエンドからnp.ndarrayが渡される）
    screen_positions = []
    for img_path, position in [
        ('backend/data/Images/FaceSamples/input/top_left.jpg', (0, 0)),
        ('backend/data/Images/FaceSamples/input/top_right.jpg', (1280, 0)),
        ('backend/data/Images/FaceSamples/input/down_left.jpg', (0, 720)),
        ('backend/data/Images/FaceSamples/input/down_right.jpg', (1280, 720))
    ]:
        img = cv2.imread(img_path)
        calibration_images.append(img)
        screen_positions.append(position)

    # キャリブレーションの実行
    M = calibrate(calibration_images, screen_positions)
    screen_positions = []
    logging.info("___________start of inference_______________")
    for img_path in os.listdir('backend/data/Images/FaceSamples/input'):
        img = cv2.imread(f'backend/data/Images/FaceSamples/input/{img_path}')
        # 視線位置を推定
        screen_position = infer_gaze_position(img, (1280, 720), M)
        screen_positions.append((screen_position, img_path))
        logging.info(screen_position)
        # 視線結果を描く
        img = draw_gaze_vector(img)
        os.makedirs('backend/data/Images/FaceSamples/with_gaze', exist_ok=True)
        cv2.imwrite(f'backend/data/Images/FaceSamples/with_gaze/{img_path.replace(".jpg", "_res.jpg")}', img)
        logging.info(f"___________end of {img_path}_______________")
