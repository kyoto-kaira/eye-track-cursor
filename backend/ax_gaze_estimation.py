import numpy as np
import cv2
from typing import List, Tuple
from eye_points_estimation import calculate_gaze_point, initialize_gaze_estimator, fetch_eye_locations_from_image
import ax_gaze_estimation_utils as gut

# 設定のdefault値
default_estimator_setting = initialize_gaze_estimator()


def predict_gaze(img: np.ndarray, estimator_data: dict = default_estimator_setting, gazes_only=True):
    """
    画像から視線を推定します。

    Parameters
    ----------
    img : NumPy array
        BGRチャンネルの画像。
    estimator_data : dict
        モデルや設定を含む辞書。
    gazes_only : bool, optional
        Trueの場合、推定された視線のみを返します。

    Returns
    -------
    gazes_vec : NumPy array
        推定された3D視線ベクトル(x, y, z)。
    gaze_centers : NumPy array, optional
        視線ベクトルの原点の推定値。
    eyes_iris : tuple[NumPy array, NumPy array], optional
        推定された目と虹彩のランドマーク。
    hps_orig : NumPy array, optional
        頭部姿勢（ラジアン）。
    roi_centers : NumPy array, optional
        クロップされた顔画像の中心点(x, y)。
    """
    include_iris = estimator_data['include_iris']
    include_head_pose = estimator_data['include_head_pose']

    face_detector = estimator_data['face_detector']
    face_estimator = estimator_data['face_estimator']
    gaze_estimator = estimator_data['gaze_estimator']

    if include_iris:
        iris_estimator = estimator_data['iris_estimator']
    if include_head_pose:
        hp_estimator = estimator_data['hp_estimator']

    gazes_vec = None
    gaze_centers = None
    eyes_iris = None
    hps_orig = None
    roi_centers = None

    # 顔検出
    input_face_det, scale, padding = gut.face_detector_preprocess(img)
    preds_det = face_detector.predict([input_face_det])
    detections = gut.face_detector_postprocess(preds_det)

    # 顔ランドマーク推定
    if detections[0].size != 0:
        face_imgs, face_affs, roi_centers, theta = gut.face_lm_preprocess(
            img, detections, scale, padding
        )
        face_estimator.set_input_shape(face_imgs.shape)
        landmarks, _ = face_estimator.predict([face_imgs])
        if not include_iris:
            gaze_centers = gut.face_lm_postprocess(landmarks, face_affs)
        else:
            # 虹彩ランドマーク推定（オプション）
            eye_imgs, eye_origins = gut.iris_preprocess(face_imgs, landmarks)
            iris_estimator.set_input_shape(eye_imgs.shape)
            eyes_norm, iris_norm = iris_estimator.predict([eye_imgs])
            gaze_centers, eyes_iris = gut.iris_postprocess(eyes_norm, iris_norm, eye_origins, face_affs)

        # 頭部姿勢推定（オプション）
        if include_head_pose:
            input_hp = gut.head_pose_preprocess(face_imgs)
            hp_estimator.set_input_shape(input_hp.shape)
            hps = hp_estimator.predict([input_hp])
            hps, hps_orig = gut.head_pose_postprocess(hps, theta)

        # 視線推定
        gaze_input_blob = gaze_estimator.get_input_blob_list()
        gaze_input1 = np.moveaxis(face_imgs, 1, -1)
        gaze_estimator.set_input_blob_shape(gaze_input1.shape, gaze_input_blob[0])
        gaze_estimator.set_input_blob_data(gaze_input1, gaze_input_blob[0])
        if include_head_pose:
            gaze_input2 = hps
            gaze_estimator.set_input_blob_shape(gaze_input2.shape, gaze_input_blob[1])
            gaze_estimator.set_input_blob_data(gaze_input2, gaze_input_blob[1])
        gaze_estimator.update()
        gazes = gaze_estimator.get_results()[0]
        gazes_vec = gut.gaze_postprocess(gazes, face_affs)
        print("gazes_vec", gazes_vec)
    if gazes_only:
        return gazes_vec
    else:
        return gazes_vec, gaze_centers, eyes_iris, hps_orig, roi_centers


def calibrate(calibration_images: List[np.ndarray], screen_positions: List[Tuple[int, int]], estimator_data: dict = default_estimator_setting):
    """
    キャリブレーション画像と対応するスクリーン位置を使用してキャリブレーションを行います。

    Parameters:
    calibration_images (List[np.ndarray]): キャリブレーション用の画像リスト。
    screen_positions (List[Tuple[int, int]]): 対応するスクリーン上の位置（x, y）のリスト。
    estimator_data : dict
        モデルや設定を含む辞書。

    Returns:
    M (np.ndarray): 射影変換行列。
    """
    # 視線ベクトルと視線の原点を保持するリストを初期化
    gaze_vectors = []
    gaze_centers = []
    # 各キャリブレーション画像を処理
    for image in calibration_images:
        # 視線ベクトルを推定
        preds = predict_gaze(image, estimator_data, gazes_only=False)
        if preds[0] is not None:
            gaze_vec = preds[0][0]  # 画像内の顔が1つであると仮定
            eye_left, eye_right = fetch_eye_locations_from_image(image)
            gaze_center = (eye_left + eye_right) / 2
            gaze_vectors.append(gaze_vec)
            gaze_centers.append(gaze_center)
        else:
            # 視線ベクトルが取得できない場合の処理
            pass
    # 各視線ベクトルと視線の原点から視点座標を計算
    gaze_points = []
    for gaze_vec, gaze_center in zip(gaze_vectors, gaze_centers):
        print(gaze_center, gaze_vec)
        gaze_point = calculate_gaze_point(gaze_center, gaze_vec)
        gaze_points.append(gaze_point)
    # 推定された視点座標と実際のスクリーン位置から射影変換を計算
    src_points = np.float32(gaze_points)
    dst_points = np.float32(screen_positions)
    # 射影変換行列を計算
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return M


def infer_gaze_position(image: np.ndarray, screen_size: Tuple[int, int], M: np.ndarray, estimator_data: dict = default_estimator_setting) -> Tuple[int, int]:
    """
    キャリブレーション結果を使用して、画像からスクリーン上の視線位置を推定します。

    Parameters:
    image (np.ndarray): 入力画像。
    screen_size (Tuple[int, int]): スクリーンのサイズ（幅、高さ）。
    estimator_data : dict
        モデルや設定を含む辞書。

    Returns:
    screen_position (Tuple[int, int]): 推定されたスクリーン上の位置（x, y）。
    """
    if estimator_data['calibration_result'] is None:
        raise ValueError("キャリブレーションが実行されていません。")
    # 視線ベクトルを推定
    preds = predict_gaze(image, estimator_data, gazes_only=False)
    if preds[0] is not None:
        gaze_vec = preds[0][0]  # 画像内の顔が1つであると仮定
        gaze_center = preds[1][0]
        # 視点座標を計算
        gaze_point = calculate_gaze_point(gaze_center, gaze_vec)
        # 変換行列を視点座標に適用
        gaze_point_homogeneous = np.array([gaze_point[0], gaze_point[1], 1.0], dtype=np.float32)
        transformed_point = np.dot(M, gaze_point_homogeneous)
        # 同次座標を通常の座標に変換
        screen_x = transformed_point[0] / transformed_point[2]
        screen_y = transformed_point[1] / transformed_point[2]
        # 値をスクリーンサイズにクランプ(スクリーンの範囲内に視点座標を収める)
        screen_x = np.clip(screen_x, 0, screen_size[0])
        screen_y = np.clip(screen_y, 0, screen_size[1])
        screen_position = (int(screen_x), int(screen_y))
        return screen_position
    else:
        # 視線ベクトルが取得できない場合の処理
        return None


def draw_gaze_vector(image: np.ndarray, estimator_data: dict = default_estimator_setting) -> np.ndarray:
    """
    画像に視線ベクトルを描画します。

    Parameters:
    image (np.ndarray): 入力画像。
    estimator_data : dict
        モデルや設定を含む辞書。

    Returns:
    image_with_gaze_vector (np.ndarray): 視線ベクトルが描画された画像。
    """
    preds = predict_gaze(image, estimator_data, gazes_only=False)
    if preds[0] is not None:
        img_draw = image.copy()
        gazes, gaze_centers, eyes_iris, hps, roi_centers = preds
        img_draw = gut.draw(
            img_draw, gazes, gaze_centers, eyes_iris=eyes_iris, hps=hps, roi_centers=roi_centers,
            draw_iris=True, draw_head_pose=True, horizontal_flip=False
        )
        return img_draw
    else:
        return image  # 視線ベクトルが取得できない場合は元の画像を返す

# ======================
# 使用例
# ======================


if __name__ == '__main__':
    # キャリブレーション画像の読み込み（例として4つの画像を使用）
    calibration_images = []
    screen_positions = []  # それぞれの画像に対応するスクリーン上の位置

    # 例として画像を読み込む（実際にはフロントエンドからnp.ndarrayが渡される）
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

    # 推論の実行例（カメラの設定までは環境構築しておりません）
    # 例としてカメラからのフレームを取得
    cap = cv2.VideoCapture(0)
    screen_size = (1280, 720)  # スクリーンのサイズ（幅、高さ）

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 視線位置を推定
        screen_position = infer_gaze_position(frame, screen_size, M)
        if screen_position is not None:
            # スクリーン上に視線位置を表示するなどの処理
            print(f"視線位置: {screen_position}")

        # 視線ベクトルを描画
        frame_with_gaze = draw_gaze_vector(frame)

        # フレームを表示
        cv2.imshow('Gaze Estimation', frame_with_gaze)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
