import cv2
import numpy as np
import dlib
from typing import Tuple
PREDICTOR_PATH = "backend/data/params/shape_predictor_68_face_landmarks.dat"
CAMERA_CALIB_PATH = "backend/data/params/cameraCalib.xml"
FACE_PATH = "backend/data/params/faceModelGeneric.txt"
fx, fy, cx, cy = 960, 960, 640, 360  # カメラの焦点距離（fx, fy）と中心点（cx, cy）
# 顔検出モデルと顔の特徴点予測モデルを取得
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_model = np.loadtxt(FACE_PATH)
# カメラ行列を設定
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
]).astype(np.float64)
# カメラの歪み係数を読み込む
fid = cv2.FileStorage(CAMERA_CALIB_PATH, cv2.FileStorage_READ)
camera_distortion = fid.getNode("cam_distortion").mat()


def get_facial_landmarks(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    顔の輪郭を取得し、特定のランドマーク（目と口の端）の2D座標を抽出します。
    Args:
        image: 画像
    Returns:
        keypoints: 特定のランドマークの2D座標
        points: 68のランドマークの2D座標
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, None

    # Use the largest face if multiple faces are detected
    face = max(faces, key=lambda rect: rect.width() * rect.height())

    landmarks = predictor(gray, face)
    points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(68)])

    # Extract the 2D coordinates of specific landmarks (eyes and mouth corners)
    keypoints = points[[36, 39, 42, 45, 48, 54]]  # 右目右端、右目左端、左目右端、左目左端、口右端、口左端（写真基準）
    return keypoints, points


def apply_perspective_transform(src_corners: np.ndarray, dst_corners: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    4隅の視点座標（２D）に射影変換を適用してスクリーン座標と一致させ、他の視点座標を変換します。
    Args:
        src_corners: 4隅の視点座標（２D）
        dst_corners: 4隅の本来のスクリーン座標（２D）
        points: 変換する点の座標（２D）

    Returns:
        transformed_points: スクリーン座標系に変換された点の座標（２D）
    """
    # 射影変換行列の計算
    matrix = cv2.getPerspectiveTransform(dst_corners, src_corners)

    # 点の座標を変換するために、同次座標に変換
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # (x, y) -> (x, y, 1)

    # 射影変換を適用
    transformed_points_homogeneous = np.dot(matrix, points_homogeneous.T).T  # 行列の積

    # 同次座標から通常の座標に戻す
    transformed_points = (transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2][:, np.newaxis])

    return transformed_points


def estimate_head_pose(landmarks: np.ndarray, face_points: np.ndarray, iterate=True):
    """
    顔のランドマークと顔モデルを使用して、顔の姿勢・位置座標(カメラを基準とした３D座標系上)を推定します。
    Args:
        landmarks: 顔のランドマーク座標（2D）
        face_points: 一般的な顔モデルの相対的な位置座標（3D）
        iterate (bool, optional): Defaults to True.
    Returns:
        rvec: 顔の姿勢ベクトル
        tvec: 3D位置ベクトル
    """
    _, rvec, tvec = cv2.solvePnP(face_points, landmarks, camera_matrix, camera_distortion, flags=cv2.SOLVEPNP_EPNP)
    if iterate:
        _, rvec, tvec = cv2.solvePnP(face_points, landmarks, camera_matrix, camera_distortion, rvec, tvec, True)
    print("rvec:", rvec)
    print("tvec:", tvec)

    return rvec, tvec


def estimate_eye_location(hr: np.ndarray, ht: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    顔、姿勢、位置、視線を使用して、目の位置を推定します。
    Args:
        face: 顔の位置(3D・グローバル座標系)
        hr: 顔の姿勢
        ht: 顔の位置
    Returns:
        re: 右目の位置（３D）
        le: 左目の位置（３D）
    """
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # 3×3の回転行列（グローバル（Sによる接近前）→ローカル）を算出（cv2.Rodrigues(hr)[1]はヤコビアン行列（x,y,z用））
    Fc = np.dot(hR, face_model) + ht  # ローカル座標でのFcの位置をローカル用（３D）に変換
    re = 0.5 * (Fc[:, 0] + Fc[:, 1]).reshape((3,))  # ローカルのre
    le = 0.5 * (Fc[:, 2] + Fc[:, 3]).reshape((3,))  # ローカルのle
    return re, le


def fetch_eyes(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    画像から両目の中心座標を取得します。
    Args:
        path: 画像のパス

    Returns:
        local_left_eye_center: 左目の中心座標（３D）
        local_right_eye_center: 右目の中心座標（３D）
    """
    img_original = cv2.imread(path)
    img = cv2.undistort(img_original, camera_matrix, camera_distortion)  # 歪みをなくす

    # Assuming detector and predictor have been loaded from Dlib
    landmarks, _ = get_facial_landmarks(img)
    # グローバル(S接近前)座標での、それぞれの顔座標を定義
    num_pts = face_model.shape[1]
    face_points = face_model.T.reshape(num_pts, 1, 3)
    landmarks = landmarks.astype(np.float32)
    landmarks = landmarks.reshape(num_pts, 1, 2)
    hr, ht = estimate_head_pose(landmarks, face_points)  # カメラの位置をグローバル(S接近前)→ローカルにする際のhr,htを算出

    local_right_eye_center, local_left_eye_center = estimate_eye_location(hr, ht)  # グローバルカメラから見たときの[img_warped,hr_norm,gc_normalized](gc_normalizedはgc参照で算出したものなのであてにならない)

    # 両目の中心（３D）を計算
    print("Right eye center(local):", local_right_eye_center)
    print("Left eye center(local):", local_left_eye_center)
    return local_left_eye_center, local_right_eye_center


def calculate_gaze_point(eye_position: np.ndarray, gaze_vector: np.ndarray) -> np.ndarray:
    """
    PCスクリーンをXY平面としたときの視点座標を求める.

    Parameters:
    eye_position (np.ndarray): 目の座標 (x, y, z)
    gaze_vector (np.ndarray): 視線ベクトル (x, y, z)

    Returns:
    np.ndarray: PCスクリーンをXY平面としたときの視点座標 (x, y)
    """
    if gaze_vector[2] == 0:
        return None  # Parallel to xy-plane, no intersection

    t = -eye_position[2] / gaze_vector[2]  # Solve for z=0
    gaze_x = eye_position[0] + t * gaze_vector[0]
    gaze_y = eye_position[1] + t * gaze_vector[1]

    return np.array([gaze_x, gaze_y])
