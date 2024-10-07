import sys
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
import ailia
import cv2
import numpy as np
import json
import ax_gaze_estimation_utils as gut

sys.path.append('utils')
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'woman_face.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 1240
IMAGE_WIDTH = 680


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Gaze estimation.',IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
parser.add_argument(
    '--include-iris',
    action='store_true',
    help='By default, the model does not estimate iris landmarks and uses a' +
    'rough estimation for the pupil centers. This option allows a more ' +
    'accurate estimation but adds overhead (slower).'
)
parser.add_argument(
    '--draw-iris',
    action='store_true',
    help='Whether to draw the iris landmarks or not.'
)
parser.add_argument(
    '--include-head-pose',
    action='store_true',
    help='By default, the model only uses the face images to predict the' +
    'gaze. This option allows including the head pose for prediction (higher' +
    'accuracy but slower).'
)
parser.add_argument(
    '--draw-head-pose',
    action='store_true',
    help='Whether to draw the head pose(s) or not.'
)
parser.add_argument(
    '-l', '--lite',
    action='store_true',
    help='With this option, a lite version of the head pose model is used ' +
    '(only valid when --include-head-pose is specified).'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
FACE_DET_MODEL_NAME = 'blazeface'
FACE_LM_MODEL_NAME = 'facemesh'
IRIS_LM_MODEL_NAME = 'iris'
if args.lite:
    HEAD_POSE_MODEL_NAME = 'hopenet_lite'
else:
    HEAD_POSE_MODEL_NAME = 'hopenet_robust_alpha1'
if args.include_head_pose:
    GAZE_MODEL_NAME = 'ax_gaze_estimation_hp'
else:
    GAZE_MODEL_NAME = 'ax_gaze_estimation'
if args.normal:
    FACE_DET_WEIGHT_PATH = f'{FACE_DET_MODEL_NAME}.onnx'
    FACE_DET_MODEL_PATH = f'{FACE_DET_MODEL_NAME}.onnx.prototxt'
    FACE_LM_WEIGHT_PATH = f'{FACE_LM_MODEL_NAME}.onnx'
    FACE_LM_MODEL_PATH = f'{FACE_LM_MODEL_NAME}.onnx.prototxt'
    IRIS_LM_WEIGHT_PATH = f'{IRIS_LM_MODEL_NAME}.onnx'
    IRIS_LM_MODEL_PATH = f'{IRIS_LM_MODEL_NAME}.onnx.prototxt'
    HEAD_POSE_WEIGHT_PATH = f'{HEAD_POSE_MODEL_NAME}.onnx'
    HEAD_POSE_MODEL_PATH = f'{HEAD_POSE_MODEL_NAME}.onnx.prototxt'
    GAZE_WEIGHT_PATH = f'{GAZE_MODEL_NAME}.onnx'
    GAZE_MODEL_PATH = f'{GAZE_MODEL_NAME}.onnx.prototxt'
else:
    FACE_DET_WEIGHT_PATH = f'{FACE_DET_MODEL_NAME}.opt.onnx'
    FACE_DET_MODEL_PATH = f'{FACE_DET_MODEL_NAME}.opt.onnx.prototxt'
    FACE_LM_WEIGHT_PATH = f'{FACE_LM_MODEL_NAME}.opt.onnx'
    FACE_LM_MODEL_PATH = f'{FACE_LM_MODEL_NAME}.opt.onnx.prototxt'
    IRIS_LM_WEIGHT_PATH = f'{IRIS_LM_MODEL_NAME}.opt.onnx'
    IRIS_LM_MODEL_PATH = f'{IRIS_LM_MODEL_NAME}.opt.onnx.prototxt'
    HEAD_POSE_WEIGHT_PATH = f'{HEAD_POSE_MODEL_NAME}.opt.onnx'
    HEAD_POSE_MODEL_PATH = f'{HEAD_POSE_MODEL_NAME}.opt.onnx.prototxt'
    GAZE_WEIGHT_PATH = f'{GAZE_MODEL_NAME}.opt.obf.onnx'
    GAZE_MODEL_PATH = f'{GAZE_MODEL_NAME}.opt.obf.onnx.prototxt'
FACE_DET_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{FACE_DET_MODEL_NAME}/'
FACE_LM_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{FACE_LM_MODEL_NAME}/'
IRIS_LM_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/mediapipe_{IRIS_LM_MODEL_NAME}/'
HEAD_POSE_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/hopenet/'
GAZE_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/ax_gaze_estimation/'


# ======================
# Utils
# ======================
import os
import cv2
import numpy as np
import scipy.io as sio
import dlib
predictor_path="shape_predictor/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)  # Replace with path to shape predictor
def get_facial_landmarks(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, None

    # Use the largest face if multiple faces are detected
    face = max(faces, key=lambda rect: rect.width() * rect.height())

    landmarks = predictor(gray, face)
    points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(68)])

    # Extract the 2D coordinates of specific landmarks (eyes and mouth corners)
    keypoints = points[[36, 39, 42, 45,48,54]]  # 右目右端、右目左端、左目右端、左目左端、口右端、口左端（写真基準）
    print(keypoints)
    return keypoints, points
import cv2
import numpy as np

def apply_perspective_transform(src_corners, dst_corners, points):
    # 射影変換行列の計算
    matrix = cv2.getPerspectiveTransform(dst_corners, src_corners)

    # 点の座標を変換するために、同次座標に変換
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # (x, y) -> (x, y, 1)

    # 射影変換を適用
    transformed_points_homogeneous = np.dot(matrix, points_homogeneous.T).T  # 行列の積

    # 同次座標から通常の座標に戻す
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2][:, np.newaxis]

    return transformed_points

def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):#pitchyaw:[theta,phi]
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = np.min([h, w]) / 2.0
    pos = (int(w / 2.0), int(h / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:  # Convert to RGB if grayscale
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    #視線を書く
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(int)),
                  tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                 thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)
    print("rvec:",rvec)
    print("tvec:",tvec)

    return rvec, tvec

def normalizeData(img, face, hr, ht, gc, cam):
    focal_norm = 960
    distance_norm = 600
    roiSize = (60, 36)

    img_u = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ht = ht.reshape((3, 1))
    gc = gc.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]#3×３の回転行列（グローバル（Sによる接近前）→ローカル）を算出（cv2.Rodrigues(hr)[1]はヤコビアン行列（x,y,z用））
    Fc = np.dot(hR, face) + ht#ローカル座標でのFcの位置をローカル用（３D）に変換
    re = 0.5 * (Fc[:, 0] + Fc[:, 1]).reshape((3, 1))#ローカルのre
    le = 0.5 * (Fc[:, 2] + Fc[:, 3]).reshape((3, 1))#ローカルのle

    data = []
    for et in [re, le]:
        distance = np.linalg.norm(et)

        z_scale = distance_norm / distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ])
        S = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])

        hRx = hR[:, 0]
        forward = (et / distance).reshape(3)
        down = np.cross(forward, hRx)#外積
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T

        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))

        img_warped = cv2.warpPerspective(img_u, W, roiSize)
        img_warped = cv2.equalizeHist(img_warped)

        hR_norm = np.dot(R, hR)
        hr_norm = cv2.Rodrigues(hR_norm)[0]
        #gc(local)を勝手に仮定した場合、gc_normalize(グローバル)がどうなるのかを示す：gcはgaze spotで、gc_normalizeは視線（原点基準）のベクトル情報
        gc_normalized = gc - et
        gc_normalized = np.dot(R, gc_normalized)
        gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)
        #Mはg(local→S接近後のグローバルへと移す)
        M=np.dot(S,R)
        data.append([img_warped, hr_norm, gc_normalized,M])
    return data,re.reshape((3,)),le.reshape((3,))

def fetch_eyes(path):
    print(path)
    fx,fy,cx,cy=960,960,640,360
    camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]).astype(np.float64)
    fid = cv2.FileStorage('cameraCalib.xml', cv2.FileStorage_READ)
    camera_distortion = fid.getNode("cam_distortion").mat()#カメラの歪み係数
    filepath = os.path.join(path)
    img_original = cv2.imread(filepath)
    img = cv2.undistort(img_original, camera_matrix, camera_distortion)#歪みをなくす

    # Assuming detector and predictor have been loaded from Dlib
    landmarks, general_landmarks = get_facial_landmarks(img, detector, predictor)
    #グローバル(S接近前)座標での、それぞれの顔座標を定義
    face = np.loadtxt('faceModelGeneric.txt')
    num_pts = face.shape[1]
    facePts = face.T.reshape(num_pts, 1, 3)
    landmarks = landmarks.astype(np.float32)
    landmarks = landmarks.reshape(num_pts, 1, 2)
    hr, ht = estimateHeadPose(landmarks, facePts, camera_matrix, camera_distortion)#カメラの位置をグローバル(S接近前)→ローカルにする際のhr,htを算出

    gc = np.array([-127.790719, 4.621111, -12.025310])#どこを見てているのかを勝手に仮定して視線を描く(draw_gazeにしか使われない)

    data,local_right_eye_center,local_left_eye_center = normalizeData(img, face, hr, ht, gc, camera_matrix)#グローバルカメラから見たときの[img_warped,hr_norm,gc_normalized](gc_normalizedはgc参照で算出したものなのであてにならない)

    gaze_left = data[1][2]
    gaze_right = data[0][2]
    lr = ["right", "left"]

    def write_normalized(num):
        gaze_direction = data[num][2]
        gaze_theta = np.arcsin((-1) * gaze_direction[1])
        gaze_phi = np.arctan2((-1) * gaze_direction[0], (-1) * gaze_direction[2])

        img_normalized = data[num][0]
        cv2.imwrite(f'img_normalized_{lr[num]}({os.path.basename(path)}).jpg', img_normalized)
        # 視線を描く（どこを見ているのかを勝手に想定しているので、不適切）
        img_normalized = draw_gaze(img_normalized, np.array([gaze_theta[0], gaze_phi[0]]))

    #write_normalized(0)
    #write_normalized(1)

    # 両目の中心（３D）を計算
    print("Right eye center(local):", local_right_eye_center)
    print("Left eye center(local):", local_left_eye_center)
    global_left_eye_center= 0.5 * (facePts[0][0] + facePts[1][0])#S接近前
    global_right_eye_center = 0.5 * (facePts[2][0] + facePts[3][0])#S接近前
    return local_left_eye_center,local_right_eye_center
def calculate_gaze_point(eye_position, gaze_vector):
    """
    Calculate the intersection of gaze vector with xy-plane (z=0).

    Parameters:
    eye_position (torch.tensor): Eye position (x, y, z)
    gaze_vector (torch.tensor): Gaze vector (x, y, z)

    Returns:
    torch.tensor: Intersection point on xy-plane (x, y)
    """
    if gaze_vector[2] == 0:
        return None  # Parallel to xy-plane, no intersection

    t = -eye_position[2] / gaze_vector[2]  # Solve for z=0
    gaze_x = eye_position[0] + t * gaze_vector[0]
    gaze_y = eye_position[1] + t * gaze_vector[1]

    return np.array([gaze_x, gaze_y])
@contextmanager
def time_execution(msg):
    start = time.perf_counter()
    yield
    logger.debug(f'{msg} {(time.perf_counter() - start) * 1000:.0f} ms')

class GazeEstimator:
    """Class for estimating the gaze direction

    Wrap all neural networks in the pipeline to provide a centralized and
    easy-to-use class for estimating the gaze direction given an image.
    Include convenient draw method.
    """

    def __init__(self, include_iris=False, include_head_pose=False):
        """Initialize a gaze estimator with or without head pose estimation.

        Parameters
        ----------
        include_iris : bool, optional
            Estimate iris landmarks for more accurate centers of origin of the
            gaze vectors.
        include_head_pose : bool, optional
            Estimate the gaze with or without head pose information.
        """
        self.include_iris = include_iris
        self.include_head_pose = include_head_pose
        # net initialize
        self.face_detector = ailia.Net(
            FACE_DET_MODEL_PATH, FACE_DET_WEIGHT_PATH, env_id=args.env_id
        )
        self.face_estimator = ailia.Net(
            FACE_LM_MODEL_PATH, FACE_LM_WEIGHT_PATH, env_id=args.env_id
        )
        if self.include_iris:
            self.iris_estimator = ailia.Net(
                IRIS_LM_MODEL_PATH, IRIS_LM_WEIGHT_PATH, env_id=args.env_id
            )
        if self.include_head_pose:
            self.hp_estimator = ailia.Net(
                HEAD_POSE_MODEL_PATH, HEAD_POSE_WEIGHT_PATH, env_id=args.env_id
            )
        self.gaze_estimator = ailia.Net(
            GAZE_MODEL_PATH, GAZE_WEIGHT_PATH, env_id=args.env_id
        )

    def predict(self, img, gazes_only=True):
        """Predict the gaze given an image.

        Parameters
        ----------
        img : NumPy array
            The image in BGR channels.
        gazes_only : bool, optional
            If True, only return the predicted gaze(s).

        Returns
        -------
        gazes_vec : NumPy array
            Predicted 3D (x, y, z) gaze vector(s). The axes of
            reference correspond to x oriented positively to the right of the
            image, y oriented positively to the bottom of the image and z
            oriented positively to the back of the image (from the POV of
            someone looking at the image).
        gaze_centers : NumPy array, optional
            Estimated centers of origin for the gaze vectors.
        eyes_iris : tuple[NumPy array, NumPy array], optional
            Predicted eye-region and iris landmarks.
        hps_orig : NumPy array, optional
            Head pose(s) in radians. Roll (left+), yaw (right+), pitch (down+)
            values are given in the detected person's frame of reference.
        roi_centers : NumPy array, optional
            Centers (x, y) of the cropped face image(s). Used for drawing the
            head pose(s).
        """
        gazes_vec = None
        gaze_centers = None
        eyes_iris = None
        hps_orig = None
        roi_centers = None
        # Face detection
        with time_execution('\t\t\tpreprocessing'):
            input_face_det, scale, padding = gut.face_detector_preprocess(img)
        with time_execution('\t\tBlazeFace'):
            preds_det = self.face_detector.predict([input_face_det])
        with time_execution('\t\t\tpostprocessing'):
            detections = gut.face_detector_postprocess(preds_det)

        # Face landmark estimation
        if detections[0].size != 0:
            with time_execution('\t\t\tpreprocessing'):
                face_imgs, face_affs, roi_centers, theta = gut.face_lm_preprocess(
                    img, detections, scale, padding
                )
                self.face_estimator.set_input_shape(face_imgs.shape)
            with time_execution('\t\tFace Mesh'):
                landmarks, confidences = self.face_estimator.predict([face_imgs])
            if not self.include_iris:
                with time_execution('\t\t\tpostprocessing'):
                    gaze_centers = gut.face_lm_postprocess(landmarks, face_affs)
            else:
                # Iris landmark estimation (optional)
                with time_execution('\t\t\tpreprocessing'):
                    eye_imgs, eye_origins = gut.iris_preprocess(face_imgs, landmarks)
                    self.iris_estimator.set_input_shape(eye_imgs.shape)
                with time_execution('\t\tIris'):
                    eyes_norm, iris_norm = self.iris_estimator.predict([eye_imgs])
                with time_execution('\t\t\tpostprocessing'):
                    gaze_centers, eyes_iris = gut.iris_postprocess(eyes_norm, iris_norm, eye_origins, face_affs)

            # Head pose estimation (optional)
            if self.include_head_pose:
                with time_execution('\t\t\tpreprocessing'):
                    input_hp = gut.head_pose_preprocess(face_imgs)
                    self.hp_estimator.set_input_shape(input_hp.shape)
                with time_execution('\t\tHopenet'):
                    hps = self.hp_estimator.predict([input_hp])
                with time_execution('\t\t\tpostprocessing'):
                    hps, hps_orig = gut.head_pose_postprocess(hps, theta)

            # Gaze estimation
            with time_execution('\t\t\tpreprocessing'):
                gaze_input_blob = self.gaze_estimator.get_input_blob_list()
                gaze_input1 = np.moveaxis(face_imgs, 1, -1)
                self.gaze_estimator.set_input_blob_shape(gaze_input1.shape, gaze_input_blob[0])
                self.gaze_estimator.set_input_blob_data(gaze_input1, gaze_input_blob[0])
                if self.include_head_pose:
                    gaze_input2 = hps
                    self.gaze_estimator.set_input_blob_shape(gaze_input2.shape, gaze_input_blob[1])
                    self.gaze_estimator.set_input_blob_data(gaze_input2, gaze_input_blob[1])
            with time_execution('\t\tGaze estimation'):
                self.gaze_estimator.update()
                gazes = self.gaze_estimator.get_results()[0]
            with time_execution('\t\t\tpostprocessing'):
                gazes_vec = gut.gaze_postprocess(gazes, face_affs)

        if gazes_only:
            return gazes_vec
        else:
            return gazes_vec, gaze_centers, eyes_iris, hps_orig, roi_centers

    def draw(self, img, gazes, gaze_centers, eyes_iris=None, hps=None, roi_centers=None, draw_iris=False,
             draw_head_pose=False, horizontal_flip=False):
        """Draw the gaze(s) and landmarks (and head pose(s)) on the image.

        Regarding the head pose(s), (person POV) the axes correspond to
        x (blue) oriented positively to the left, y (green) oriented positively
        to the bottom and z (red) oriented positively to the back.

        Parameters
        ----------
        img : NumPy array
            The image to draw on (BGR channels).
        gazes : NumPy array
            The gaze(s) to draw.
        gaze_centers : NumPy array
            The centers of origin of the gaze(s).
        eyes_iris : NumPy array, optional
            The eye-region and iris landmarks to draw.
        hps : NumPy array, optional
            The head pose(s) to draw.
        roi_centers : NumPy array, optional
            The center(s) of origin of the head pose(s).
        draw_iris : bool, optional
            Whether to draw the iris landmarks or not.
        draw_head_pose : bool, optional
            Whether to draw the head pose(s) or not.
        horizontal_flip : bool, optional
            Whether to consider a horizontally flipped image for drawing.

        Returns
        -------
        img_draw : NumPy array
            Image with the gaze(s) and landmarks (and head pose(s)) drawn on it.
        """
        with time_execution('\t\tDrawing'):
            img_draw = img.copy()
            if eyes_iris is not None and draw_iris:
                eyes, iris = eyes_iris
                for i in range(len(eyes)):
                    gut.draw_eye_iris(
                        img_draw, eyes[i, :, :16, :2], iris[i, :, :, :2], size=1
                    )
            if horizontal_flip:
                img_draw = np.ascontiguousarray(img_draw[:, ::-1])
            if hps is not None and roi_centers is not None and draw_head_pose:
                gut.draw_head_poses(img_draw, hps, roi_centers, horizontal_flip=horizontal_flip)
            gut.draw_gazes(img_draw, gazes, gaze_centers, horizontal_flip=horizontal_flip)

        return img_draw

    def predict_and_draw(self, img, draw_iris=False, draw_head_pose=False, results=None):
        """Predict and draw the gaze(s) and landmarks (and head pose(s)).

        Convenient method for predicting the gaze(s) and landmarks (and head
        pose(s)) and drawing them at once.

        Parameters
        ----------
        img : NumPy array
            The image in BGR channels.

        Returns
        -------
        img_draw : NumPy array
            Image with the gaze(s) and landmarks (and head pose(s)) drawn on it.
        draw_iris : bool, optional
            Whether to draw the iris landmarks or not.
        draw_head_pose : bool, optional
            Whether to draw the head pose(s) or not.
        results: list, optional
            Result values stored to this list.
        """
        if results is not None:
            results.clear()
        img_draw = img.copy()
        preds = self.predict(img, gazes_only=False)
        if preds[0] is not None:
            img_draw = self.draw(img, *preds, draw_iris=draw_iris,
                                 draw_head_pose=draw_head_pose)
            if results is not None:
                results.append({
                    'gazes': preds[0],
                    'gaze_centers': preds[1],
                    'eyes_iris': preds[2],
                    'head_poses': preds[3],
                    'roi_centers': preds[4]
                })
        return img_draw


def save_result_json(json_path, results):
    output = []
    for r in results:
        output.append({k: v.tolist() for k, v in r.items() if v is not None})
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)


# ======================
# Main functions
# ======================
def recognize_from_image():
    estimator = GazeEstimator(args.include_iris, args.include_head_pose)
    gaze_points=[]
    # input image loop
    for image_path in args.input:
        results = []
        logger.info(image_path)
        src_img = imread(image_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for _ in range(5):
                start = int(round(time.time() * 1000))
                img_draw = estimator.predict_and_draw(src_img, args.draw_iris, args.draw_head_pose, results)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            img_draw = estimator.predict_and_draw(src_img, args.draw_iris, args.draw_head_pose, results)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img_draw)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_result_json(json_file, results)
        left_eye_center,right_eye_center=fetch_eyes(image_path)
        gaze_at=calculate_gaze_point((left_eye_center+right_eye_center)/2,results[-1]["gazes"][0])
        gaze_points.append([gaze_at,image_path])

    logger.info('Script finished successfully.')
    gaze_position_plot=[]
    true_position=[]
    object_points=[]
    object_filenames=[]
    for gaze_at,image_path in gaze_points:
        if image_path.find("down_right")!=-1:
            true_position.append([1280,640])
            gaze_position_plot.append(gaze_at.tolist())
        elif image_path.find("down_left")!=-1:
            true_position.append([0,640])
            gaze_position_plot.append(gaze_at.tolist())
        elif image_path.find("top_right")!=-1:
            true_position.append([1280,0])
            gaze_position_plot.append(gaze_at.tolist())
        elif image_path.find("top_left")!=-1:
            true_position.append([0,0])
            gaze_position_plot.append(gaze_at.tolist())
        else:
            object_points.append(gaze_at.tolist())
            object_filenames.append(image_path)
    print("true_position:",true_position)
    print("gaze_position_plot:",gaze_position_plot)
    applied_points=apply_perspective_transform(np.float32(true_position),np.float32(gaze_position_plot),np.array(object_points))
    
    
        
    for applied_point,file_name in zip(applied_points,object_filenames):
        plt.scatter(applied_point[0], applied_point[1], color='blue')

        # テキストをプロット（座標と名前を表示）
        plt.text(applied_point[0], applied_point[1], os.path.basename(file_name), fontsize=12, ha='right')
    for true_point in true_position:
        plt.scatter(true_point[0], true_point[1], color='red')
    #線分を引く
    plt.plot([1280,0],[0,0],color='red')
    plt.plot([1280,1280],[0,640],color='red')
    plt.plot([0,1280],[640,640],color='red')
    plt.plot([0,0],[0,640],color='red')
    # グラフ表示
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Center Point with Filename')
    plt.grid(True)
    plt.savefig('gaze_point.jpg')

def recognize_from_video():
    estimator = GazeEstimator(args.include_iris, args.include_head_pose)

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w, fps=capture.get(cv2.CAP_PROP_FPS))
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        preds = estimator.predict(frame, gazes_only=False)
        if preds[0] is not None:
            frame_draw = estimator.draw(frame, *preds, draw_iris=args.draw_iris, draw_head_pose=args.draw_head_pose)
        else:
            frame_draw = frame.copy()

        if args.video == '0': # Flip horizontally if camera
            visual_img = cv2.flip(frame_draw, 1)
        else:
            visual_img = frame_draw

        cv2.imshow('frame', visual_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame_draw)

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')
    pass


def main():
    # model files check and download
    check_and_download_models(
        FACE_DET_WEIGHT_PATH, FACE_DET_MODEL_PATH, FACE_DET_REMOTE_PATH
    )
    check_and_download_models(
        FACE_LM_WEIGHT_PATH, FACE_LM_MODEL_PATH, FACE_LM_REMOTE_PATH
    )
    if args.include_iris:
        check_and_download_models(
            IRIS_LM_WEIGHT_PATH, IRIS_LM_MODEL_PATH, IRIS_LM_REMOTE_PATH
        )
    if args.include_head_pose:
        check_and_download_models(
            HEAD_POSE_WEIGHT_PATH, HEAD_POSE_MODEL_PATH, HEAD_POSE_REMOTE_PATH
        )
    check_and_download_models(
        GAZE_WEIGHT_PATH, GAZE_MODEL_PATH, GAZE_REMOTE_PATH
    )
    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
