import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
from backend.ax_gaze_estimation import (
    draw_gaze_vector,
    calibrate,
    infer_gaze_position
)

# セッションステートの初期化
if 'calibration_data' not in st.session_state:
    st.session_state.calibration_data = []

if 'points' not in st.session_state:
    st.session_state.points = []

if 'cap' not in st.session_state:
    st.session_state.cap = None

# レイアウトの設定
st.title("カメラキャリブレーションとリアルタイムプロットアプリ")

tabs = st.tabs(["キャリブレーション", "メイン"])

# キャリブレーションタブ
with tabs[0]:
    st.header("キャリブレーション")
    camera_input = st.camera_input("カメラから画像を取得")

    if camera_input:
        image = Image.open(camera_input)
        image = draw_gaze_vector(np.array(image))
        st.image(image, caption="取得した画像", use_column_width=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("左上"):
                st.session_state.calibration_data.append({
                    'type': '左上',
                    'image': camera_input
                })
                st.success("左上の画像を保存しました。")
        with col2:
            if st.button("右上"):
                st.session_state.calibration_data.append({
                    'type': '右上',
                    'image': camera_input
                })
                st.success("右上の画像を保存しました。")
        with col3:
            if st.button("左下"):
                st.session_state.calibration_data.append({
                    'type': '左下',
                    'image': camera_input
                })
                st.success("左下の画像を保存しました。")
        with col4:
            if st.button("右下"):
                st.session_state.calibration_data.append({
                    'type': '右下',
                    'image': camera_input
                })
                st.success("右下の画像を保存しました。")

    st.subheader("保存されたキャリブレーションデータ")
    for idx, data in enumerate(st.session_state.calibration_data):
        st.write(f"{idx+1}. {data['type']} の画像")
        st.image(data['image'], use_column_width=True)

# メインタブ
with tabs[1]:
    st.header("リアルタイム2次元座標プロット")
    
    # プロット間隔の設定
    interval = st.slider("フレーム取得間隔（秒）", min_value=0.1, max_value=5.0, value=1.0, step=0.1)


    # 仮のフレーム処理関数
    def process_frame(frame):
        """
        フレームを処理して2次元座標を返す関数のサンプル。
        実際の処理内容に応じて実装してください。
        """
        # 例として、フレームの範囲内でランダムな座標を返す
        print(type(frame), frame.shape)
        print(M)
        if M is not None:
            try:
                screen_position = infer_gaze_position(frame, (1920, 1080), M)
            except:
                screen_position =(0,0)
            return screen_position
        else:
            raise ValueError("NOT キャリブ")
#        h, w, _ = frame.shape
#        x = np.random.randint(0, w)
#        y = np.random.randint(0, h)
#        return (x, y)

    # カメラの初期化
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)

    cap = st.session_state.cap

    if not cap.isOpened():
        st.error("カメラが開けません。")
    else:
        # 実行中かどうかのフラグ
        if 'running' not in st.session_state:
            st.session_state.running = False

        start_button, stop_button = st.columns(2)

        with start_button:
            if st.button("開始"):
                st.session_state.running = True
                # pointsをクリアする
                st.session_state.points.clear()

        with stop_button:
            if st.button("停止"):
                st.session_state.running = False
                # pointsをクリアする
                st.session_state.points.clear()

                # カメラを解放
                if cap:
                    cap.release()
                    st.session_state.cap = None

        placeholder = st.empty()

        if st.session_state.running:
            while st.session_state.running:
                # キャリブレーション
                # calibration_images = [data['image'] for data in st.session_state.calibration_data]
                # ndarrayを取り出すように修正
                calibration_images = [np.array(Image.open(data['image'])) for data in st.session_state.calibration_data]
                screen_positions = [(0, 0), (1280, 0), (0, 720), (1280, 720)]
                if len(calibration_images) == 4:
                    M = calibrate(calibration_images, screen_positions)
                else:
                    M = None

                current_time = time.time()
                if 'last_capture_time' not in st.session_state:
                    st.session_state.last_capture_time = 0

                if current_time - st.session_state.last_capture_time >= interval:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("フレームを取得できませんでした。")
                        st.session_state.running = False
                        break

                    # フレームをBGRからRGBに変換
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 2次元座標を取得
                    coords = process_frame(frame)
                    st.session_state.points.append(coords)

                    # 画像に線をプロット
                    if len(st.session_state.points) > 1:
                        for i in range(1, len(st.session_state.points)):
                            cv2.line(frame_rgb, st.session_state.points[i-1], st.session_state.points[i], (255, 0, 0), 2)

                    # 画像を表示
                    placeholder.image(frame_rgb, channels="RGB")

                    st.session_state.last_capture_time = current_time

                # ストリームリットの更新待ち
                time.sleep(0.1)

        # セッション終了時にカメラを解放（念のため）
        def release_camera():
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None

        # Streamlitのセッション終了フックを使用
        # 現在のStreamlitでは、セッション終了フックが提供されていないため、
        # 代替としてキャッシュや他の方法を検討する必要があります。
        # ここでは、停止ボタンでカメラを解放するようにしています。

