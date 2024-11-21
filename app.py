import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from backend.ax_gaze_estimation import calibrate, draw_gaze_vector, infer_gaze_position

# Streamlitのページ設定
st.set_page_config(page_title="目線で操るマウスカーソル", layout="wide", page_icon="👀")

# セッションステートの初期化
if "calibration_images" not in st.session_state:
    st.session_state.calibration_images = {
        "左上": None,
        "右上": None,
        "左下": None,
        "右下": None,
    }

if "screen_positions" not in st.session_state:
    st.session_state.screen_positions = {
        "左上": None,
        "右上": None,
        "左下": None,
        "右下": None,
    }

if "M" not in st.session_state:
    st.session_state.M = None

if "gaze_points" not in st.session_state:
    st.session_state.gaze_points = []

if "video_running" not in st.session_state:
    st.session_state.video_running = False


# キャリブレーション用のラベルと対応するスクリーン位置
def get_calibration_labels(screen_size: tuple) -> dict:
    height, width = screen_size
    CALIBRATION_LABELS = {
        "左上": (width, 0),
        "右上": (0, height),
        "左下": (width, height),
        "右下": (0, 0),
    }
    return CALIBRATION_LABELS


# タブの作成
tab1, tab2 = st.tabs(["キャリブレーション", "メイン"])

with tab1:
    st.header("キャリブレーション画面")
    st.write("カメラに向かって各ボタンを押してください。")

    # カメラから画像を取得
    camera_input = st.camera_input("カメラ画像を取得")

    cols = st.columns(2)
    with cols[0]:
        if st.button("左上"):
            if camera_input is not None:
                image = Image.open(camera_input)
                image_np = np.array(image)
                image_with_gaze = draw_gaze_vector(image_np)
                CALIBRATION_LABELS = get_calibration_labels(image_np.shape[:2])
                st.image(image_with_gaze, channels="RGB")
                st.session_state.calibration_images["左上"] = image_np
                st.session_state.screen_positions["左上"] = CALIBRATION_LABELS["左上"]
                st.success("左上のキャリブレーションポイントを保存しました。")
            else:
                st.error("カメラ画像が取得できませんでした。")
    with cols[1]:
        if st.button("右上"):
            if camera_input is not None:
                image = Image.open(camera_input)
                image_np = np.array(image)
                image_with_gaze = draw_gaze_vector(image_np)
                CALIBRATION_LABELS = get_calibration_labels(image_np.shape[:2])
                st.image(image_with_gaze, channels="RGB")
                st.session_state.calibration_images["右上"] = image_np
                st.session_state.screen_positions["右上"] = CALIBRATION_LABELS["右上"]
                st.success("右上のキャリブレーションポイントを保存しました。")
            else:
                st.error("カメラ画像が取得できませんでした。")

    with cols[0]:
        if st.button("左下"):
            if camera_input is not None:
                image = Image.open(camera_input)
                image_np = np.array(image)
                image_with_gaze = draw_gaze_vector(image_np)
                CALIBRATION_LABELS = get_calibration_labels(image_np.shape[:2])
                st.image(image_with_gaze, channels="RGB")
                st.session_state.calibration_images["左下"] = image_np
                st.session_state.screen_positions["左下"] = CALIBRATION_LABELS["左下"]
                st.success("左下のキャリブレーションポイントを保存しました。")
            else:
                st.error("カメラ画像が取得できませんでした。")
    with cols[1]:
        if st.button("右下"):
            if camera_input is not None:
                image = Image.open(camera_input)
                image_np = np.array(image)
                image_with_gaze = draw_gaze_vector(image_np)
                CALIBRATION_LABELS = get_calibration_labels(image_np.shape[:2])
                st.image(image_with_gaze, channels="RGB")
                st.session_state.calibration_images["右下"] = image_np
                st.session_state.screen_positions["右下"] = CALIBRATION_LABELS["右下"]
                st.success("右下のキャリブレーションポイントを保存しました。")
            else:
                st.error("カメラ画像が取得できませんでした。")

    # キャリブレーションが完了したかチェック
    if all(v is not None for v in st.session_state.calibration_images.values()):
        st.success(
            "すべてのキャリブレーションポイントが保存されました。キャリブレーションを実行します。"
        )
        if st.button("キャリブレーション実行"):
            with st.spinner("キャリブレーション中..."):
                cal_imgs = st.session_state.calibration_images
                scr_pos = st.session_state.screen_positions
                calibration_images = [cal_imgs[k] for k in sorted(cal_imgs.keys())]
                screen_positions = [scr_pos[k] for k in sorted(scr_pos.keys())]
                try:
                    M, src, dst = calibrate(
                        calibration_images=calibration_images,
                        screen_positions=screen_positions,
                    )
                    print("====src====")
                    print(src)
                    print("====dst====")
                    print(dst)
                    print("====M====")
                    print(M)
                    # srcをplotlyで表示
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=[src[0, 0]],
                            y=[src[0, 1]],
                            mode="markers",
                            name="左上",
                            marker=dict(color="red", size=20),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[src[1, 0]],
                            y=[src[1, 1]],
                            mode="markers",
                            name="左下",
                            marker=dict(color="blue", size=20),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[src[2, 0]],
                            y=[src[2, 1]],
                            mode="markers",
                            name="右下",
                            marker=dict(color="green", size=20),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[src[3, 0]],
                            y=[src[3, 1]],
                            mode="markers",
                            name="右上",
                            marker=dict(color="orange", size=20),
                        )
                    )

                    fig.update_layout(
                        title="キャリブレーション位置", width=400, height=400
                    )
                    st.plotly_chart(fig)

                    st.session_state.M = M
                    st.success(
                        "キャリブレーションが完了しました。メイン画面で視線推定を開始できます。"
                    )
                except Exception as e:
                    st.error(f"キャリブレーション中にエラーが発生しました: {e}")

with tab2:
    if st.session_state.M is None:
        st.warning(
            "キャリブレーションが完了していません。キャリブレーションタブでキャリブレーションを行ってください。"
        )
    else:
        # Canvasの準備（Plotlyを使用）
        screen_size = (1920, 1080)  # 実際のスクリーンサイズに合わせて調整

        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(range=[0, screen_size[0]], autorange=False, title="X"),
            yaxis=dict(range=[0, screen_size[1]], autorange=False, title="Y"),
            title="視線位置",
            width=1920 // 2,
            height=1080 // 2,
        )
        # ステータス
        status_placeholder = st.empty()

        # Plotlyチャートのプレースホルダー
        chart_placeholder = st.empty()

        # 取得したカメラフレームを表示するプレースホルダー
        frame_placeholder = st.empty()

        # OpenCV VideoCaptureの初期化
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("カメラを開くことができませんでした。")
        else:
            col1, col2, col3 = st.columns(3)
            # 実行中フラグ
            if not st.session_state.video_running:
                start_button = col1.button("視線推定開始")
                if start_button:
                    st.session_state.video_running = True

            if st.session_state.video_running:
                stop_button = col2.button("視線推定停止")
                if stop_button:
                    st.session_state.video_running = False
                    cap.release()
                    cv2.destroyAllWindows()
                    st.rerun()

                # 視線ポイントのリセット
                if col3.button("視線ポイントのリセット"):
                    st.session_state.gaze_points = []

                # Plotlyチャートの初期化
                fig = go.Figure()
                fig.update_layout(
                    xaxis=dict(range=[0, screen_size[0]], autorange=False, title="X"),
                    yaxis=dict(range=[0, screen_size[1]], autorange=False, title="Y"),
                    title="視線位置",
                    width=1920 // 2,
                    height=1080 // 2,
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                # フレームの取得と処理
                while st.session_state.video_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("カメラからフレームを取得できませんでした。")
                        break

                    # BGRからRGBに変換
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 画像を表示
                    frame_with_gaze = draw_gaze_vector(frame_rgb)
                    frame_placeholder.image(frame_with_gaze, channels="RGB")

                    # 推論を実行
                    try:
                        screen_position = infer_gaze_position(
                            image=frame_rgb,
                            screen_size=screen_size,
                            M=st.session_state.M,
                        )
                        if screen_position is not None:
                            st.session_state.gaze_points.append(screen_position)
                            status_placeholder.success(
                                f"X = {screen_position[0]}, Y = {screen_position[1]}"
                            )
                    except Exception as e:
                        print("No face detected: ", e)
                        status_placeholder.error("顔をカメラに向けてください。")

                    if len(st.session_state.gaze_points) > 0:
                        # Plotlyチャートにポイントを追加
                        x_vals = [point[0] for point in st.session_state.gaze_points]
                        y_vals = [point[1] for point in st.session_state.gaze_points]

                        fig = go.Figure()
                        fig.update_layout(
                            xaxis=dict(
                                range=[0, screen_size[0]], autorange=False, title="X"
                            ),
                            yaxis=dict(
                                range=[0, screen_size[1]], autorange=False, title="Y"
                            ),
                            title="視線位置",
                            width=1920 // 2,
                            height=1080 // 2,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=y_vals,
                                mode="markers",
                                marker=dict(color="red", size=5),
                            )
                        )

                        chart_placeholder.plotly_chart(fig, use_container_width=True)

            cap.release()
            cv2.destroyAllWindows()
