# 視線推定ライブラリの動かし方(backend/ax_gaze_estimation.py)
## 前提
・Linux(or Ubuntu)のコマンドをルートディレクトリ上で動かしている  
・pythonをコマンド上にてインストール済(バージョンは3.12.4)  
## フロントとの接続方法
以下backend/ax_gaze_estimation.py中の関数について語る。  
1\. calibrate(calibration_images:画像のリスト,screen_positions:画像に対応したスクリーン上での位置座標)で、実際の視点座標(スクリーン上)を求めるのに必要な射影行列Mを取得  
2\. infer_gaze_position(img:画像,screen_size:スクリーン座標系の長さ([x座標,y座標]),M:1\.で取得した行列)で、実際の視点座標(スクリーン上)をnp.ndarray([x座標,y座標])で取得  
3\. draw_gaze_vector(img:画像)で視線ベクトルの描写付きの画像をnp.ndarrayとして返す
## 補足（フロントとの接続方法）
X:X座標が大きくなるほど画面の右にいく（０～１２８０）  
Y:Y座標が大きくなるほど、画面の下に目線がいく（０～６４０）   
{~(ファイル名:str):~~(スクリーン座標系における視点座標:np.ndarray)}において、np.ndarrayの方はnp.array([X座標,Y座標])となっている。