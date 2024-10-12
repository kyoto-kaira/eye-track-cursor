# 視線推定ライブラリの動かし方(backend/ax_gaze_estimation.py)
## 前提
・Linux(or Ubuntu)のコマンドをルートディレクトリ上で動かしている  
・pythonをコマンド上にてインストール済(バージョンは3.12.4)  
・インストールしてほしいものはrequirements.txtに書かれてある。
## フロントとの接続方法(ルートディレクトリ上でしか使えない)
2\. ～4\. で述べる関数は全てbackend/ax_gaze_estimation.py中にあり、backend.ax_gaze_estimationからインポートできる。  
ルートディレクトリ上で1\. を行った後に  
```
python -m backend.ax_gaze_estimation
```
と動かせば2\. ～4\. が動く。  
1\. ルートディレクトリ上で以下のコードを動かし、環境構築をする。
```
python -m venv venv  
source venv/bin/activate  
pip install -r backend/requirements.txt  
```
2\. calibrate(calibration_images:画像のリスト,screen_positions:画像に対応したスクリーン上での位置座標)で、実際の視点座標(スクリーン上)を求めるのに必要な射影行列Mを取得  
3\. infer_gaze_position(img:画像,screen_size:スクリーン座標系の長さ([x座標,y座標]),M:1\.で取得した行列)で、実際の視点座標(スクリーン上)をnp.ndarray([x座標,y座標])で取得  
4\. draw_gaze_vector(img:画像)で視線ベクトルの描写付きの画像をnp.ndarrayとして返す
## 補足（フロントとの接続方法）
X:X座標が大きくなるほど画面の右にいく（０～１２８０）  
Y:Y座標が大きくなるほど、画面の下に目線がいく（０～６４０）   
