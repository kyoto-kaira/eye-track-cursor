# 視線推定ライブラリの動かし方(backend/ax_gaze_estimation.py)
## 前提
・Linux(or Ubuntu)のコマンドをルートディレクトリ上で動かしている  
・pythonをコマンド上にてインストール済(バージョンは3.12.4)  
## 手順
1\. 以下のコードを動かす
``` 
cd backend  
python -m venv .....#(仮想環境の名前で、好きなもので大丈夫です)  
source ...../bin/activate #(....:仮想環境の名前)  
pip install -r requirements.txt  
``` 
2\. backend/data/Images/~~~/input（好きに付けたディレクトリ名）上に５枚以上の画像を入れる  
この時、  
・PCの４隅を見たときの画像を１枚ずつ入れ、名前にも(top_left,top_right,down_left,down_rightを入れる)  
・その他の画像には、top_left,top_right,down_left,down_rightのいずれの名前も含まないようにする   
　（backend/data/Images/FaceSamples/inputが参考になる）  
3\. 以下のコードを動かす  
```
python3 ax_gaze_estimation.py --input data/Images/~~~/input　#(画像を入れたディレクトリ名)
```
これでbackend/data/Images/~~~/with_gazeディレクトリ下に視線の向きを付けた画像が渡され、  
backend/data/Images/~~~/result/gaze_point.jpg上に４隅を見たときの画像を元にした視線の位置をプロット
## 補足（手順）
X:X座標が大きくなるほど画面の右にいく（０～１２８０）  
Y:Y座標が大きくなるほど、画面の下に目線がいく（０～６４０）    
赤線：スクリーン（４隅を見たときの視線座標（２D）から求めたスクリーン（歪四角形）をアフィン変換によって綺麗な長方形に変化）  
## フロントとの接続方法
以下backend/ax_gaze_estimation.py中の関数について語る。  
1\. calibrate(calibration_images:画像のリスト,screen_positions:画像に対応したスクリーン上での位置座標)で、実際の視点座標(スクリーン上)を求めるのに必要な射影行列Mを取得  
2\. infer_gaze_position(img:画像,screen_size:スクリーン座標系の長さ([x座標,y座標]),M:1\.で取得した行列)で、実際の視点座標(スクリーン上)をnp.ndarray([x座標,y座標])で取得  
3\. draw_gaze_vector(img:画像)で視線ベクトルの描写付きの画像をnp.ndarrayとして返す
## 補足（フロントとの接続方法）
X:X座標が大きくなるほど画面の右にいく（０～１２８０）  
Y:Y座標が大きくなるほど、画面の下に目線がいく（０～６４０）   
{~(ファイル名:str):~~(スクリーン座標系における視点座標:np.ndarray)}において、np.ndarrayの方はnp.array([X座標,Y座標])となっている。