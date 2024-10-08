# 視線推定ライブラリの動かし方(backend/ax_gaze_estimation.py)
## 前提
・Linux(or Ubuntu)のコマンドをルートディレクトリ上で動かしている  
・pythonをコマンド上にてインストール済
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
　（backend/data/Images/example/FaceSamplesが参考になる）  
3\. 以下のコードを動かす  
```
python3 ax_gaze_estimation.py --input data/Images/~~~　#(画像を入れたディレクトリ名)
```
これでbackend/data/Images/~~~_resultsディレクトリ下に視線の向きを付けた画像が渡され、  
backend/data/results/gaze_point.jpg上に４隅を見たときの画像を元にした視線の位置をプロット
## 補足
X:X座標が大きくなるほど画面の右にいく（０～１２８０）  
Y:Y座標が大きくなるほど、画面の下に目線がいく（０～６４０）    
赤線：スクリーン（４隅を見たときの視線座標（２D）から求めたスクリーン（歪四角形）をアフィン変換によって綺麗な長方形に変化）  
