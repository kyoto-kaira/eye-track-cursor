# 目線で操るマウスカーソル
## 動かし方

1. uvを入れる（入っている人はこの手順は飛ばしてください）

```shell
# macOS または Linuxの人
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windowsの人
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. cmakeを入れる（入っている人はこの手順は飛ばしてください）

```shell
# macOS または Linuxの人
brew install cmake

# Windowsの人
# https://qiita.com/matskeng/items/c466c4751e1352f97ce6
# を参考にして入れる
```

※インストールが終わったら、ターミナルを再起動してください

2. リポジトリをクローンする

```shell
# 任意のディレクトリに移動して以下を実行すると、このリポジトリがその下に作成されます
git clone https://github.com/kyoto-kaira/eye-track-cursor.git

# 作成されたリポジトリに移動する
cd eye-track-cursor
```

3. Python3.10の仮想環境を作成する

```shell
# 以下を実行すると、現在のディレクトリに.venv/という名前の仮想環境が作成されます
uv venv -p 3.10.15
```

4. 仮想環境を有効化する

```shell
# macOS または Linuxの人
source .venv/bin/activate

# Windowsの人 (バックスラッシュです!!)
.venv\Scripts\activate

# ディレクトリ名の左側に (eye-track-cursor) と表示されれば成功です
```

5. 依存パッケージをインストールする

```shell
# dlibのインストールに時間がかかるので気長にお待ちください
uv pip install -r requirements.txt
```

6. アプリを起動する

```shell
streamlit run app.py
```
