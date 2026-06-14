# mask2dataset (Mask2Former ADE20K)

360度画像/動画から、Mask2Former(ADE20K)で推論したタイル画像とラベルJSONを生成するツールです。

GUIの基本フロー:
1. プレビュー1で入力確認
2. 切り出し設定を調整
3. プレビュー2で方向別確認
4. 実行

## What Changed (Performance Refactor)

今回のリファクタで、処理経路を高速化しました。

1. ffmpeg v360をメモリ直結化
: 一時PNGを大量に作らず、rawvideo pipeで一括投影
2. バッチ推論
: Mask2Formerを複数タイルまとめて推論
3. 生成時の不要処理削減
: 学習データ生成では、不要なプレビュー用オーバーレイ/内訳レポート生成をスキップ
4. ADE->dataset ID変換のLUT化
: 画素単位の変換を高速化
5. JSON出力をミニファイ
: 書き込みサイズとI/O時間を削減

## Requirements

1. ffmpeg (v360 filter有効)
2. Python 3.10+
3. Tkinter (Linuxでは `python3-tk` が別パッケージの場合あり)
4. `pip install -r requirements.txt`

初回実行時はモデルをダウンロードします。
GPUがある場合はCUDAを自動利用します。

## Quick Setup

```bash
git clone git@github.com:shimpeisasaki/mask2dataset.git
cd mask2dataset
bash scripts/setup_venv.sh .venv
source .venv/bin/activate
```

## Run

```bash
python3 -m src.app
```

## Output Format

現在の出力は以下です。

```text
OUTPUT_DIR/
└── images/
    ├── frame_xxxxxx_view.png
    ├── frame_xxxxxx_view.json
    └── ...
```

JSONはx-anylabeling/LabelMe系互換フォーマットです。

## Performance Tuning

### GUI側の推奨

1. 推論粗さ(px)を上げる (`2`, `4`, `8`, `16`)
2. 必要な方向だけチェックON
3. 出力サイズを必要最小限にする

### 環境変数

1. `MASK2DATASET_INFER_BATCH`
: 推論バッチサイズ (既定 `4`)

```bash
MASK2DATASET_INFER_BATCH=8 python3 -m src.app
```

2. `MASK2DATASET_POLYGON_BACKEND`
: ポリゴン化方式 (`fast` or `topology`, 既定 `fast`)

```bash
MASK2DATASET_POLYGON_BACKEND=topology python3 -m src.app
```

`fast` は速度重視、`topology` は境界共有の整合性重視です。

## Class Mapping

クラス定義は [config/new_class_map.yaml](config/new_class_map.yaml) を使います。

`ignore_id` は255固定、`unmapped` は未対応ADEクラスの行き先です。

## Repository Guide

主要な責務は次の通りです。

1. [src/gui.py](src/gui.py)
: GUIと実行制御
2. [src/pipeline.py](src/pipeline.py)
: 生成パイプライン本体
3. [src/v360.py](src/v360.py)
: ffmpeg v360投影（高速メモリ経路を含む）
4. [src/segmentation/mask2former.py](src/segmentation/mask2former.py)
: Mask2Former推論エンジン
5. [src/dataset/writer.py](src/dataset/writer.py)
: PNG/JSON出力
