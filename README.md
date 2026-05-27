# mask2dataset (Mask2Former ADE20K)

360°の画像フォルダ or 360°動画から、Mask2Former(ADE20K)でセグメンテーションした学習用データセットを生成します。

GUIは1ウィンドウで、
プレビュー1(正面=yaw指定) → 切り出し設定 → プレビュー2(切り出し結果+セグ表示切替) → 実行
の流れです。

## Requirements

- `ffmpeg` (v360フィルタが有効なもの)
- Python 3.10+（`venv` が使えること）
- GUI用: Tkinter（Linuxだと `python3-tk` が別パッケージのことが多い）
- Pythonパッケージ: `pip install -r requirements.txt`

### OS packages install example (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y \
  ffmpeg \
  git \
  python3 python3-venv python3-pip python3-tk \
  libgl1 libglib2.0-0
```

`ffmpeg` が v360 を持っているか確認:

```bash
ffmpeg -hide_banner -filters | grep v360
```

## Quick Setup (venv)

```bash
git clone git@github.com:shimpeisasaki/mask2dataset.git
cd mask2dataset
bash scripts/setup_venv.sh .venv
source .venv/bin/activate
```

補足:

- `TORCH_CHANNEL` で torch 取得先を変更可能です（例: `cu128`, `cu126`, `cu124`, `cpu`）

```bash
TORCH_CHANNEL=cu128 bash scripts/setup_venv.sh .venv
```

## Run

```bash
python3 -m src.app
```

※ `python` コマンドが使える環境では `python -m src.app` でもOKです。

## Dataset Output (MMSegmentation style)

出力フォルダ配下に以下を作成します:

```
OUTPUT_DIR/
├── img_dir/
│   ├── train/
│   └── val/
└── ann_dir/
    ├── train/
    └── val/
```

### 重要: ラベルPNGは「クラスID画像」

- `ann_dir/` に保存されるPNGは、見た目が真っ黒〜暗いグレーになります。
- これは **RGBで色を塗った画像ではなく**、ピクセル値そのものがクラスIDです。
- 値は `0..7` と `255(ignore)` のみを使います。

## Class Mapping (ADE20K -> 0..7)

クラス統合は GUI では選ばず、設定ファイルを自動で読み込みます:

- `config/class_map.yaml`

例:

```yaml
ignore_id: 255
unmapped: 7
classes:
  0:
    name: sky
    ade20k: [sky]
  6:
    name: person
    ade20k: [person]
  7:
    name: unlabeled
    ade20k: [windowpane, curtain, cushion, lamp]
```
