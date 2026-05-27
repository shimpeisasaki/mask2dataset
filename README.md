# mask2dataset (Mask2Former ADE20K)

360°の画像フォルダ or 360°動画から、Mask2Former(ADE20K)でセグメンテーションした学習用データセットを生成します。

GUIは1ウィンドウで、
プレビュー1(正面=yaw指定) → 切り出し設定 → プレビュー2(切り出し結果+セグ表示) → 実行

## Requirements

- `ffmpeg` (v360フィルタが有効なもの)
- Python 3.10+（`venv` が使えること）
- GUI用: Tkinter（Linuxだと `python3-tk` が別パッケージのことが多い）
- Pythonパッケージ: `pip install -r requirements.txt`

### 実行環境の注意

- **初回実行時は Mask2Former のモデルをダウンロード**します（ネット接続と十分なディスク容量が必要）。
- GPUがある場合は自動でCUDAを使います（CPUでも動きますがかなり遅いです）。

### 簡易セルフチェック

```bash
ffmpeg -hide_banner -filters | grep v360
python -c "import cv2, PIL, yaml, numpy, tkinter"
python -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available())"
```

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

- `TORCH_CHANNEL` で torch 取得先を変更可能（例: `cu128`, `cu126`, `cu124`, `cpu`）

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

### train/val の分割

- 分割は `val_ratio=0.2` のランダム（seed固定）です。
- **同一の元画像/元フレームから生成される複数ビューは、同じ split (train/val) に入る** ようにしています（リーク防止）。
```

### 重要: ラベルPNGは「クラスID画像」

- `ann_dir/` に保存されるPNGは、見た目が真っ黒〜暗いグレーになります。
- これは **RGBで色を塗った画像ではなく**、ピクセル値そのものがクラスIDです。
- 値は `0,1,2,,,` と `255(ignore)` のみを使います。

## Class Mapping (Prompts)

クラスは `config/class_map.yaml` で設定します。

- `prompts:` に自由なテキストを入れます（複数可）。
- `feat/sam` では FastSAM が生成するマスクを、CLIP でプロンプトに近いクラスへ割り当てます。

例:

```yaml
ignore_id: 255
unmapped: 6
classes:
  0:
    name: sky
    prompts: ["sky", "blue sky"]
  5:
    name: person
    prompts: ["person", "human"]
  6:
    name: unlabeled
    prompts: ["window", "curtain", "lamp"]
```
