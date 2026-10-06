# Adaptive Feature Occlusion Analyzer v3

## 目的

学習済み画像分類モデルに対して, **四分木（quadtree）方式で画像領域を再帰的に遮蔽し, モデル出力の変化から重要領域を推定する**. 

v3では, 遮蔽画像のバッチ処理, 数値ヒートマップの保存, 詳細な解析メタデータの保存などを実装している. 

---

## v3の主な変更点

### 1. 遮蔽画像をバッチ化してモデルへ一括投入

- 複数の遮蔽画像をまとめてモデルへ入力
- CPU環境でも推論呼び出し回数を削減できる
- `--batch-size` で一度に処理する画像数を調整可能

### 2. ヒートマップの数値をテキストとして保存

各ヒートマップについて, 以下の形式で保存する. 

#### `heatmap_*.csv`

画像と同じ `H × W` の2次元配列として保存する. 

#### `heatmap_*_long.csv`

```text
x,y,value
```

形式で保存する. 

#### `regions.csv`

四分木探索によって実際に評価された各領域の生データを保存する. 

#### `regions.json`

`regions.csv` と同じ領域情報をJSON形式で保存する. 

### 3. PNGヒートマップと元画像オーバーレイを出力

各特徴量およびimportanceについて, 

- ヒートマップ単体
- 元画像とのオーバーレイ

をPNGとして出力する. 

### 4. `metadata.json` を出力

以下のような解析条件・実行情報を保存する. 

- 使用モデル
- 使用デバイス
- 解析対象クラス
- threshold
- 最大深度
- 最小領域サイズ
- batch size
- 遮蔽方式
- 評価領域数
- 推論画像数
- 推論回数
- 処理時間
- 使用した画像特徴量

これにより, **後から解析条件を再現・比較できる**ようにする. 

### 5. 四分木による探索構造を維持

最初に画像全体を評価し, importanceがthreshold以上となった領域だけを4分割してさらに詳しく評価する. 

```text
画像全体
    │
    ├── importance < threshold
    │       └── これ以上分割しない
    │
    └── importance >= threshold
            │
            ├── 4領域へ分割
            │
            └── 重要な領域をさらに分割
```

これにより, 重要度の低い領域に対して無駄に細かい解析を行わず, 重要領域を重点的に探索する. 

---

## 現在の候補画像特徴量

各領域について, 以下の画像特徴量を計算する. 

- `mean_luminance`
  - 平均輝度

- `mean_hue`
  - 平均色相

- `mean_saturation`
  - 平均彩度

- `edge_density`
  - エッジ密度

- `texture_std`
  - テクスチャの標準偏差

- `high_frequency_energy`
  - 高周波成分のエネルギー

### 注意

これらは画像から直接計算した**「候補特徴量」**であり, ニューラルネットワーク内部の特徴量を直接読み出しているわけではない. 

したがって, 

> 「ニューラルネットワークが内部でこの特徴量を使用している」

ことを直接示すものではない. 

将来的には, 

```text
画像特徴量
    ↓
特徴量への介入操作
    ↓
AIモデルへ入力
    ↓
モデル出力の変化
    ↓
特徴量寄与・特徴量間相互作用の推定
```

へ拡張することを想定している. 

---

# 使用方法

## 最小構成

```bash
python adaptive_feature_occlusion_v3.py image.jpg
```

デフォルトでは以下の条件で実行する. 

```text
model       : resnet18
threshold   : 0.10
max-depth   : 4
min-size    : 32
batch-size  : 8
occlusion   : mean
output      : feature_analysis
```

---

## CPUで軽量に実行

```bash
python adaptive_feature_occlusion_v3.py image.jpg --max-depth 3 --min-size 32 --threshold 0.10 --batch-size 8
```

探索深度を3に抑えることで, 詳細度と計算量を抑えられる. 

---

## より詳細に解析

```bash
python adaptive_feature_occlusion_v3.py image.jpg --max-depth 5 --min-size 16 --threshold 0.05 --batch-size 8 --occlusion blur --out result
```

より細かい領域まで探索し, 遮蔽方法としてぼかしを使用する. 

---

# オプション

| オプション | デフォルト | 説明 |
|---|---:|---|
| `image` | 必須 | 入力画像 |
| `--out` | `feature_analysis` | 出力フォルダ |
| `--model-type` | `torchvision` | 使用するAIの種類 |
| `--model-name` | `resnet18` | 使用するモデル |
| `--target-class` | `None` | 解析対象クラスID |
| `--threshold` | `0.10` | この値以上のimportanceを持つ領域を再分割 |
| `--max-depth` | `4` | 四分木の最大深度 |
| `--min-size` | `32` | 分割する最小領域サイズ[pixel] |
| `--batch-size` | `8` | 1回に処理する画像数 |
| `--occlusion` | `mean` | 遮蔽方法 |

### `--model-type`

現在対応しているモデル種別：

```text
torchvision
yolo
```

### `--model-name`

`torchvision` の場合はモデル名を指定する. 

例：

```bash
--model-name resnet18
```

```bash
--model-name resnet50
```

```bash
--model-name efficientnet_b0
```

```bash
--model-name vit_b_16
```

YOLOの場合はモデルファイルを指定する. 

```bash
--model-type yolo --model-name best.pt
```

### `--target-class`

解析対象とするクラスIDを指定する. 

例えばYOLOでクラス0を解析する場合：

```bash
--target-class 0
```

### `--threshold`

重要領域として扱うimportanceの閾値. 

```bash
--threshold 0.05
```

なら, 

```text
importance >= 0.05
```

の領域をさらに4分割する. 

**小さくするほど探索領域が増え, 計算量が増加する. **

### `--max-depth`

四分木探索の最大深度. 

```bash
--max-depth 5
```

とすると, より細かい領域まで解析する. 

### `--min-size`

これ以上小さい領域には分割しない. 

```bash
--min-size 16
```

なら, 16 pixel以下の領域では分割を停止する. 

### `--batch-size`

モデルに一度に投入する遮蔽画像の枚数. 

```bash
--batch-size 8
```

などと指定する. 

GPUメモリに余裕がある場合は大きくできるが, CPUでは必ずしも大きくすれば高速になるとは限らない. 

### `--occlusion`

遮蔽方法を指定する. 

使用可能な方式：

```text
mean
gray
blur
```

#### mean

画像全体の平均RGBで対象領域を塗りつぶす. 

```bash
--occlusion mean
```

#### gray

対象領域をRGB=127の灰色で置換する. 

```bash
--occlusion gray
```

#### blur

対象領域をGaussian Blurでぼかす. 

```bash
--occlusion blur
```

---

# 出力ファイル

例えば, 

```bash
--out result
```

とした場合, 

```text
result/
├── original.png
│
├── heatmap_importance.png
├── heatmap_importance_overlay.png
├── heatmap_importance.csv
├── heatmap_importance_long.csv
│
├── heatmap_mean_luminance.png
├── heatmap_mean_luminance_overlay.png
├── heatmap_mean_luminance.csv
├── heatmap_mean_luminance_long.csv
│
├── heatmap_mean_hue.*
├── heatmap_mean_saturation.*
├── heatmap_edge_density.*
├── heatmap_texture_std.*
├── heatmap_high_frequency_energy.*
│
├── regions.csv
├── regions.json
└── metadata.json
```

という構成になる. 

---

# 実行例

```bash
python adaptive_feature_occlusion_v3.py test.jpg --max-depth 3 --min-size 32 --threshold 0.10 --batch-size 8
```

YOLOを使用する場合：

```bash
python adaptive_feature_occlusion_v3.py test.jpg --model-type yolo --model-name best.pt --target-class 0 --max-depth 5 --min-size 16 --threshold 0.05 --occlusion blur --out result
```

---

## 必要ライブラリ

基本的な分類モデルを使用する場合：

```bash
pip install torch torchvision pillow numpy matplotlib
```

YOLOを使用する場合は追加で：

```bash
pip install ultralytics
```

---

# 解析の考え方

このプログラムで得られる `importance` は, 

$
I(R)=S_{\mathrm{original}}-S_{\mathrm{occluded}}(R)
$

として考えることができる. 

つまり, 領域 \(R\) を遮蔽したことでAIの対象クラスに対するスコアがどれだけ低下したかを表す. 

したがって, 

- **importanceが大きい**
  → その領域を遮蔽するとAIの判断が大きく変化する

- **importanceが小さい**
  → その領域を遮蔽してもAIの判断はあまり変化しない

という解釈になる. 

ただし, これは**因果的に「AIがその部分だけを見ている」と証明するものではなく, 遮蔽という介入に対するモデル出力の感度**として解釈するのが適切である. 
