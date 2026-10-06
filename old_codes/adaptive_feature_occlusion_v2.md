Adaptive Feature Occlusion Analyzer v2
=======================================

目的
----
学習済み画像分類モデルに対して、四分木（quadtree）方式で画像領域を
再帰的に遮蔽し、モデル出力の変化から重要領域を推定する。

v2 の主な変更点
----------------
1. 遮蔽画像をバッチ化してモデルへ一括投入
   - CPUでも推論呼び出し回数を減らせる
   - --batch-size で調整可能

2. ヒートマップの数値をテキストとして保存
   - heatmap_*.csv
       画像と同じ H x W の2次元配列
   - heatmap_*_long.csv
       x,y,value 形式
   - regions.csv
       四分木で評価した各領域の生データ
   - regions.json
       同じ情報をJSON形式で保存

3. PNGヒートマップと元画像オーバーレイを出力

4. metadata.json にモデル、条件、推論回数、処理時間などを保存

5. 四分木の探索構造を維持
   - importance >= threshold の領域だけ細分化

現在の候補画像特徴量
--------------------
- mean_luminance
- mean_hue
- mean_saturation
- edge_density
- texture_std
- high_frequency_energy

注意
----
これらは画像から計算した「候補特徴量」であり、
ニューラルネットワーク内部の特徴量を直接読み出しているわけではない。

将来は、
「特徴量を介入操作 → モデル出力変化」
へ拡張し、特徴量寄与・特徴量間相互作用を推定することを想定している。

使用例
------
最小構成：

    python adaptive_feature_occlusion_v2.py image.jpg

CPUで軽量に：

    python adaptive_feature_occlusion_v2.py image.jpg --max-depth 3 --min-size 32 --threshold 0.10 --batch-size 8

より詳細：

    python adaptive_feature_occlusion_v2.py image.jpg --max-depth 5 --min-size 16 --threshold 0.05 --batch-size 8 --occlusion blur --out result

必要ライブラリ
--------------
pip install torch torchvision pillow numpy matplotlib

実行例
```bash
python adaptive_feature_occlusion_v2.py test.jpg --max-depth 3 --min-size 32 --threshold 0.10 --batch-size 8
```