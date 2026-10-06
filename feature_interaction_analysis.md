feature_interaction_analysis.py
================================

regions.csv / regions.json を入力として, 
画像特徴量とモデル予測への寄与の関係を解析する.

目的
----
1. 各特徴量と importance の関係を推定
2. 特徴量間の相互作用を推定
3. 予測に対する寄与を統計モデルで評価
4. 結果をCSV / JSON / PNGで保存

重要な注意
----------
regions.csv に含まれる特徴量は「画像から観測された候補特徴量」であり, 
ニューラルネットワーク内部の特徴量そのものではない.

また, 単純相関だけでは「AIがその特徴を使っている」ことは証明できない.
本スクリプトでは, 
    importance ~ feature + feature_i * feature_j + control variables
という局所的な統計モデルを使い, 関連と相互作用を調べる.

必要:
    pip install pandas numpy scipy statsmodels matplotlib

実行:
    python feature_interaction_analysis.py feature_analysis/regions.csv