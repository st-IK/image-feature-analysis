counterfactual_degradation_analysis.py
=======================================

最終的な寄与ヒートマップを利用して, 
「寄与の低い領域」を段階的に劣化させた画像を作り, 
元画像と比較する. 

目的
----
例えば寄与ヒートマップ H(x,y) に対して, 

    H < threshold

の領域を, 

- blur
- noise
- desaturation
- blur + noise
- desaturation + noise

などで劣化させる. 

その後, 元画像と劣化画像を同じモデルに入力し, 

    target class probability
    top-1 class
    top-k probabilities

などの変化を比較する. 

重要な解釈
----------
「低寄与領域を壊しても予測が変わらない」なら, 
その領域が現在のモデル判断に必要ない可能性を支持する. 

ただし, 
「予測が変わらなかった」=「その特徴をモデルが見ていない」
とは限らない. 

また, 低寄与領域を人工的に変形することで
OOD（学習分布外）入力になる可能性があるため, 
複数の劣化方法と複数の閾値を比較することを推奨する. 

必要:
    pip install torch torchvision pillow numpy pandas matplotlib

実行:
    python counterfactual_degradation_analysis.py test.jpg interaction_analysis/region_contributions.csv