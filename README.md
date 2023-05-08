# DPPH_analyzer

python 3.8を使用<br>
requirements.txtはwindowsでの仕様<br>


## ライブラリーのインストール


```
pip install -r requirements.txt
```


## Streamlitの起動

localではターミナルで<code>streamlit run main.py</code>を実行して動かす
```
streamlit run main.py
```

## モデルの作成について
1. poclab-web/features/dataに研究室での測定データと文献データを結合し,重複を除いたもの(dpph_lab_lit.csv)がある
2. これを/featuresdataprocess/fingerprint.pyに入力すると,Morgan fingerprintが
3. jupyter notebookのxgboost_fpDPPH.ipynbで入力すると、予測モデルのpickleファイルが得られる