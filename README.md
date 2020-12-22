# Requirement
tensorflow==2.3.0
keras==2.4.3
# Installation
Anaconda上でpython3.6の環境にkeras-gpuをインストールした環境で動作確認してます. 
```bash
conda install keras-gpu
```
# Usage
```bash
python train.py
```

# Note
* 勝率の報告はLINE Notifyを使用しています. 
* 環境変数:LINE_API_TOKENに設定されたtokenを利用して通知を送ってます. 
* 不要な場合はtrain.pyにあるsend_result2lineの呼び出しを削除してください. 
* 環境については時間があるときにもう少し丁寧まとめます. 

メモリの実装は下記サイトを参考にしました. 

[【強化学習】Keras-rlでRainbowを実装/解説](https://qiita.com/pocokhc/items/fc00f8ea9dca8f8c0297)
