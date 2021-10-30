# keras-retinanet-mibi #

fizyr/keras-retinanet をベースとして MIBI タスクの為にカスタマイズしたもの。

- MIBI タスクのためのデータセットや分類クラス定義を追加。
- 入力画像形式をグレースケールに変更し、対応する入力テンソル形状を調整。
- EarlyStopping を無効化。
- MIBI タスクに必要な評価を行うために評価用スクリプトを調整。

ネットワーク（モデル）には変更を加えていない。

### install ###

Cython コードをコンパイルするために
```
python setup.py build_ext --inplace
```

### トレーニングデータセット ###

以下の場所にそれぞれ PASCAL 形式のディレクトリ構成でトレーニングとバリデーションのデータセットを用意する。

- ../pascal-mibi-both-dataset-notest
- ../pascal-mibi-delay-dataset-notest
- ../pascal-mibi-early-dataset-notest

例えば pascal-mibi-both-dataset-notest は以下のような構成とする。
```
├── Annotations
│   ├── delay_P001.xml
│   ├── ...
│   ├── delay_mag_P001.xml
│   ├── ...
│   ├── early_P001.xml
│   ├── ...
│   ├── early_mag_P001.xml
│   ├── ...
├── ImageSets
│   └── Main
│       ├── test.txt    # 空っぽ
│       ├── train.txt   # トレーニングデータの名前を列挙したファイル
│       └── val.txt     # バリデーションデータの名前を列挙したファイル
├── JPEGImages
│   ├── delay_P001.png
│   ├── ...
│   ├── delay_mag_P001.png
│   ├── ...
│   ├── early_P001.png
│   ├── ...
│   ├── early_mag_P001.png
│   ├── ...
```

### テストデータセット ###

以下の場所にそれぞれ PASCAL 形式のディレクトリ構成でテストデータのみのデータセットを用意する。

- ../pascal-mibi-both-dataset-test-all
- ../pascal-mibi-delay-dataset-test-all
- ../pascal-mibi-early-dataset-test-all

例えば pascal-mibi-both-dataset-test-all は以下のような構成とする。
```
├── Annotations
│   ├── delay_P001.xml
│   ├── ...
│   ├── delay_mag_P001.xml
│   ├── ...
│   ├── early_P001.xml
│   ├── ...
│   ├── early_mag_P001.xml
│   ├── ...
├── ImageSets
│   └── Main
│       ├── test.txt        # 元々アノテーションが無かったデータを含まないリスト
│       ├── test_all.txt    # 使用する全てのテストデータの名前を列挙したファイル
│       ├── train.txt       # 空っぽ
│       └── val.txt         # 空っぽ
├── JPEGImages
│   ├── delay_P001.png
│   ├── ...
│   ├── delay_mag_P001.png
│   ├── ...
│   ├── early_P001.png
│   ├── ...
│   ├── early_mag_P001.png
│   ├── ...
```

### トレーニング ###

`./train-mibi-serial.sh` を実行する。

三種のデータセットそれぞれに対して学習が実行され、結果は以下のディレクトリに格納される。

- ./train-mibi-both/
- ./train-mibi-delay/
- ./train-mibi-early/

例えば ./train-mibi-both/ の中身は以下のようになる。

```
├── logs
│   └── events.out.tfevents....     # TensorBoard のログ
├── snapshots
│   ├── resnet50_pascal-mibi_01.h5  # 学習済みモデル
│   ├── ...
├── train.err                       # 学習中の標準エラーログ
└── train.log                       # 学習中の標準出力ログ
```

### 評価 ###

前項のトレーニングを実行するか、またはトレーニング済みモデル（h5ファイル）を
https://github.com/Tc-MIBI/keras-retinanet-mibi/releases
からダウンロードし ./snapshots/ に配置する。

```
snapshots/
├── both-best-performing.h5
├── delay-best-performing.h5
└── early-best-performing.h5
```

`./evaluate-mibi-serial.sh` を実行する。

３つのモデル x 三種のデータセット x バリデーションデータセットとテストデータセットに対し、以下の評価を行う。

- FROC
- mAP
- 候補バウンディングボックスのスコア列挙と画像作成
- スコア基準値による有無判定

評価結果は ./evaluate-mibi/ に格納される。
