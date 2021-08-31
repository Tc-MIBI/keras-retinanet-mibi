# Explanation: English
# keras-retinanet-mibi #

This project was developed based on fizyr/keras-retinanet and modified it for MIBI.

### Training dataset ###

Prepare the training and validation datasets in the following locations, each in a PASCAL format directory structure.

- ../pascal-mibi-both-dataset-notest
- ../pascal-mibi-delay-dataset-notest
- ../pascal-mibi-early-dataset-notest

Examples
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
│       ├── test.txt    # Empty
│       ├── train.txt   # Filenames for training
│       └── val.txt     # Filenames for validation
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

### Test dataset ###

Prepare a test dataset in the following locations, each with a directory structure in PASCAL format.

- ../pascal-mibi-both-dataset-test-all
- ../pascal-mibi-delay-dataset-test-all
- ../pascal-mibi-early-dataset-test-all

Examples
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
│       ├── test.txt        # Filenames for test excluding no annotations
│       ├── test_all.txt    # Filenames for test
│       ├── train.txt       # Empty
│       └── val.txt         # Empty
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

### Training ###

Run ./train-mibi-serial.sh

Training is performed on each of the three datasets, and the results are stored in the following directories.

- ./train-mibi-both/
- ./train-mibi-delay/
- ./train-mibi-early/

Examples in the ./train-mibi-both/

```
├── logs
│   └── events.out.tfevents....     # TensorBoard log
├── snapshots
│   ├── resnet50_pascal-mibi_01.h5  # Trained model
│   ├── ...
├── train.err                       # Error log
└── train.log                       # Log
```

### Evaluation ###

Run the training described in the previous section, or copy the trained model (h5 file) from https://github.com/Tc-MIBI/keras-retinanet-mibi/releases
Set it to ./snapshots/

```
snapshots/
├── both-best-performing.h5
├── delay-best-performing.h5
└── early-best-performing.h5
```

Run ./evaluate-mibi-serial.sh

Evaluate the following against "three models" x "three datasets" x "validation dataset and test dataset".

- FROC
- mAP
- Image generation with likelihood scores for bounding boxes
- Presence/absence judgment by score criteria value

Evaluation results are stored in . /evaluate-mibi/.


# 説明: 日本語
# keras-retinanet-mibi #

fizyr/keras-retinanet をベースとして MIBI タスクの為にカスタマイズしたもの。

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
