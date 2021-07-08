flower-stuff-lab/yuueguci
===

🌻 FlowerStuff project; yuu-eguci 実験室。

## 各ファイル概説

順番は作った順序です。

|                          File                          |                                                                        Description                                                                        |
|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `yolov5_easy_detection.ipynb`                          | いっちばん最初にググりまくって作った習作。これは分類を行うモデルではない。スクリーンショットにレクタングルをつけるモデル。すべてはここから始まった……。    |
| `fine_tuning_with_17flowers.ipynb`                     | 分類を行うモデル VGG16 を発見して、これまたググりまくり、 fine-tuning して 17flowers を学習します。 hdf5 保存なし。ホントに習作。                         |
| `mnist_fitting.ipynb`                                  | MNIST dataset を LeNet CNN で学習します。結果は hdf5 で保存します。                                                                                       |
| `mnist_prediction.ipynb`                               | これ↑で作った hdf5 をロードして prediction を試します。                                                                                                   |
| `cifar10_fitting.ipynb`                                | CIFAR-10 dataset を LeNet CNN をちっと強化したモデルで学習します。結果は hdf5 で保存します。                                                              |
| `cifar10_prediction.ipynb`                             | これ↑で作った hdf5 をロードして prediction を試します。                                                                                                   |
| `my_utils.py`                                          | この実験中に、共通して使いたい処理があります。それを詰め込むモジュールです。 Notebook 内で本 repository を clone し、本モジュールを import して使います。 |
| `my_utils_test.ipynb`                                  | これ↑を clone -> import -> 使うテストをします。                                                                                                           |
| `inception_v3_fine_tuning.ipynb`                       | InceptionV3 モデルをベースとした fine-tuning を行い 17flowers を学習します。結果は hdf5 で保存します。                                                    |
| `inception_v3_fine_tuning_hyperparameter_tuning.ipynb` | 🚧 これ↑のハイパーパラメータチューニングをもっと効率的に行うことを目標として制作中。                                                                       |
| `vgg16_fine_tuning.ipynb`                              | 🚧 VGG16 モデルをベースとして fine-tuning を行う予定。                                                                                                     |
