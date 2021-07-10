"""great_predictor
もちろん Great Pretender オマージュ。
"""

# Built-in modules
import sys

# 他、必要なライブラリを import します。
# NOTE: 今回は試しに keras. から参照するようにしてみます。
import tensorflow.keras
import numpy

# Prediction に使う hdf5 のパスです。
HDF5 = './inception_v3_n_flowers_fine_tuning_2nd_cell.hdf5'
INCEPTION_V3_TARGET_SIZE = (299, 299)
CLASSES_FOR_N_FLOWERS = [
    'Bluebell',
    'Buttercup',
    'ColtsFoot',
    'Cowslip',
    'Crocus',
    'Daffodil',
    'Daisy',
    'Dandelion',
    'Fritillary',
    'Iris',
    'LilyValley',
    'Pansy',
    'Snowdrop',
    'Sunflower',
    'Tigerlily',
    'Tulip',
    'Windflower',
]


def main(prediction_target_path: str):

    # Prediction に使う hdf5 をダウンロードしてきます。
    # https://drive.google.com/file/d/10PjbaGKUoNs1a0vPkAZ5z4Inl-ioaW6c/view?usp=sharing
    # TODO: これダウンロード方法わかんねーな。 requests で取得できるか?
    #       とりあえず手動で用意していただく。
    model = tensorflow.keras.models.load_model(HDF5)

    # 画像を img_nad へ変換します。
    # NOTE: nad = Normalization and Division(正規化と割り算)
    img_nad = convert_image_for_prediction(
        prediction_target_path,
        target_image_size=INCEPTION_V3_TARGET_SIZE,
        color_mode='rgb',
        verbose=False,
    )

    # 予測を行います。
    prediction = predict_top_classes(model, img_nad, CLASSES_FOR_N_FLOWERS, 3)

    # 予測の結果を出力します。
    print('\n'.join([
        '*****',
        f'Image: {prediction_target_path}',
        f'    Predicted: {prediction}',
        '*****',
    ]))


def predict_top_classes(model,
                        img_nad: numpy.ndarray,
                        classes: list,
                        top_n: int) -> list:
    """ひとつの画像を predict します。結果はクラス名と確信度のリスト。
    トップ n 件出します。

    Args:
        model (keras.engine.functional.Functional): モデル。
        img_nad (numpy.ndarray): 正規化を終えた画像。
        classes (list): クラスのリスト。
        top_n (int): トップ n 件結果を返します。

    Returns:
        list: Prediction 結果のリスト。
    """
    print(f'model type: {type(model)}')
    prediction = model.predict(img_nad)[0]
    top_indices = prediction.argsort()[-top_n:][::-1]
    return [(classes[i], prediction[i]) for i in top_indices]


def convert_image_for_prediction(image_path: str,
                                 target_image_size: tuple,
                                 color_mode: str,
                                 verbose: bool) -> numpy.ndarray:
    """メチャクチャ苦労して、画像を prediction へ変換する処理を整理しました。

    Args:
        image_path (str): 画像パス。
        target_image_size (tuple): 変換後の画像サイズ。
        color_mode (str): 'rgb', 'rgba', 'grayscale'
        verbose (bool): 途中のロギングを行う。

    Returns:
        numpy.ndarray: Prediction へ投げられる形式へ変換された画像。
    """

    # 引数チェックを実施します。
    assert len(target_image_size) == 2, (
        f'target_image_size には2要素が期待されています。与えられた要素数: {len(target_image_size)}'
    )
    assert color_mode in ['rgb', 'rgba', 'grayscale'], (
        f"color_mode には次の3種類が期待されています。与えられた引数: {['rgb', 'rgba', 'grayscale']}"
    )

    # load_img は画像をファイルから読み込み、PIL 形式で返します。
    # NOTE: だから内部で Pillow を必要としています。
    img = tensorflow.keras.preprocessing.image.load_img(
        image_path,
        # grayscale と color_mode の指定により、読み込んだあとに rgb に変換します。
        color_mode=color_mode,
        # 読み込んだあと(たとえば)32x32にリサイズします。
        target_size=target_image_size,
    )
    if verbose:
        print(f'load_img の返り値: {type(img)}')  # <class 'PIL.Image.Image'>
        print(f'PIL.Image.Image size: {img.size}')

    # PIL 形式を numpy.ndarray へ変換します。
    # NOTE: ここでエラーが出る可能性があります。
    #       TypeError: __array__() takes 1 positional argument but 2 were given
    #       このエラーは Pillow のバージョンを変更することで解消します。
    #       具体的には pip install "pillow!=8.3.0"
    img = tensorflow.keras.preprocessing.image.img_to_array(img)
    if verbose:
        print(f'img_to_array の返り値: {type(img)}')  # <class 'numpy.ndarray'>

    # 0-1 に変換します。
    # NOTE: 色は 0~255 で表されているので 255 で割ると 0~1 に変換できる。
    # NOTE: nad = Normalization and Division(正規化と割り算)
    img_nad = img / 255
    if verbose:
        print(f'正規化後の img: {type(img_nad)}')  # <class 'numpy.ndarray'>
        print(f'正規化後の img.shape: {img_nad.shape}')  # (32, 32, 3)

    # 4次元配列に
    # XXX: なんで4次元配列にするのかわかってない。
    img_nad = img_nad[None, ...]

    return img_nad


if __name__ == '__main__':

    assert len(sys.argv) >= 2, 'Usage "python great_predictor.py foo.png"'
    main(sys.argv[1])
