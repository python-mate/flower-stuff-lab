import keras.preprocessing.image
import keras.engine.functional
import numpy


def predict_top_class(model: keras.engine.functional.Functional,
                      img_nad: numpy.ndarray,
                      classes: list) -> list:
    """ひとつの画像を predict します。結果はひとつのクラス名のみ。

    Args:
        model (keras.engine.functional.Functional): モデル。
        img_nad (numpy.ndarray): 正規化を終えた画像。
        classes (list): クラスのリスト。

    Returns:
        list: Prediction 結果のリスト。
    """
    predicted = model.predict(img_nad, batch_size=1, verbose=1)
    predicted = numpy.argmax(predicted, axis=-1)
    return [classes[i] for i in predicted]


def predict_top_classes(model: keras.engine.functional.Functional,
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
    img = keras.preprocessing.image.load_img(
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
    img = keras.preprocessing.image.img_to_array(img)
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
