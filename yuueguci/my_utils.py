import keras.preprocessing.image
import numpy


def convert_image_for_prediction(image_path: str) -> numpy.ndarray:
    """メチャクチャ苦労して、画像を prediction へ変換する処理を整理しました。

    Args:
        image_path (str): 画像パス。

    Returns:
        numpy.ndarray: Prediction へ投げられる形式へ変換された画像。
    """

    # load_img は画像をファイルから読み込み、PIL 形式で返します。
    # NOTE: だから内部で Pillow を必要としています。
    img = keras.preprocessing.image.load_img(
        image_path,
        # grayscale と color_mode の指定により、読み込んだあとに rgb に変換します。
        grayscale=False,
        color_mode='rgb',
        # 読み込んだあと 32x32 にリサイズします。
        # NOTE: これは CIFAR10 dataset の画像サイズです。
        target_size=(32, 32),
    )
    print(f'load_img の返り値: {type(img)}')  # <class 'PIL.Image.Image'>
    print(f'PIL.Image.Image size: {img.size}')

    # PIL 形式を numpy.ndarray へ変換します。
    # NOTE: ここでエラーが出る可能性があります。
    #       TypeError: __array__() takes 1 positional argument but 2 were given
    #       このエラーは Pillow のバージョンを変更することで解消します。
    #       具体的には pip install "pillow!=8.3.0"
    img = keras.preprocessing.image.img_to_array(img)
    print(f'img_to_array の返り値: {type(img)}')  # <class 'numpy.ndarray'>

    # 0-1 に変換します。
    # NOTE: 色は 0~255 で表されているので 255 で割ると 0~1 に変換できる。
    # NOTE: nad = Normalization and Division(正規化と割り算)
    img_nad = img / 255
    print(f'正規化後の img: {type(img_nad)}')  # <class 'numpy.ndarray'>
    print(f'正規化後の img.shape: {img_nad.shape}')  # (32, 32, 3)

    # 4次元配列に
    # XXX: なんで4次元配列にするのかわかってない。
    img_nad = img_nad[None, ...]

    return img_nad
