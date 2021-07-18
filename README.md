flower-stuff-lab
===

🌻 FlowerStuff project; this repository is for experiment. This repository was constructed to enable each member be able to fetch other members' experiment results and to share them. And to each member be able to execute them on the Google Colaboratory.

![flower-stuff-project main image](https://user-images.githubusercontent.com/28250432/125736317-740cd173-d30c-4e55-ab4a-765182601558.jpg)

⚗️ 「花のやつ」プロジェクト、研究用リポジトリです。このリポジトリは、「メンバーそれぞれが、他メンバーの研究結果を簡単に取得、共有できて」「メンバーそれぞれが、他メンバーの研究結果を Colaboratory でかんたんに実行できる」ことを目指して構築されました。 Colaboratory ってこういうふうに GitHub に保存しながら使うものではないことはわかっているけれど、最初に思いついたのがこういうやり方だったのでとりあえずこれでやってみるよ。

## LT document

このプロジェクトは Lightening Talk を見据えて進められました。 LT 資料は…… @ayano1212 が持っています。

## Experiment purpose

1. 花の画像を渡したら、それが何の花なのか高確率であててくる Python を作る。
1. ウケる LT をする。

そのために必要だと思うコト。

- [x] それが何の花なのか高確率であててくる Python を作るコト。
- [x] それを別の環境でも動かす方法を確立するコト。
    - 「これ clone してねー」「それで `python foo.py flower.jpg` してねー」で完結するようなものがいい。
- [x] Anything else?

## How to experiment

### はじめに

- 今回の研究では、今のトコ、 ipynb ファイルのみでの実践を見込んでいます。
- 研究結果である ipynb をひとつの repository に保存していきます。
- メンバーが repository を共有することで、互いの成果を簡単に参照しあえる……というのを目指しています。

### ほかのメンバーの(あるいは自分の)ipynb を Colaboratory に開く

![Colaboratory で Open notebook](https://user-images.githubusercontent.com/28250432/122402516-83f90800-cfb8-11eb-9246-46dcbdd60ee8.png)

![GitHub から yuu-eguci/flower-stuff-lab を開く](https://user-images.githubusercontent.com/28250432/122405442-0682c700-cfbb-11eb-9cab-a2df34e01e45.png)

### 自分の成果である ipynb を repository へ保存する

![Colaboratory で Save a copy in GitHub](https://user-images.githubusercontent.com/28250432/122406341-bd7f4280-cfbb-11eb-840d-5dd2e0f811ff.png)

![GitHub へ ipynb を保存する](https://user-images.githubusercontent.com/28250432/122406648-f7e8df80-cfbb-11eb-9bf4-3818773a62f0.png)

## Project rules

- 自分の名前のついたフォルダの中に保存するコト。
- データセットをリポジトリに push しないコト。
- データセットは自分のマイドライブに保存して、 ipynb で実行するときは一時領域へ移動させて学習をするコト。
    - 参考: [fine_tuning_with_17flowers.ipynb](https://github.com/yuu-eguci/flower-stuff-lab/blob/main/yuueguci/fine_tuning_with_17flowers.ipynb)
    - ↓画像参照。

![5](https://user-images.githubusercontent.com/28250432/122409790-629b1a80-cfbe-11eb-8e56-fb712420f37d.png)

## Organization の repository に保存できないとき

こちら↓に案内画像を用意しました。これに従って、 googlecolab があなたの GitHub アカウントを通じて organization repository にアクセスすることを認可してください。

![googlecolab](https://user-images.githubusercontent.com/28250432/125185101-9e43a000-e25d-11eb-94bb-8746b36549b4.png)
