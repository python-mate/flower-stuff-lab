flower-stuff-lab
===

🌻 FlowerStuff project; this repository is for experiment. This repository was constructed to enable each member be able to fetch other members' experiment results and to share them. And to each member be able to execute them on the Google Colaboratory.

⚗️ 「花のやつ」プロジェクト、研究用リポジトリです。このリポジトリは、「メンバーそれぞれが、他メンバーの研究結果を簡単に取得、共有できて」「メンバーそれぞれが、他メンバーの研究結果を Colaboratory でかんたんに実行できる」ことを目指して構築されました。 Colaboratory ってこういうふうに GitHub に保存しながら使うものではないことはわかっているけれど、最初に思いついたのがこういうやり方だったのでとりあえずこれでやってみるよ。

## Experiment purpose

1. 花の画像を渡したら、それが何の花なのか高確率であててくる Python を作る。
1. ウケる LT をする。

そのために必要だと思うコト。

- [ ] それが何の花なのか高確率であててくる Python を作るコト。
- [ ] それを別の環境でも動かす方法を確立するコト。
    - 「これ clone してねー」「それで `python foo.py flower.jpg` してねー」で完結するようなものがいい。
- [ ] Anything else?

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

