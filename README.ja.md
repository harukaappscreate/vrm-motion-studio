# VRM Motion Studio

ダンスや動きの動画から、VRM のモーション（`.vrma`）をブラウザだけで作るツールです。Pyxel 3 のレトロなアバターで動きを確かめられます。[VRM RETRO VIEWER](https://github.com/harukaappscreate/vrm-retro-viewer) の姉妹ツールです。

- itch.io で遊べます：https://haruka-apps-games.itch.io/
- すべてブラウザの中で動きます。インストールは不要で、動画はどこにも送られません。

## 使い方

1. `.mp4` を開きます（Sample 1〜3 でも試せます）。
2. MediaPipe が動画の人の姿勢を読み取り、52 本の骨の VRM を Pyxel 3 の 3D（`pyxel.cube`）で動かします。
3. 動きを `.vrma` として保存できます。ほかの VRM アプリでも使えます。
4. 自分の VRM（0.x / 1.0）をドロップすると、その動きで確かめられます。

手元で動かすときは、このフォルダで `python -m http.server 8000` を実行し、`http://localhost:8000/` を開きます。

## クレジット

- 初期アバター：千駄ヶ谷 渋（VRoid の CC0 サンプルモデル、VRM データ：coati 氏）
- サンプル動画：このツールのために AI（Google Gemini）で生成したもの
- [Pyxel](https://github.com/kitao/pyxel)（Takashi Kitao、MIT）、MediaPipe（Apache-2.0）、Kalidokit（MIT）

## ライセンス

MIT License（haruka_apps）。読み込む VRM や動画のライセンスは、それぞれに従ってください。
