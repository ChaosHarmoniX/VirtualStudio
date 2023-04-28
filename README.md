由于模型太大，github传不上去，所以使用时需要手动在BodyRelight/下新建目录checkpoints/example/，把模型net_latest拷贝到该目录下

可以运行`python BodyRelight/app/relight.py`和`python MODNet/app/matte.py`来测试

main.py仅作为代码样例，还达不到运行的要求

支持的功能：对图片中的人物重新打光



增加了MHFormer的代码，可以运行`python MHFormer/app/gen.py`来测试。

支持的功能：将视频拆分成图片、图片中提取人物3D关键点。