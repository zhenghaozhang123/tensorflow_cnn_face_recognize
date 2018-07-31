# tensorflow_cnn_face_recognize
-------让系统认识我-------

-------人脸识别系统-------

---VERSION2：基于tensorflow_cnn来做模型

---背景：

上一篇我的github:zhenghaozhang讲了利用dlib来进行人脸识别的例子，列举了三个缺点。

此处模型解决了上一篇讲到的两个缺点：

1.判定是否同一个人的阈值难以确定。

2.模型适合小型人脸数据库，一旦人脸数据库人数过多，此处的阈值更加难以确定。

---优点：

1.不再利用dist距离法，也就不再需要去定义阈值。因为上一篇我写到，阈值的值非常难定，特别是当人脸数据库人数多时，dist这个值是不稳定，甚至没办法确定的。
因此我们利用cnn的模型，将思路从两张图片的dist距离转换为“是非”问题，即二分类问题。（1.是我数据库的人。0.不是我数据库的人）。从而解决阈值选择的困难。

2.模型可以是多样本模型，解决了DLIB存在的小样本模型的局限。上一篇我讲到，当人脸数量一旦增加时，dist将是不稳定的。而用tensorflow的cnn模型，我们可以将
我们需要存进我们人脸数据库的人脸存进数据库，并将其定义为Label 1.训练时将不属于人脸数据库的数据定义为LABEL 0。因此成功将问题转换为二分类问题。相信
二分类问题大家很熟悉了。

---缺点：

当我们完成训练好的模型之后，一旦我们要往人脸数据库中增加新成员，则需要重新去跑模型（因为模型需要去记住新的成员，所以模型需要重新训练）。我们知道在现实
中，训练一个好的模型参数出来是需要花费大量时间的，如果一旦增加新成员，便重新训练模型，这个成本是相当高的，也不符合实际。
（此处的缺点解决方案，我将在下一篇进行讲解）


此处为版本2.通过tensorflow_cnn，将定阈值问题转换为二分类问题。

此处案例为了简便讲解和与上一篇的案例做对比，依旧以识别本人项目为主。

文件解析：

get_my_faces.py -----通过电脑摄像头实现对自己人脸的抓取，并储存在个人人脸数据库中，以备后面进行识别。

get_other_people.py -----提取他人照片存放在另一个数据库中，这个数据库是为了训练网络时候，与本人人脸数据库做比对的数据。

完成上面两个步骤之后，将存在两个数据库，一个是我本人的数据库，一个是他人的数据库。这两个数据库就是我们的二分类问题。我们需要网络去完成的任务就是：
告诉网络，照片A是来自哪个数据库的。比如告诉网络：照片A 是我的数据库的，它的标签为1. 照片B 不是我的数据库的，它的标签为0.我们需要通过不断训练网络
在不过拟合的情况下，让网络熟悉我告诉它要做的事情。

train.py -----这个就是我们的网络，也是整个环节的重点，但是其实网络并不难，只需要补补基础卷积神经网络就会用。

recognize_me  -----加载我们已经训练好的模型，并将模型利用摄像头来识别是否本人。

神经网络中参数的选择：

我建议使用已经他人选择好的网络，免去一些步骤，一般已经落实下去的网络参数都是具备通用性的。当然除了矩阵是需要修改的。好比当你的图片尺寸进行修改后，你的
W 值 的shape 是需要进行修改的。

备注：这个模型是做系统识别时候的个人见解，如果存在知识点上的误导，希望大家可以email来联系我：493200517@qq.com
