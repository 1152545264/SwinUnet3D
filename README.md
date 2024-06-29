# SwinUnet3D
demo中最后一个字母为1的表示可以正常使用，为0的表示图像没有预处理好，效果很差，dice系数小于0.001 ，所有训练脚本都放在了demo文件夹中。

下载好对应的数据集之后，记得去Config类中把下图中的data_path、TrainPath和PredDataDir改成自己的路径即可完成训练
![img.png](img.png)

训练主函数位置：
![image](https://github.com/1152545264/SwinUnet3D/assets/44309924/701a2631-7561-4d86-a4fa-9bdec941318a)
其他的demo相同
版本问题：V1只是单纯的使用transformer实现的，V2才是论文中的主版本，混合了transformer和卷积

如果我们的工作对您有用，请引用对应的论文：https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-023-02129-z
