CenterNet:Objects as Points

（只分析2D检测）



Anchor free VS based on anchors

（1）先验框；（2）最大值抑制后处理；



输入：512X512

输出：128X128



数据增强方式：随机flip, 随机scaling (比例在0.6到1.3)，裁剪，颜色jittering；

优化器：Adam 

主干网：

ResNet-18, ResNet-101, DLA-34， Hourglass-104.  





整体架构：





损失函数：

带focal loss 的分类损失 ：





中心偏离损失：因下采样引起的量化误差



box_size损失：



total Loss:





网络优点

1.采用全卷积网络直接得到4倍下采样的热力图，所以不需要提前设定anchors, 所以大大减少了网络参数量和计算量。 

2.热力图的通道数等于要检测的目标类别数，热力图的前100个峰值作为网络提取的目标中心点，然后设定一个阈值进行筛选得到最终的目标中心点。 

3.Centernet 中所有的上采样前都有deformable卷积，这种卷积的作用就是使得网络的感受野变得更加精确，而不是限定为3*3的矩形卷积框内。 

4.4倍的下采样feature map 比一般网络的分辨率高很多，所有不需要进行多尺度预测和特征金字塔也可以同时较好的检测大小目标。 

5.Centernet 不需要NMS，因为所有检测的中心点是由热力图的峰值得到的，这样就已经有一个非极大值抑制的过程，而且NMS是十分耗时的，所以Centernet才能又好又快。 



CenterTrack的inference 过程：

1.提取heatmap上所有的peak ；

2.根据score排序，得到top-100的极值点； 

3.结合预测出的offset和size，得到目标的包围框信息如下。所有的输出结果都直接从feature map上获得，而不需要再做NMS等后处理操作。







CenterTrack：Tracking Objects as Points

将CenterNet检测器推广到目标跟踪领领域

CenterTrack = CenterNet + 4 * Input-channels + 2 * output-channels. 

作为一个检测器，CenterNet已经能够给出跟踪所需的很多信息，如位置、大小和得分。但是它不具备预测未直接出现在当前帧的目标的功能，所以在CenterTrack中，将当前帧及其上一帧图像共同输入模型当中，旨在帮助网络估计场景中对象的变化并根据上一帧图像提供的线索恢复当前帧中可能未观察到的对象。

CenterTrack中将跟踪看做一个从局部角度观察的问题，比如当一个目标离开画面或被遮挡又重新出现时，跟踪模型不会记住它，而是会重新给它分配一个新的ID. 因此，CenterTrack将跟踪建模成了一个在连续帧之间传递检测结果ID的问题，而没有考虑如何给时间上有较大间隔的对象重新建立联系的问题。 (张珺老师昨天说的CenterTrack的ID Switch 较高的原因，CenterTrack并不重点关注这个ID的问题)



CenterTrack还将上一帧图像的检测结果添加到输入中，具体做法是根据上一帧的检测结果绘制一张单通道heatmap，其中peak位置对应目标中心点，并使用与训练CenterNet过程中相同的高斯核渲染办法（根据目标大小调整高斯参数）进行模糊处理，为了降低误报概率，作者只对检测结果中得分高于一定阈值的目标进行渲染（即得分低的目标不会体现在新生成的heatmap上）。 



Association

为了能够在时间上建立检测结果直接的联系，CenterTrack添加了2个额外的输出通道，用于预测一个2维的偏移向量，即描述的是各对象在当前帧中的位置相对于其在前一帧图像当中的位置的X/Y方向的偏移量。此处的训练监督方式与CenterNet中对目标对象长宽或中心偏移情况的部分训练方式相同。 





在有了各对象的位置，及其对应的偏移情况后，使用简单的贪婪匹配策略即可建立起对应目标在帧间的联系。 



实验结果对比

MOT17数据集上进行训练，train：val =1:1

检测结果：

CenterNet ：   76.4%  58.3% 

CenTerTrack： 95.5%  62.5%

原因：

（1）选用的数据集MOT17数据集是一个跟踪的数据集；其中的标注信息包括一些被遮挡的目标，这些目标可以被追踪网检测到，但检测网CenterNet检测不到；

（2）训练集数据量较少（train：2857， val：2452），检测网训练不充分；训练集的多样性较少；

