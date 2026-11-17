# 图注意力：公式、手写实现与消融验收

本仓库完成图注意力任务的最小复现：手写注意力打分、邻域 softmax、多头聚合，与 DGL 2.2.1 GATConv 对照；运行去注意力和不同头数消融。工程代码由 Codex 辅助编写与验证，不声称已获得导师验收分数。

## 目录与复现

```text
图注意力/
  gat.py
  test_gat.py
  experiment.py
GNN 入门/
  nodes.csv
  edges.csv
  train.csv
README.md
```

保持已有图数据的相邻目录关系。仅包含实现、必要测试、实验和报告所需文件，不包含环境、模型、预测或额外日志。使用已有 pytorch 环境，没有新建环境或安装升级依赖。此次验证代码在内存中执行，发布版本与被验证源码逐字核对；原本地项目不改动。

在下载的仓库根目录运行：

```powershell
& 'E:\dev\miniforge3\envs\pytorch\python.exe' -B '.\图注意力\test_gat.py'
& 'E:\dev\miniforge3\envs\pytorch\python.exe' -B '.\图注意力\experiment.py'
```

在其他机器使用具有相同依赖的 python 替换具体解释器路径。实验打印 DATA、逐次 RUN 和汇总 SUMMARY，不写入文件。

实测：Windows，Python 3.11.16，PyTorch 2.1.2+cu121，DGL 2.2.1+cu121，NVIDIA GeForce RTX 4050 Laptop GPU。正确性对照使用 CPU float64 与 CUDA float32；消融固定使用 CPU float32、单线程，以统一精度和执行条件。无 CUDA 时测试只执行 CPU 对照，组数会减少。不作 CPU/GPU 性能优劣结论。

## 公式与源码对应

处理后的有向边为 j→i，头编号为 k，输入 x 为 N×F_in。层本身不隐式加边；实验显式调用 add_self_loops，先去掉已有全部自环再为每节点加一个。非自环重复边保留为多条消息，不自动加反向边。

```text
z_i^k = W^k x_i
e_ji^k = LeakyReLU((a_src^k)^T z_j^k + (a_dst^k)^T z_i^k, slope=0.2)
alpha_ji^k = exp(e_ji^k) / sum_{l→i} exp(e_li^k)
u_i^k = sum_{j→i} alpha_ji^k z_j^k + b^k
concat: u_i = concatenate_k(u_i^k)
mean:   u_i = mean_k(u_i^k)
```

| 步骤 | 源码 | 形状 |
| --- | --- | --- |
| 一次计算所有头的投影 | fc(x).reshape(...) | N×K×D |
| 源/目的节点打分项 | (z * attn_l/attn_r).sum(-1) | N×K |
| 取边端点并 LeakyReLU | left[src] + right[dst] | E×K |
| 邻域 softmax | neighbor_softmax(scores,dst,N) | E×K |
| 发送消息 | z[src] * alpha.unsqueeze(-1) | E×K×D |
| 按目的节点求和 | index_add_(0,dst,messages) | N×K×D |
| 拼接或平均各头 | flatten(1) / mean(1) | N×(KD) / N×D |

重点是 softmax 的分组轴：每个接收节点、每个头，在该节点的入边上归一化；不能对全图所有边或对头维直接 softmax。实现用 scatter_reduce_ 求每组最大值，减去最大值后取 exp，再用 index_add_ 求每组分母。每组减去的最大值 detach，不改变 softmax 数学梯度，并避免指数溢出。重复边各自参与分母；每个有入边节点每头的系数和应为 1。

无边节点聚合为零，输出为偏置；实验通过自环保留自身信息。支持无边图及零节点图。不支持边权、异构/二部图、残差或注意力 dropout；对应 DGL 对照的 feat_drop=0、attn_drop=0、residual=False、activation=None。训练网络只在隐藏表示后使用 Dropout(0.2)，不把它当作 attention dropout。

去注意力消融将各边打分固定为 0，因此每个头的 alpha=1/in_degree(i)，不再创建可学习注意力向量。它仍使用同一图和两层消息传递，只去掉学习邻居权重的部分。拼接与平均均先包含各头偏置，与 DGL 返回 N×K×D 后再聚合的语义一致。

## 正确性与边界验收

测试方法共 5 个：

- test_dgl_parity：有向图和孤立点、自环及重复边、无边图、零节点图、单节点；两种自环处理、1/4 头、拼接/平均；CPU/GPU。将投影、注意力向量、非零偏置复制到 DGL，核对输出、每边 alpha、输入及全部参数梯度。
- test_softmax：大幅正负打分的数值稳定性，邻域内加常数不变性；与小邻域 torch.softmax 的结果对比，数值差分 gradcheck。
- test_uniform_ablation：手算均值聚合，重复边按多重边计数，确认去注意力后没有 attn 参数。
- test_aggregation_and_permutation：多头拼接后取均值等于 mean 输出，边排列不改变结果，自环每节点一个。
- test_invalid_inputs：越界、负索引、非 int64 或错误形状、错误特征维度、NaN/Inf 明确报错。

同一上游梯度用于比较自动求导。CPU 容差 atol=1e-10、rtol=1e-9；GPU 容差 atol=3e-5、rtol=3e-4。有限测试不构成对任意输入的形式化证明。

实测全部5个测试通过；CPU/GPU共80组参考对比。最大输出绝对误差 1.192092896e-07，最大注意力系数绝对误差 0.000000000e+00，最大梯度绝对误差 5.960464478e-07。这是本次测试观测值，不是任意输入的误差上界。

## 消融计划与固定实验协议

预先固定四种设置：均值聚合(4头、去注意力)、GAT 1头、GAT 2头、GAT 4头；每种运行 42、43、44 三个初始化种子，合计 12 次。已全部完成，未执行的扩展实验在下文单独标明。

两层结构：4→16→3。第一层头数 K=1/2/4，每头 D=16/K，总隐藏宽度恒定为16；ELU、Dropout(0.2)，第二层为单头三类 logits。注意力开启时的参数量均为169，去注意力为131。各设置使用相同的投影初值和 Dropout 随机流；仅头间划分和注意力开关改变。去注意力与有注意力的参数量差异是本消融本身的差异。

统一采用 CrossEntropyLoss、Adam(lr=0.01, weight_decay=0.0005)，最多300轮。监督损失仅取训练子集；每轮验证，以 Macro F1 最大者选检查点，相同时验证损失更低者优先。连续50轮没有更好记录则停止；恢复最佳内存参数后一次性评估独立留出集。没有根据留出结果追加调参或增加实验设置。

图来自已有平台教学合成数据：240节点，888原始边，加自环后1128边。只使用公开 train.csv 的156个标签；按类别、划分种子42固定为92训练、32验证、32独立留出。类别0/1/2分别为训练31/31/30、验证11/11/10、留出11/11/10。其余84个节点不读隐藏标签，保留在图中参与公开特征的传导式消息传递。ID不作为特征。均值和总体标准差仅从92个训练节点估计，再应用于全图。

下面的原始实验记录包括数据 SHA256、划分指纹与完整节点ID，避免只给一个种子却无法核查划分。数据生成来源为平台原创教学数据；未发现明确的再分发许可证，未自行附加数据许可证或声称额外授权。这里的验证和留出都是从公开标签中划分，不是平台隐藏测试分数。

## 实际结果

2026-11-17，全部12次运行完成。表中均值±标准差为同一固定划分上3个初始化种子的样本标准差，不包含数据划分不确定性。

| 设置 | 参数量 | 训练 Macro F1 | 验证 Macro F1 | 独立留出 Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 去注意力 4头 | 131 | 0.909937 ± 0.012526 | 0.915693 ± 0.017925 | 0.871222 ± 0.000000 |
| GAT 1头 | 169 | 0.934991 ± 0.010757 | 0.948268 ± 0.018544 | 0.841941 ± 0.058822 |
| GAT 2头 | 169 | 0.953017 ± 0.006208 | 0.936926 ± 0.032309 | 0.851025 ± 0.043089 |
| GAT 4头 | 169 | 0.960354 ± 0.032968 | 0.917442 ± 0.017762 | 0.838882 ± 0.035273 |


结论：本次留出集上，去注意力均值聚合的平均 Macro F1 高于全部注意力配置。注意力模型在验证集上的优势未转化为更好的留出结果；4头设置训练分数较高，但没有显示更多头带来更好的泛化。这里只能支持这份小型教学图和固定协议下的观察，不能推出注意力普遍无效。留出集仅32节点，三次初始化不能代替多数据集、多划分或显著性检验；均值聚合标准差为0仅表示三次预测指标相同，不表示不存在其他不确定性。

未执行的扩展：真实大图基准、多数据划分、注意力系数可视化、attention dropout/残差消融、性能计时。不得把这些内容作为已完成实验或性能结论。没有训练全量标签模型、生成提交CSV或上传到题目平台。

## 原始实验记录

以下为实际内存执行所得结果，包含所有12次的轮次、训练/验证损失和三组 Macro F1，运行 experiment.py 会重新打印相同结构的记录：

```json
{
  "metadata": {
    "nodes": 240,
    "edges_with_loops": 1128,
    "split_counts": {
      "fit": 92,
      "validation": 32,
      "holdout": 32
    },
    "split_sha256": "ef3cb22399e8709686a24dba33b6b53764542a9f88d57d22f5464906f6e98fed",
    "data_sha256": {
      "nodes": "3fe158f35db0cb6886f43f6d9e63c0e5aecd86045e5622fbe5ccbff22f7ccf80",
      "edges": "577a1629b5a83fd49d197c053c9899d898ea65917a62563b2ef101c89638d04a",
      "train": "c75a34ae4d684f84becb8336a7bf2c3a9cbc1b62d2a11f4556c5b715883dc832"
    },
    "split_ids": {
      "fit": [
        "n0075",
        "n0228",
        "n0232",
        "n0052",
        "n0150",
        "n0110",
        "n0031",
        "n0127",
        "n0236",
        "n0089",
        "n0234",
        "n0062",
        "n0148",
        "n0045",
        "n0153",
        "n0128",
        "n0224",
        "n0214",
        "n0199",
        "n0049",
        "n0209",
        "n0115",
        "n0113",
        "n0063",
        "n0186",
        "n0134",
        "n0087",
        "n0222",
        "n0176",
        "n0069",
        "n0091",
        "n0117",
        "n0177",
        "n0029",
        "n0138",
        "n0003",
        "n0105",
        "n0187",
        "n0172",
        "n0219",
        "n0124",
        "n0132",
        "n0122",
        "n0120",
        "n0085",
        "n0092",
        "n0072",
        "n0061",
        "n0038",
        "n0191",
        "n0178",
        "n0143",
        "n0125",
        "n0096",
        "n0218",
        "n0088",
        "n0197",
        "n0162",
        "n0195",
        "n0066",
        "n0239",
        "n0179",
        "n0108",
        "n0212",
        "n0126",
        "n0076",
        "n0005",
        "n0182",
        "n0042",
        "n0048",
        "n0159",
        "n0043",
        "n0173",
        "n0051",
        "n0056",
        "n0112",
        "n0057",
        "n0205",
        "n0053",
        "n0035",
        "n0021",
        "n0185",
        "n0158",
        "n0136",
        "n0231",
        "n0086",
        "n0181",
        "n0220",
        "n0184",
        "n0018",
        "n0208",
        "n0034"
      ],
      "validation": [
        "n0213",
        "n0196",
        "n0235",
        "n0064",
        "n0012",
        "n0238",
        "n0200",
        "n0077",
        "n0082",
        "n0135",
        "n0095",
        "n0101",
        "n0028",
        "n0037",
        "n0129",
        "n0083",
        "n0065",
        "n0073",
        "n0014",
        "n0149",
        "n0165",
        "n0009",
        "n0192",
        "n0198",
        "n0015",
        "n0010",
        "n0174",
        "n0144",
        "n0194",
        "n0033",
        "n0050",
        "n0104"
      ],
      "holdout": [
        "n0146",
        "n0025",
        "n0225",
        "n0119",
        "n0081",
        "n0046",
        "n0040",
        "n0026",
        "n0041",
        "n0233",
        "n0011",
        "n0206",
        "n0130",
        "n0004",
        "n0171",
        "n0151",
        "n0002",
        "n0155",
        "n0190",
        "n0006",
        "n0013",
        "n0142",
        "n0111",
        "n0139",
        "n0099",
        "n0164",
        "n0168",
        "n0079",
        "n0230",
        "n0147",
        "n0036",
        "n0154"
      ]
    }
  },
  "runs": [
    {
      "attention": false,
      "heads": 4,
      "seed": 42,
      "parameters": 131,
      "best_epoch": 223,
      "run_epochs": 273,
      "train_loss": 0.2220800220966339,
      "validation_loss": 0.24090717732906342,
      "train_f1": 0.9028909802436829,
      "validation_f1": 0.903717577457428,
      "holdout_f1": 0.8712215423583984
    },
    {
      "attention": false,
      "heads": 4,
      "seed": 43,
      "parameters": 131,
      "best_epoch": 199,
      "run_epochs": 249,
      "train_loss": 0.23051920533180237,
      "validation_loss": 0.24428774416446686,
      "train_f1": 0.9243993163108826,
      "validation_f1": 0.9363009333610535,
      "holdout_f1": 0.8712215423583984
    },
    {
      "attention": false,
      "heads": 4,
      "seed": 44,
      "parameters": 131,
      "best_epoch": 179,
      "run_epochs": 229,
      "train_loss": 0.25092560052871704,
      "validation_loss": 0.26450875401496887,
      "train_f1": 0.9025206565856934,
      "validation_f1": 0.907059371471405,
      "holdout_f1": 0.8712215423583984
    },
    {
      "attention": true,
      "heads": 1,
      "seed": 42,
      "parameters": 169,
      "best_epoch": 149,
      "run_epochs": 199,
      "train_loss": 0.17861664295196533,
      "validation_loss": 0.21342425048351288,
      "train_f1": 0.9458582997322083,
      "validation_f1": 0.938805341720581,
      "holdout_f1": 0.8759018778800964
    },
    {
      "attention": true,
      "heads": 1,
      "seed": 43,
      "parameters": 169,
      "best_epoch": 123,
      "run_epochs": 173,
      "train_loss": 0.18800249695777893,
      "validation_loss": 0.22875651717185974,
      "train_f1": 0.9347670078277588,
      "validation_f1": 0.9696342349052429,
      "holdout_f1": 0.8759018778800964
    },
    {
      "attention": true,
      "heads": 1,
      "seed": 44,
      "parameters": 169,
      "best_epoch": 51,
      "run_epochs": 101,
      "train_loss": 0.309065580368042,
      "validation_loss": 0.3079906702041626,
      "train_f1": 0.9243473410606384,
      "validation_f1": 0.936363697052002,
      "holdout_f1": 0.7740183472633362
    },
    {
      "attention": true,
      "heads": 2,
      "seed": 42,
      "parameters": 169,
      "best_epoch": 153,
      "run_epochs": 203,
      "train_loss": 0.17026770114898682,
      "validation_loss": 0.1919589638710022,
      "train_f1": 0.9458582997322083,
      "validation_f1": 0.938805341720581,
      "holdout_f1": 0.8759018778800964
    },
    {
      "attention": true,
      "heads": 2,
      "seed": 43,
      "parameters": 169,
      "best_epoch": 151,
      "run_epochs": 201,
      "train_loss": 0.15387748181819916,
      "validation_loss": 0.15175296366214752,
      "train_f1": 0.9569197297096252,
      "validation_f1": 0.9682539105415344,
      "holdout_f1": 0.8759018778800964
    },
    {
      "attention": true,
      "heads": 2,
      "seed": 44,
      "parameters": 169,
      "best_epoch": 113,
      "run_epochs": 163,
      "train_loss": 0.20632542669773102,
      "validation_loss": 0.24672168493270874,
      "train_f1": 0.9562723636627197,
      "validation_f1": 0.903717577457428,
      "holdout_f1": 0.8012698292732239
    },
    {
      "attention": true,
      "heads": 4,
      "seed": 42,
      "parameters": 169,
      "best_epoch": 181,
      "run_epochs": 231,
      "train_loss": 0.10226010531187057,
      "validation_loss": 0.20167717337608337,
      "train_f1": 0.967656672000885,
      "validation_f1": 0.9073153138160706,
      "holdout_f1": 0.8441558480262756
    },
    {
      "attention": true,
      "heads": 4,
      "seed": 43,
      "parameters": 169,
      "best_epoch": 277,
      "run_epochs": 300,
      "train_loss": 0.07173161208629608,
      "validation_loss": 0.12330038845539093,
      "train_f1": 0.9890592694282532,
      "validation_f1": 0.9379509091377258,
      "holdout_f1": 0.8712215423583984
    },
    {
      "attention": true,
      "heads": 4,
      "seed": 44,
      "parameters": 169,
      "best_epoch": 85,
      "run_epochs": 135,
      "train_loss": 0.2548809349536896,
      "validation_loss": 0.28517264127731323,
      "train_f1": 0.9243473410606384,
      "validation_f1": 0.907059371471405,
      "holdout_f1": 0.8012698292732239
    }
  ],
  "summaries": [
    {
      "attention": false,
      "heads": 4,
      "parameters": 131,
      "train_f1_mean": 0.9099369843800863,
      "train_f1_std": 0.012526115461718947,
      "validation_f1_mean": 0.9156926274299622,
      "validation_f1_std": 0.01792536197179999,
      "holdout_f1_mean": 0.8712215423583984,
      "holdout_f1_std": 0.0
    },
    {
      "attention": true,
      "heads": 1,
      "parameters": 169,
      "train_f1_mean": 0.9349908828735352,
      "train_f1_std": 0.01075722667649926,
      "validation_f1_mean": 0.9482677578926086,
      "validation_f1_std": 0.01854414091082334,
      "holdout_f1_mean": 0.841940701007843,
      "holdout_f1_std": 0.05882248382757601
    },
    {
      "attention": true,
      "heads": 2,
      "parameters": 169,
      "train_f1_mean": 0.9530167977015177,
      "train_f1_std": 0.0062078853551050354,
      "validation_f1_mean": 0.9369256099065145,
      "validation_f1_std": 0.032309203267107865,
      "holdout_f1_mean": 0.8510245283444723,
      "holdout_f1_std": 0.043088833353351105
    },
    {
      "attention": true,
      "heads": 4,
      "parameters": 169,
      "train_f1_mean": 0.9603544274965922,
      "train_f1_std": 0.032968174037458935,
      "validation_f1_mean": 0.9174418648084005,
      "validation_f1_std": 0.017761814408939834,
      "holdout_f1_mean": 0.8388824065526327,
      "holdout_f1_std": 0.035272757612917066
    }
  ]
}
```

## 公式与源码问答准备

1. 为什么对目的节点分组？每个节点需要对接收到的邻居分配总权重1；全图softmax会令互不相连的邻域竞争。
2. 多头拼接与平均差异是什么？拼接保留各头的不同通道，维度为KD；平均维度为D。本实验第一层拼接，最终层单头输出。
3. 为什么没有在层内加自环？图预处理显式可见，DGL参考图能采用完全相同的边；测试也能独立验证不开自环的零入度行为。
4. 去注意力是否等于没有图？不是，仍在同一邻域上传递消息，只把学习到的alpha改为均匀权重。
5. 能否保证注意力更好？不能，本次留出结果没有支持这一结论。
6. 当前实现属于什么GAT？原始的加性/静态注意力形式，非GATv2；未主张复现论文的完整基准结果。

参考：[DGL 2.2.1 GATConv](https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.nn.pytorch.conv.GATConv.html)、[Graph Attention Networks 原论文](https://arxiv.org/abs/1710.10903)。源码中的源节点 attn_l、目的节点 attn_r 对应对拼接向量打分的拆分，不需要为每条边显式构建两倍宽度的拼接张量。
