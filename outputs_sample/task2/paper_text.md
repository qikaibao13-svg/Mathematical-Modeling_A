# 可写入论文的任务二方法与结果

本文首先将 XJTU-SY 全寿命轴承振动数据转化为分钟级特征表，提取时域、频域、包络及时频域特征，并构造双通道融合特征。随后参考 *Remaining useful life prediction of rolling bearings via wavelet feature fusion and LSTM networks* 的特征筛选思想，计算候选特征的相关性、单调性和鲁棒性，并采用熵权法自动确定三项指标权重，得到综合评价函数 `J(X)`。根据 `J_score` 筛选核心特征后，以得分归一化权重构造轴承健康指标 `HI_bearing`，并进行平滑和累计最大值修正。

在退化建模阶段，本文正式比较 PF-based、EKF-based、Random Forest 和 BiLSTM-Attention 四类模型。所有模型均在相同测试轴承、相同 inspection points 和相同全寿命伪在线截断规则下预测 `RUL_ratio`，并统一采用 MAE、MSE、RMSE 和 Score 评价。实验结果表明，按正式 Score 口径，本轮正式最优模型为 `Random Forest`；同时 BiLSTM-Attention 在 MAE、MSE 和 RMSE 上取得最低误差，说明其对退化序列的拟合能力较强，但在当前非对称 Score 规则下未排第一。Hybrid_RVM_exponential_Frechet 作为文献复现参考单独报告，不进入正式主比较表。

最终将核心特征、`HI_bearing`、三阶段边界、源域候选样本、归一化退化模板和正式最优源模型交付给任务三迁移学习使用。其中主源域样本为 `Bearing3_1`，辅助源域样本为 `Bearing2_5`、`Bearing1_1` 和 `Bearing3_5`。
