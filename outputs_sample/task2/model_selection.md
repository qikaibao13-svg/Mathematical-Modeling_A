# 任务二正式主比较模型总结

统一协议：

- 测试轴承：Bearing3_1, Bearing2_5, Bearing1_1, Bearing3_5
- inspection points：[0.5, 0.6, 0.7, 0.8, 0.9]
- 目标：`RUL_ratio`
- 指标：MAE/MSE/RMSE 越小越好，Score 越大越好

正式主比较：

| model            | MAE    | MSE    | RMSE   | Score  | seq_len | rank_by_score | rank_by_rmse |
| ---------------- | ------ | ------ | ------ | ------ | ------- | ------------- | ------------ |
| Random Forest    | 0.2948 | 0.1227 | 0.3502 | 0.0799 |         | 1             | 2            |
| BiLSTM-Attention | 0.2342 | 0.0743 | 0.2725 | 0.0673 | 16.0000 | 2             | 1            |
| EKF-based        | 0.3884 | 0.1998 | 0.4470 | 0.0184 |         | 3             | 4            |
| PF-based         | 0.3523 | 0.1519 | 0.3898 | 0.0153 |         | 4             | 3            |

最终选择：`Random Forest`。

按正式 Score 口径，`Random Forest` 排名第一；按 MAE/MSE/RMSE 误差口径，BiLSTM-Attention 的点误差最低。论文中建议表述为：Random Forest 在当前 Score 规则下综合最优，BiLSTM-Attention 体现出更强的误差拟合能力，但其 Score 受预测偏差方向影响未排第一。本轮结论完全按实验结果给出，不预设深度模型一定更好。

Hybrid 参考结果单列：

| model                  | MAE    | MSE    | RMSE   | Score  |
| ---------------------- | ------ | ------ | ------ | ------ |
| Hybrid_RVM_exp_Frechet | 0.2513 | 0.0807 | 0.2841 | 0.0776 |
