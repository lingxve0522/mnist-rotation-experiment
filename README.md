# MNIST 随机旋转增强完整复现包

本目录是独立提交包，已包含论文全部实验的代码、MNIST 数据、55 份权重、训练记录和评估结果。解压后保留文件夹结构即可运行。数据、代码和权重均按相对路径定位。

## 安装

安装 Python 3.10 或更高版本。正式训练环境为 Python 3.12.14、NumPy 2.3.5、Pillow 13.0.0。打开终端，进入本目录。

```powershell
python -m pip install -r requirements.txt
```

训练和评估采用 CPU 和 NumPy。run.py 默认将 OpenBLAS 线程设置为 2。不同 BLAS 或 Pillow 版本可能产生微小数值差异。

## 统一入口

```powershell
python run.py verify
python run.py evaluate
python run.py retrain
```

- verify 核对包内文件 SHA-256，并从保存的逐样本预测重新计算主实验汇总。输出 verified_summary.json。
- evaluate 重新加载全部 55 份权重核对原记录，并复算主实验统计、插值敏感性、复合扰动、概率诊断和图表。线性模型的逐样本预测、少样本模型的逐角度准确率均与原记录核对。
- retrain 从头运行全部 55 次训练，随后执行各项评估与绘图。固定 15 轮、同一数据划分和五个配对种子。总耗时取决于机器性能，运行期间会打印进度。

evaluate 默认写入 reproduced_evaluate，retrain 默认写入 reproduced_retrain。输出目录必须尚不存在。也可以指定路径。

```powershell
python run.py evaluate --out result_check_01
python run.py retrain --out result_train_01
```

## 文件对应

- mnist.npz 包含官方 60,000 张训练和 10,000 张测试图像。SHA-256 由训练代码核验。
- final_experiment/train.py 是主实验训练代码。final_experiment/runs 保存 20 份 MLP 权重和原始评估记录。
- linear.py、limited.py 对应线性分类器与少样本训练。根目录分别保存 15 份和 20 份权重。
- stress_fast.py 复算平移、噪声、大角度和多视图推理结果。
- diagnostics.py 复算 NLL、ECE、Brier 与逐类错误转移。
- final_experiment/interpolation_sensitivity.py 复算测试插值敏感性。
- final_experiment/make_figures.py 生成四幅实测图和对应数据。
- new_summary.py 汇总补充实验表格，final_experiment/summarize.py 汇总主实验。
- verify_inference.py 重新读取权重核对原始预测，manifest.json 用于检查提交包文件完整性。

主实验最终汇总为 final_experiment/summary.json。补充汇总为 new_summary.json。概率诊断为 diagnostics.json。重跑输出使用相同命名，原始记录保留在提交包中。

MNIST 来源 https://keras.io/api/datasets/mnist/ 和 https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz 。数据版权和许可见 Keras 官方页面 https://keras.io/api/datasets/mnist/ 。MNIST 数据在该页面标注为 CC BY-SA 3.0。

此前试跑的自定义测试集划分未计入本包。论文的正式模型都使用官方训练文件训练。所有结果均为已有运行记录，论文中的流程图属于说明性示意图，可由其描述重建。

## 本次整合验证

统一 evaluate 入口已完整跑通。55 份权重通过 595 个角度条件核对。压力测试、概率诊断和补充表格与原记录一致。完整 retrain 流程沿用已有训练算法，本次整合未再次执行全部 55 次训练。
