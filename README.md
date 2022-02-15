# my_ML
a machine learning package based on golang

这是一个基于golang的机器学习库和一些机器学习的数据集，可作为学习的辅助代码，也可以作为小型项目的demo，对于结果的敛散性没有做特殊设计，可根据具体问题具体考虑

基础的工具包：
read_excel: 读取特定格式的excel数据，返回float64或者string
test_data：用于测试算法的微型数据集，内含目录
entropy: 用于计算一些有关熵的内容
math_tool: 一些常用的数学工具（nn全排列，nk全排列，多元积分数值求解）

基础的机器学习算法：
liner_regression:线性回归（包含岭回归）
perception：感知机
logistic_regression: 逻辑回归
naive_bayes: 朴素贝叶斯
gda: 高斯判别分析
pca: 主成分分析
smo: 序列最小优化算法
svm: 支持向量机
knn: k近邻算法
k_means: k平均聚类算法
decision_tree: 决策树（ide3、c4.5）
