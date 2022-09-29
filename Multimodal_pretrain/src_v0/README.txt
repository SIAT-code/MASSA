在v4的基础上，做了如下改变：
（1）从一次性读数据变成一个batch一个batch读；
（2）将单卡运行变成多卡运行，主要是改了GVP的输入数据
（3）改变了pro，go，motif，domain和region的dict读取方式