import delimited D:\data1.csv, varnames(1) clear   //录入数据
tobit pd x1-x10,ll(0) ul(4)   //tobit回归,ll(#)下边界，ul(#)上边界 x1-x12是自变量 变量区分大小写
est store m1   //储存回归结果
tobit nd x1-x10,ll(0) ul(4)
est store m2 
suest m1 m2   //显示两个回归结果
testnl [m1_pd]x1=[m2_nd]x1   //Wald检验
testnl [m1_pd]x2=[m2_nd]x2
testnl [m1_pd]x3=[m2_nd]x3
testnl [m1_pd]x4=[m2_nd]x4
testnl [m1_pd]x5=[m2_nd]x5
testnl [m1_pd]x6=[m2_nd]x6
testnl [m1_pd]x7=[m2_nd]x7
testnl [m1_pd]x8=[m2_nd]x8
testnl [m1_pd]x9=[m2_nd]x9
testnl [m1_pd]x10=[m2_nd]x10
/*
参考：https://blog.csdn.net/weixin_39825322/article/details/113034025