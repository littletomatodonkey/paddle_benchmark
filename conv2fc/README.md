# fc+layernorm与conv+layernorm预测速度对比

不同bs下的预测耗时结果如下:


| model       | bs=1    | bs=4     | bs=4     |
|------------|---------|----------|----------|
| base fc+ln | 0.13094 | 0.161158 | 0.20078  |
| conv + fn  | 0.11338 | 0.149882 | 0.187119 |

测试条件:开启mkldnn,线程数为10

* 结论:与layernorm结合时,conv+reshape速度稍微优于fc
