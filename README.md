# Linear Attention Modeling for Learned Image Compression (LALIC) [CVPR 2025]


PyTorch modification of the CVPR 2025 paper **"Linear Attention Modeling for Learned Image Compression"**  
*[Donghui Feng](https://medialab.sjtu.edu.cn/author/donghui-feng/), [Zhengxue Cheng](https://medialab.sjtu.edu.cn/author/zhengxue-cheng/), Shen Wang, et al.*

---
### Update Log

**2025-6-3:** Add bfloat training for v6 model. For a 128*128 patch with **1** batchsize, the memory usage cost decreased from 9298MB to 8510MB(-9.25%,test on a RTX3080 GPU) 