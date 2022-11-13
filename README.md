
# SENG

The Sketchy Empirical Natural Gradient Methods (**SENG**) is an algorithm for solving large-scale deep learning problems. It utilizes efficient sketching methods to reduce the computational and memory complexity of the Empirical Natural Gradient method.

Currently, **SENG** supports the convolution layer, batch normalization layer and fully-connected layer. 

In this repository, we provide two versions of SENG in Pytorch and MindSpore, respectively. These codes are organized as follows:

```Current-Folder
    ├── MindSpore    : Url of the MindSpore version on ImageNet-1k with ResNet50. 
    |                  See README in the subfolder for more details.  
    └── PyTorch
          ├──cifar10 : the PyTorch version on CIFAR10 with ResNet18 and VGG16_bn. 
          |            See README in the subfolder for more details.
          └──imagenet: the PyTorch version on ImageNet-1k with ResNet50. 
                       See README in the subfolder for more details.
```

## Useage
To use **SENG** on other models, please refer to the `./PyTorch/imagenet/main_seng_bs.py` or the following codes as an example:

  ```python
  optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
  preconditioner = SENG(net, 1.2, update_freq=200)

  for inputs, targets in trainloader:
     output = net(inputs)
     loss = criterion(outputs, targets)
     optimizer.zero_grad()
     loss.backward()

     preconditioner.step()
     optimizer.step()
  ```

## Contact 

We hope that the package is useful for your application. If you have any bug reports or comments, please feel free to email one of the toolbox authors:

- Minghan Yang, yangminghan at pku.edu.cn
- Dong Xu, taroxd at pku.edu.cn
- Zaiwen Wen, wenzw at pku.edu.cn

## Reference

Minghan Yang, Dong Xu, Zaiwen Wen, Mengyun Chen, Pengxiang Xu, Sketch-Based Empirical Natural Gradient Methods for Deep Learning. Journal of Scientific Computing 92.3 (2022): 1-29.
