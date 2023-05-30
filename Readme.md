# C++
## 智能指针
### unqiue_ptr
```
// 手动实现：万能引用 + 完美转发
template <class T, class... Args>
unique_ptr<T> make_unique(Args&&... args){
  reutrn unique_ptr<T>(new T(std::forward(args)...));
}
```
- unique_ptr 堆内存进行唯一管理的行为，应该由程序员保证
- unique_prt 构造函数接收 T* 类型的指针，但是这样就失去了使用 unique_ptr 的作用了，如
  ```
  int *p = new int;
  unique_ptr<int> p1(p);
  // or
  unique_ptr<int> p1(new int(0));
  ```
- unique_ptr 删除了拷贝构造函数和拷贝辅助运算符，但是保留了移动拷贝赋值和移动拷贝构造函数，因此下列操作是可行的
  ```
  unique_ptr<int> p1 = make_unique<int>(0);
  unique_ptr<int> p2 = std::move(p1);
  ```
- release() 返回 T* 类型的指针，同时放弃管理权限
- reset() 回收内存，可以传入新的需要管理的内存

### shared_ptr
- make_shared<int>(0) 返回 shared_ptr 类型变量
- shared_ptr 存在拷贝构造函数和拷贝赋值运算符函数
- 没有 release 只有 reset()
### weak_ptr
- 无法操作资源，只用来查看资源是否被释放
- lock() 返回 shared_ptr 类型变量，用来进行资源的操作，如果已被释放则返回空指针
- use_count() 返回引用计数
- expired() 返回内存是否被释放

### auto_ptr C++17 delete
## 整数快速输入输出
```C++11
int read(){
    int x = 0, w = 1;
    char ch = 0
    while (ch<'0'||ch>'9'){
        if(ch == '-) w=-1;
        ch = getchar();
    }
    while(ch>='0'&&ch<'9'){
        x=(x<<1)+(x<<3)+(x^48);
        ch=getchar();
    }
    return x*w;
}
```
```C++11
int write(int num){
    if(num<0){
        num=-num;
        putchar('-');
    }
    static int stack[35];
    int top = 0;
    do{
        stack[top++] = num%10, num=num/10;
    } while(num);
    while(top) putchar(stack[--top] + 48);
}
```
# 旋转和平移
## 二维旋转
X->Y为旋转角度的正反向，则
```
  新的坐标 = （cos, -sin）
             (sin, cos) * 旧的坐标
```
## 三维旋转
### 绕 X 轴: 
X 轴坐标不变  
Y->Z 为旋转角度的正方向，则
```
  新的坐标 = (1, 0, 0)
            (0, cos, -sin)
            (0, sin, cos)  * 旧的坐标
```
### 绕 Y 轴
Y轴坐标不变   
X->Z 为旋转角度的正方向，则
```
  new = (cos, 0, -sin)
        (0, 1, 0)
        (sin, 0, cos) * old
``` 
### 绕 Z 轴
Z轴坐标不变  
X->Y 为旋转角度的正方向
```
  new = (cos, -sin, 0)
        (sin, cos, 0)
        (0, 0, 1)     * old
```
### 点旋转和轴旋转(左乘和右乘)
点旋转 a 等价于 坐标轴旋转 -a，以二维旋转为例，搭配左点右轴，有：
```
  new = (cos a, -sin a)         = old *  (cos -a. -sin -a)
        (sin a, cos a) * old             (sin -a, cos -a)
```
能看出来是转置关系
### 坐标系变换
将新坐标系想象成大坐标系，现坐标系想象为小坐标系
坐标系1下的坐标 = R12 * 坐标系2下的坐标 + T12，其中
- 坐标系2为当前坐标系
- R12 是将坐标系2的轴对齐坐标系1的轴 的旋转矩阵，X->Y, X->Z,Y->Z为旋转的正方向
- T12 为坐标系1原点到坐标系2原点的向量，在坐标系1，即新坐标下，的坐标
- 坐标系1（新坐标系） 到 坐标系2（旧坐标系） 的旋转矩阵 R21 = R12的逆， T21 = -R21 * T12

# 检测相关代码
## IOU
### NUMPY 版本
```
import numpy as np
def IOU(bbox, gts):
    '''
    Args:
        bbox: predictions (N, 4)-> (x_left_bottom, y_left_bootom, x_right_bootom, y_right_bootom)
        gts: ground truth (M, 4)
    '''
    left_bottom = np.max(bbox[:, np.newaxis, :2], gts[np.newaxis, :, :2])
    right_top = np.min(bbox[:, np.newaxis, 2:], gts[np.newaxis,:, 2:])
    width_length = np.max(0, right_top - left_bottom)
    intersection = width_length[:, :, 1] * width_length[:, :, 0]

    area_bbox = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    area_gt = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - bbox[:, 0])
    
    iou = intersection / (area_bbox[:, np.newaxis, :] + area_gt[np.newaxis, :, :] - intersection)
    return iou
```
## Rotated IOU
```
import cv2
def rotated_iou(predictions, gts):
    '''
    Args:
        predictions: (N, 5) -> (x_left_bottom, y_left_bootom, x_right_bootom, y_right_bootom, rotation_angle)
        gts: (M, 5)
    '''
    prediction_areas = np.prod(predictions[:, 2:4] - predictions[:, :2], axis=1)
    gt_areas = np.prod(gts[:, 2:4] - gts[:, :2], axis=1)
    ious = []
    for i, box1 in enumerate(predictions):
        tmp_ious = []
        rectangle_1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(gts):
            rectangle_2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
            
            intersection_pts = cv2.rotatedRectangleIntersection(rectangle_1, rectangle_2)[1]
            if intersection_pts is not None:
                order_pts = cv2.convexHull(intersection_pts, returnPoints=True)

                intersection_area = cv2.contourArea(order_pts)
                iou = intersection_area / (prediction_areas[i] + gt_areas[j] - intersection_area)
                tmp_ious.append(iou)
            else:
                tmp_ious.append(0.0)
        ious.append(tmp_ious)
    return np.array(ious, dtype=np.float32)
```
## NMS
```
import numpy as np
def NMS(predictions, scores, threshold):
    '''
    Args:
        predictions: (N,4) -> (x_left_bottom, y_left_bootom, x_right_bootom, y_right_bootom)
        scores: (N,)
        threshold: float32
    '''
    prediction_areas = np.prod(predictions[:, 2:] - predictions[:, :2], axis=1)
    order = scores.argsort()[::-1]
    res = []
    while order.size > 0:
        i = order[0]
        res.append(i)
        left_bottom = np.max(predictions[i][:2], predictions[order[1:], :2])
        right_top = np.min(predictions[i][2:], predictions[order[1:], 2:])
        width_length = np.max(0, right_top - left_bottom)
        overlaps = np.prod(width_length, axis=1)
        ious = overlaps / (prediction_areas[i] + prediction_areas[order[1:]] - overlaps)
        inds = np.where(ious < threshold)[0]
        order = order[inds + 1]
    return res
```
## Label Assignment of SSD
SSD 目标检测有两个原则：
1. 每个真实框和与其具有最大 IOU 的先验框匹配
2. 剩余的先验框和与其最大 IOU 的真实框匹配，并将 IOU 大于一定阙值的先验框作为正样本
```
def match_anchor_and_gt(threshold, truths, priors, variances, labels, location_truth, confi_truth, idx):
    '''
    Args:
        threshold: float.
        truths: [N, 4], float tensor. contains border coordinates
        priors: [M, 4], float tensor. contains border coordinates of priors
        variances: list of float. for encode predictions
            format: [0.1, 0.2]
        labels: [N,] int32 tensor. contains class number
        locations_truth: [batch_size, M, 4] float tensor. contains prediction for priors.
            OUT
        confi_truth: [batch_size, M] int32 tensor. contains label prediction for priors.
            OUT
        idx: batch index
    Returns: None
    '''
    BACKGROUND_CLASS = 0
    ious = IOU(truths, priors)
    best_prior_iou, best_prior_idx = ious.max(1, keepdim=False)
    best_truth_iou, best_truth_idx = ious.max(0, keepdim=False)
    best_truth_iou.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_truth_idx[j]] = j
    
    location_matches = truths[best_truth_idx]
    lables_matches = labels[best_truth_idx]
    lables_matches[best_truth_iou < threshold] = BACKGROUND_CLASS
    location_truth[idx] = encode(location_matches, priors, variances)
    confi_truth[idx] = lables_matches
```
## 损失计算
prediction 给出的 confidence 是没有经过 softmax 的，在困难样本挖掘时，需要根据损失排序，根据 softmax 和 交叉熵 进行等式推理能够得出损失等于 log_sum_exp(x) - x, x: format [batch, N, num_classes] or [-1, num_classes]
```
def forward(predictions, targets):
    '''
    class variable member:
        self.threshold
        self.variance
        self.use_gpu
        self.num_class
        self.negpos_ratio
    Args:
        predictions: list
            formate: [locations, confidence, priors]
        targets: 
    Returns:
        loss of confidence
        loss of locations
    '''
    # loc_data: [batch_size, M, 4] float tensor.
    # conf_data: [batch_size, M, 21] float tensor.
    loc_data, conf_data, priors = predictions
    batch_size = loc_data.size(0)
    num_priors = priors.size(0)
    
    loc_match = torch.Tensor((batch_size, num_priors, 4), require_grad = False)
    conf_match = torch.Tensor((batch_size, num_priors), require_grad = False)
    for idx in range(batch_size):
        loc_truths = targets[idx, :, :-1].data
        labels = targets[idx, :, -1].data
        match(threshold, loc_truths, priors, variance, labels, loc_match, conf_match)
    if use_gpu:
        loc_match = loc_match.cuda()
        conf_match = conf_match.cuda()
    
    # regression L1 for positive samples 
    positive = conf_match > 0
    
    positive_idx = positive.unsqueeze(positive_idx.dim()).expand_as(loc_match)
    loc_positive = loc_match[positive_idx].view(-1, 4)
    loss_loc = F.smooth_l1_loss(loc_positive, loc_truths[positive_idx].view(-1, 4), size_average=False)

    batch_conf = conf_data.view(-1, num_classes)
    loss_conf = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_match.view(-1,1))
    loss_conf[positive_idx.view(-1, 1)] = 0
    loss_conf = loss_conf.view(batch_size, -1)
    _, loss_idx = loss_conf.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)
    num_positive = positive_idx.sum(dim=1, keepdim=True)
    num_negitive = torch.clamp(negpos_ratio * num_positive, max=num_priors - num_positive)
    neg = idx_rank < num_negitive

    negitive_idx = neg.squeeze(2).expand_as(conf_data)
    positive_idx = positive.squeeze(2).expand_as(conf_data)
    loss_conf = F.cross_entropy(conf_data[(positive_idx + negitive_idx).gt(0)], conf_match[(positive + neg).gt(0)], size_average=False)

    num_positive = num_positive.sum()
    loss_conf = loss_conf / num_positive
    loss_loc = loss_loc / num_positive
    return loss_conf, loss_loc
```
## AP
```
def voc_ap(rec, prec, ues_07_metric=False):
    if ues_07_metric:
        ap = 0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec>=t])
            ap = ap + p / 11
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mprec = np.concatenate(([0.], prec, [0.]))
        for i in range(mprec.size- 1, 0, -1):
            mprec[i-1] = np.maximun(mprec[i-1], mprec[i])
        i = np.where(mprec[1:] != mprec[:-1])[0]
        
        ap = np.sum((mrec[i+1] - mrec[i]) * mprec[i+1])
    return ap
```
## points 2 voxel
```
def _points_to_voxel(
        points,
        voxel_size,
        coors_range,
        num_points_pre_voxel,
        coor_to_voxelidx,
        voxels,
        coors,
        max_points,
        max_voxels,
        nsweeps = -1,
):
    N = points.shape[0]
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int32)
    coor = np.zeros(shape=(4,), dtype=np.int32)
    voxel_num = 0
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor(points[i, j] - coors_range[j] / voxel_size[j])
            if c < 0 or c > grid_size[j]:
                failed = True
                break
            coor[ndim - j -1] = c
        coor[3] = int(points[i, -1])        # sweep idx
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2], coor[3]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num > max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2], coor[3]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_pre_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num += 1
    return voxel_num

def points_to_voxel(points, voxel_size, coors_range, nsweeps, max_points = 35, max_voxels = 120000):
    '''
    Args:
        points: [N, ndims] float tensor. points[:, :3] contain xyz and
            points[:, 3:] contain other information
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float, indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel
        max_voxels: int. indicate maximum voxels this function create.
    Returns:
        voxels: [M, max_points, ndim] float tensor.
        coordinates: [M, 3] int tensor. contains coordinates in grid
        num_points_per_voxel: [M] int tensor.
    '''
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    # voxelmap_shape: [nz, ny, nx, nsweeps], int tensor.
    voxelmap_shape = voxelmap_shape + (nsweeps, )

    num_points_pre_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    coors = np.zeros(shape=(max_voxels, 4), dtype=np.int32)

    voxels = np.zeros(shape = (max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    voxel_num = _points_to_voxel(
        points,
        voxel_size,
        coors_range,
        num_points_pre_voxel,
        coor_to_voxelidx,
        voxels,
        coors,
        max_points,
        max_voxels,
    )
    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_pre_voxel = num_points_pre_voxel[:voxel_num]

    return voxels, coors, num_points_pre_voxel
```
# Git
git 有四种 space :
- workspace
- index file
- local
- remote

当我们修改一个文件时，改变是没有被缓存的(unstaged), 为了能够 commit, 我们必须 stage it—— add it to index using git add. 当我们执行 commit 时便是将加入 index file 中的东西提交.
## Git 练习
```
https://learngitbranching.js.org/?locale=zh_CN
```
## 开发场景
1. 初始化并和远程分支创建链接
  ```
  git init
  git remote add origin https://github.com/...
  git checkout -b master origin/master
  // 或者直接 clone
  git clone https://github.com/...
  ```
2. 拉取 feature 分支
  ```
  git checkout localFeature origin/remoteFeature
  ```
3. 在自己的本地 feature 分支上进行开发
  ```
  git add .
  git commit -m "new_message"
  ```
4.  push 到远程 feature 分支上

5. 在远程 feature 分支上执行 git rebase 合并到远程 master 分支上
## 本地常用命令
### checkout and HEAD
用来将 HEAD 指向指定节点/分支
- HEAD : 指向正在其基础上进行工作的提交记录，通常指向分支名。
- git checkout - b bugFix : 创建名为 bugFix 的分支并将 HEAD 切换过去
### branch
创建、修改、查看分支
- git branch bugFix HEAD : 在 HEAD 处创建新的分支
- git branch -a : 查看所有分支
- git branch -r : 查看所有远程分支
- git branch -f main HEAD : 移动分支 main 指向 HEAD
### merge and rebase
- merge 创建一个新的提交，该提交有两个父提交
- rebase base_branch change_branch : 会从两个分支的共同祖先开始提取待变分支上的修改，将待变分支指向基分支的最新提交，最后将提取的修改应用到基分支的最新提交的后面。
### reset and revert
- reset : 将当前分支回滚到指定分支
- revert : 创建新的提交，该提交会撤销指定提交的变更。
- 区别 : reset 无法共享，而 revert 创建新的提交，可以 push 到远程共享

<img src = "img/2023-05-30 172506.png">
### cherry-pick
cherry-pick 可以将提交树上任何地方的提交记录取过来追加到 HEAD 上（只要不是 HEAD 上游的提交就没问题）。
### describe
git describe HEAD : 查看描述
## 远程常用命令
### 远程分支
远程分支的的命名规范是 remote_name/branch_name   
当切换到远程分支或在远程分支上提交时，会出现 HEAD 与 origin/main 分离的情况，这是因为origin/main 只有在远程仓库中相应的分支更新了以后才会更新。
### set origin url
```
git remote set-url origin https://github.com/JXZxiaowu/Help.git
```
### fetch
git fetch 完成两步
- 从远程仓库下载本地仓库中缺失的提交记录
- 更新远程分支指针

git fetch 并不会改变你本地仓库的状态。它不会更新你的 main 分支，也不会修改你磁盘上的文件。所以, 你可以将 git fetch 的理解为单纯的下载操作 :D   
git fetch origin master : 只现在指定的远程分支并更新指针
### pull
使用 git fetch 获取远程数据后，我们可能想将远程仓库中的变化更新到我们的工作中。其实有很多方法：
- git cherry-pick origin/main
- git rebase origin/master
- git merge origin/master   
实际上，Git 提供了 git pull 完成这两个操作, git pull = git fetch; git merge :D
### push
git push 负责将你的变更上传到指定的远程仓库，并在远程仓库上合并你的新提交记录。
```
假设你周一克隆了一个仓库，然后开始研发某个新功能。到周五时，你新功能开发测试完毕，可以发布了。但是 —— 天啊！你的同事这周写了一堆代码，还改了许多你的功能中使用的 API，这些变动会导致你新开发的功能变得不可用。但是他们已经将那些提交推送到远程仓库了，因此你的工作就变成了基于项目旧版的代码，与远程仓库最新的代码不匹配了。

这种情况下, git push 就不知道该如何操作了。如果你执行 git push，Git 应该让远程仓库回到星期一那天的状态吗？还是直接在新代码的基础上添加你的代码，亦或由于你的提交已经过时而直接忽略你的提交？

因为这情况（历史偏离）有许多的不确定性，Git 是不会允许你 push 变更的。实际上它会强制你先合并远程最新的代码，然后才能分享你的工作。

origin: c0 -> c1 -> c2 <-main
local: c0 -> c1 -> C3 <-main
             ^
             |
             origin/main
我们想要提交 C3，但是 C3 基于远程分支中的 c1，而在远程仓库中分支已经更新到 c2 了，所以 Git 拒绝 git push :D
```
我们可以这样解决
- git fetch; git rebase origin/master; git push
- 尽管 git merge 创建新的分支，但是它告诉 Git 你已经合并了远程仓库的所有变更，也能够完成 push 操作
- git pull --rebase 相当于 git fetch 和 git rebase   
单参数的 push
- git push origin src : src 是一个本地分支
- git push origin src:dest : dest 是远程分支(如果不存在则创建)
### 跟踪远程分支
- git branch -u origin/main feature : 对于已存在的分支 feature 使其跟踪远程 main 分支
- git checkout -b feature origin/main : 创建分支 feature 并使其准总远程 main 分支
## 其他命令
- git log  查看提交记录
- git reflog  可查看修改记录，包括回退记录
- git reset --hard {commit id} 回退版本
- git stash 未被提交的代码放到暂存区
- git stash pop 还原并清除最近一次的暂存记录
- git remote -v 显示所有远程仓库
- git remote add url 添加一个远程仓库
- git remote rm name 删除远程仓库
- git commit --amend -m "new_message" 重命名最新一次的 commit 记录

# setuptools
进行源码安装的工具
## Basic 使用
```
# setup.py 最小配置
from setuptools import setup
setup(
    name = "mypackage",     
    version = "0.0.1",
    install_requires =[
        'requests',
        'importlib-metadata; python_version == "3.8"',
    ],
)
```
最后将代码组织成这种形式
```
mypackage
├── pyproject.toml  # and/or setup.cfg/setup.py (depending on the configuration method)
|   # README.rst or README.md (a nice description of your package)
|   # LICENCE (properly chosen license information, e.g. MIT, BSD-3, GPL-3, MPL-2, etc...)
└── mypackage
    ├── __init__.py
    └── ... (other Python files)
```
运行以下命令构建 .whl 和 tar.gz（可以上传或安装）
```
python -m build
```

## 发现 Package
### 指定 Package
最简单的情况是
```
setup(
    # ...
    packages = ["mypkg", "mypkg.subpkg1", "mypkg.subpkg2"]  # install .whl 后 import mypkg/mypkg.subpkg
)
```
这里注意如果我们仅有 mypkg, 那么 mypkg.subpkg 是不会被 build 的。
如果 mypkg 不在当前目录的根目录下，那么可以使用 package_dir 配置。
```
setup(
    # ...
    package_dir = {"": "src"}   # build 的对象将包含 src/mypkg, src/mypkg/subpkg
    # package_dir = {
    #     "mypkg": "src",
    #     "mypkg/subpkg": "src",
    # },
)
```
默认情况下 setup.py 路径下的所有包都会被包含。
### find_package
find_packages() takes a source directory and two lists of package name patterns to exclude and include, and then returns a list of str representing the packages it could find.
```
mypkg
├── pyproject.toml  # AND/OR setup.cfg, setup.py
└── src
    ├── pkg1
    │   └── __init__.py
    ├── pkg2
    │   └── __init__.py
    ├── additional
    │   └── __init__.py
    └── pkg
        └── namespace
            └── __init__.py
```
```
form setuptools import find_packages

setup(
    # ...
    packages = find_packages(
        where = 'src',  # '.' by default
        include = ['pkg*'],   # ['*']  by default   
    ),
    package_dir = {"":"src"},
    # ...
)
```
find_packages 返回 ['pkg1', 'pkg2', 'pkg']，所以必须使用 package_dir。如果是子模块，那么字符串为 mypkg.subpkg.
- 问题是在 src 下的没有 __init__.py 文件的目录不会被考虑（查找），这需要使用 find_namespace_packages 解决.
### find_namespace_packages
```
foo
├── pyproject.toml  # AND/OR setup.cfg, setup.py
└── src
    └── timmins
        └── foo
            └── __init__.py
```
需要使用
```
setup(
    # ...
    packages=find_namespace_packages(where='src'),
    package_dir={"": "src"}
    # ...
)
```
安装 .whl 后，import timmins.foo 使用.
### 检查 build 是否正确
在 build 你的文件之后，使用
```
tar tf dist/*.tar.gz
unzip -l dist/*.whl
```
查看是否有包没有被包含.
## 依赖管理
有三种依赖:
- build system requirement
- required dependency
- optional dependency
### build system requirement
如果是 setup.py，无需指定
### required dependency
```
setup(
    # ...
    install_requires=[
        'docutils',
        'BazSpam=1.1',
    ],
)
```
当 install project 时，所有未被安装的 dependency 都会被下载、build、安装。  
URL dependency
```
setup(
    # ...
    install_requires=[
        "Package-A @ git+https://example.net/package-a.git@main",
    ],
    # ...
)
```
### 最低 python 版本
```
setup(
    python_requires=">=3.6",
    # ...
)
```
## Build extension modules
setuptools can buld C/C++ extension modules. setup() 函数中的参数 ext_modules 接收 setuptools.Extension 对象列表. 例如
```
<project_folder>
├── pyproject.toml
└── foo.c
```
指示 setuptools 将 foo.c 编译成扩展模块 mylib.foo, 我们需要
```
from setuptools import setup, Extension

setup(
    ext_modules = [
        Extension(
            name="mylib.foo",   # as it would be imported
                                # may include packages/namespaces separated by '.'
            sources = ["foo.c"], # all sources are compiled into a single binary file
        ),
    ]
)
```
### compiler and linker 选项