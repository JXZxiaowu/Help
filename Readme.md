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
## 常用命令
- git branch -a : 查看所有分支
- git branch -r : 查看所有远程分支
- git log  查看提交记录
- git reflog  可查看修改记录，包括回退记录
- git reset --hard {commit id} 回退版本
- git stash 未被提交的代码放到暂存区
- git stash pop 还原并清除最近一次的暂存记录
- git remote -v 显示所有远程仓库
- git remote add url 添加一个远程仓库
- git remote rm name 删除远程仓库
- git commit --amend -m "new_message" 重命名最新一次的 commit 记录
