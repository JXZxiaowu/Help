# Tesnor
## 存储布局 layout
常见的存储布局方式
1. Strieded layout / 步幅布局：Tesnor.stride() 定义了在内存中访问元素所需移动的字节数，通过改变 stride 属性，可以改变 Tensor 的视图而不移动或复制底层数据
2. Contiguous layout / 连续布局: 张量的数据在内存中按照连续的方式进行存储
3. Channels last layout / 末尾通道布局: 在末尾通道布局中，通道维度是最后一个维度，可以提高部分操作的性能
# 自动求导机制
使用 Function 类构成图来完成自动求导的. 每次循环都会从头创建 graph, 这也正是 torch 能使用控制语句的原因.

## Save tensors for backward
通过在 Function 类的 forward() 函数中调用 save_for_backward() 函数以为 backward pass 存储 Tensor.
## 局部禁用梯度计算
三种方法: 设置 requires_grad, no-grad mode, and inference mode
### requires_grad
在前向传播的过程中，只有当一个计算的至少一个输入 Tensor reqquire grad，该计算才被记录后向传播的 graph 中. 在 leaf tensor 中, 只有 requires_grad = True 的 tensor 才会将梯度累加到 grad 属性中.

虽然所有的 Tensor 都有 requires_grad 属性，但是其只对叶子 Tensor 有意义. 在代码中，如果将非叶子节点 Tensor.requires_grad = False, 叶子节点 Tensor 的 grad 依然能够被正确的计算.

如果想 freeze part of model, 将对应的参数应用 .requires_grad_(False). 另外 model.requires_grad_() 将会被应用到 model 的所有 parameters.
### Grad Mode (Default)
默认模式，在该模式下，参数是否更新梯度完全由 requires_grad 决定
### No-grad Mode
```
with torch.no-grad():
    # code
...
```
在 no-grad mode 中的计算不会被 backward graph 记录，即使 inputs.requires_grad=True. 该模式使得我们在禁用某一块代码或函数的梯度计算很方便，我们可以不用短暂地设置 Tensor.requires_grad=False and then back to True.
### inference mode
inference mode 是 no-grad mode 的一个极端模式，它允许 Torch 更快地进行计算. 缺点是在 inference mode 下创建地 Tensor，在 inference mode 结束后如果被记录到 backward graph 中会触发错误.

如果该模式无法正确运行代码，那么退回到 no-grad mode! :D
### evaluation mode
**Evaluation mode 并不是用来禁用梯度计算的机制**. 无论是训练模式还是 evaluation 模型，Torch 都会构建 backward graph 用以计算梯度. evaluation 模式主要是为了影响某些层 (训练阶段和评估阶段行为不一致地层) 的行为.
## 并发的不确定性
>If you are calling backward() from multiple threads concurrently and have shared inputs (i.e. Hogwild CPU training), then non-determinism should be expected.
# 自动求导包 TORCH.AUTOGRAD
## 叶子和非叶子节点
在 Python 中，可以通过检察 Tensor.grand_fn 区分叶子节点和非叶子节点
- 叶子节点: 用户创建的 Tensor, 而非通过张量操作生成的 Tensor, 叶子结点的 grad_fn 为 None.
- 非叶子节点: 非叶子节点是由张量操作生成的新张量，它们的 grad_fn 属性指向创建它们的张量操作. 非叶子节点的张量数据通常不会被存储在内存中，它们仅存储操作的引用和相关元数据。实际的数据存储在叶子节点中，并根据需要进行动态计算和更新.
## 求导函数
```
torch.autograd.backward(tensors, grad_tensors=None, retain_graph=False, create_graph=False)
```
用于计算给定 Tensors 相对于 graph leaves 的导数，即 Tensors'. 它会将 Tensor 的梯度累加到 Tensor.grad 属性中
- tensors: list[Tensor] or Tensor - 一般为损失函数
- grad_tensors: 与 tensors 形状相同的张量，用于给定梯度的权重。默认为None，表示所有张量的梯度权重都为1
- retain_graph: 表示是否保留计算图以供后续计算梯度使用。默认为False，表示在反向传播完成后会释放计算图
- create_graph: 表示是否创建计算高阶导数所需的图。默认为False，一般只在需要计算高阶导数时设置为True
- inputs: list[Tensors] or Tensors - 默认为 None, 表示计算所有 leaves 的 gred, 指定后其他的 Tesnor grad 将会忽略

```
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)
```
用于计算目标张量相对于输入张量的梯度，返回梯度张量. 输出结果是一个元组，其中包含与 inputs 中每个张量对应的梯度张量。如果某个输入张量不可微或未被使用，则对应的梯度张量为 None
## gradient 布局
如果 para.grad is None, 那么会创建和张量存储布局相同的 gradient layout; 如果 para.grad 已经被创建了，那么进行累加.

model.zero_grad() 或 optimizer.zero_grad() 旨在将 para.grad 重置为 None，如
```
for iterations ...
    ...
    for param in model.parameters():
        param.grad = None
    loss.backward()
```
不同的是，model.zero_grad 将模型中所有可学习参数（具有requires_grad=True）的梯度属性重置; optimizer.zero_grad 将优化器中管理的参数的梯度属性重置.
## Tensor 的原地操作
无论是叶子张量还是非叶子张量，原地操作可能破坏计算图的结构，从而导致梯度计算不准确或无法进行。为了确保梯度计算的正确性和稳定性，建议尽量避免，以下是一些原地操作的例子:
· x += y, or x.add_(y)
- x *=y, or x.mul_(y)
- x.neg_() 原地取负
- 原地切片赋值 x[index] = y

如果某个 param.grad 已被保存, 但后续 param 又被原地修改了，那么在进行反向传播时会引发错误。
```
import torch

# 创建一个需要梯度计算的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 对张量进行一系列操作，构建计算图
y = x * 2
z = y + 1

# x.grad 已被保存
z.backward(retain_graph=True)

# 原地修改张量的值
x.mul_(2)

# 进行反向传播，引发错误
z.backward()
```
下面是正确的
```
import torch

# 创建一个需要梯度计算的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 对张量进行一系列操作，构建计算图
y = x * 2
z = y + 1

# 将张量保存用于反向传播
z.backward(retain_graph=True)

# 分离张量并创建一个新的张量进行原地操作
x_detached = x.detach()
x_detached.mul_(2)

# 进行反向传播，梯度计算正确
z.backward()
```
detach() 用于将一个张量从计算图中分离出来，创建一个新的张量，该新张量与原始张量共享相同的底层数据，但不再具有梯度属性。
## Tensor 梯度函数
```
torch.Tensor.is_leaf
```
在调用 backward() 后, 只有 leaf Tensors 的 grad 属性会被填充，如果想要填充非叶子节点的 grad 属性，可以使用 Tensor.retain_grad().
```
Tensor.register_hook(hook)
```
The hook will be called every time a gradient with respect to the Tensor is computed. hook 是一个这样形式的函数: hook(grad) -> Tensor or None. hook 不应该改变 grad 而是返回一个新的 grad 进行替换，
## torch.autograd.Function
用于定义自定义的操作和自动微分函数。它提供了一种扩展 PyTorch 框架功能的方式，允许用户自定义计算图节点的前向传播和反向传播方法。

创建子类继承 Function 类并实现 forward() 和 backward() 静态方法，然后在前向传播中调用 apply() 而不要直接调用 forward(). 这样做的好处是，apply() 方法会处理操作的注册和跟踪，以确保操作能够正确地与计算图的其他部分交互和连接.

例子，在实际使用中通过调用 Exp.apply() 来应用这个自定义操作
```
class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result
# Use it by calling the apply method:
output = Exp.apply(input)
```
最后使用 gradcheck() 检查梯度计算的正确性
## gradcheck()
通过对比解析梯度 (analytical gradient) 和数据梯度 (numberical gradient) 来验证后向传播的正确性.
## 性能分析工具
```
with torch.autograd.profiler.profile(
    use_cuda=False,        # 是否使用 CUDA 进行分析，默认为 False
    profile_memory=False,  # 是否分析内存使用情况，默认为 False
    record_shapes=False,   # 是否记录张量的形状，默认为 False
    profile_name=None     # 保存剖析结果的文件名，默认为 None
) as prof:
    # 执行需要进行性能分析的代码块
    y = model(x)
    y.backward()
# 在块外
print(prof)
```
## 异常检测 / Anomaly detection
```
CLASS torch.autograd.detect_anomaly(check_nan=True)
```
用于异常检测的上下文管理器.
- 将前向和后向传播过程加入到该类的上下文管理块内，该类将会回溯发生错误的 backward 对应的 forward;
- 如果 check_nan = True, 当 backward 计算产生 nan 值时会触发错误.

因为会拖累程序的运行，下面的版本将允许我们控制是否开启检测
```
CLASS torch.autograd.set_detect_anomaly(mode, check_nan=True)
# mode (bool) - 是否开启检测
```
## 梯度 graph
非叶子 Tensor 的 grad_fn 是 torch.autograd.graph.node 的 holder，该类支持我们检查 backward pass.

```
graph.node.name()  # 返回名字，操作的名字
graph.node.metadata() # 返回元数据, 可以理解为 forward pass 的输入
```
对非叶子 Tensor, 我们也能注册 hook.
```
graph.node.register_hook()
```
The hook will be called every time a gradient with respect to the Node is computed, hook 的形式是 `hook(grad_inputs: Tuple[Tensor], grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None`, 与 Tensor.register_hook() 相同的是我们应该通过返回一个 Tensor 来代替 grad_outputs.
