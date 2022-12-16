from torchstat import stat
# @desc 评价模型的参数复杂度
# @author frankliu
# @time 2022.7.29


# 输入模型返回模型参数
def get_info(model):
    # 打印模型参数
    # for param in model.parameters():
    #     print(param)
    # 打印模型名称与shape
    print('模型的参数信息如下：')
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())


# 模型参数的可训练量
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'总数': total_num, '可训练数': trainable_num})


# 使用stat返回参数的总量
def get_from_stat(model, c, h, w):
    print('第三方检查如下：')
    stat(model, (c, h, w))

