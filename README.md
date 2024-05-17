查看运行结果：
任何安装了tensorbord的环境,查看结果运行:
tensorboard --logdir /home/peace/Downloads/Log/BSNN --port=1234

运行：
账号:peace 密码:peace
cd /home/peace/Downloads/
conda activate webui

直接运行：
python ./BSNN/main.py

后台运行：
nohup python ./BSNN/main.py > ./BSNN/main.log 2>&1 &


ps :
！！！！！！！！！！
再次运行会清除上一次运行的记录，请务必修改main函数中的 exp_name ，以及trainable函数中的 torch.save(network.state_dict(),f'/home/peace/Downloads/checkpoints/{name}.pth')即可
！！！！！！！！！！
默认会拉满两张显卡，如果显卡在运行其他程序会爆显存

选择显卡，使用:
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

限制总可用资源，使用:
ray.init(num_cpus=16, num_gpus=2,)

关于参数:
    config = {
        'binarize_weight':tune.grid_search(['ste','xnor','none']),
        'norm_method':tune.grid_search(['BN','BNTT','tdBN','TEBN']),
        'binarize_norm':tune.grid_search([True,False]),
        'pool_method':tune.grid_search(['max','avg']),
        'net_structure':tune.grid_search(['CBNPBN','CBNPXN','CBNPXX','CXXPBN'])
    }
'binarize_weight'---二值化权重:
    'ste'---使用ste进行二值化
    'xnor'----使用xnor进行二值化
    'none'---不进行二值化

'norm_method'---norm方法选择:
    'BN'---普通BatchNorm
    'BNTT'---
    'tdBN'---
    'TEBN'---

'binarize_norm'---是否使用shiftBN:
    'True'---开启
    'False'---关闭

'pool_method'---pool的方法:
    'max'--maxpooling
    'avg'--avgpooling

net_structure--网络结构:
    'C'---卷积
    'B'---Norm
    'N'---neuron
    'P'---pooling
    'X'---空(None)