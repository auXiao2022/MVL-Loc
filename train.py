import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #,1,2,3,4,5,6,7"

import torch
import sys
import time
import os.path as osp
import numpy as np
import tqdm

from torch.backends import cudnn
cudnn.benchmark = True

from tensorboardX import SummaryWriter
from tools.options import Options
from network.atloc import AtLoc, AtLocPlus
from network.efficientvit import EfficientViT_M0
from network.efficientvit import EfficientViT_M3
from network.efficientvit import EfficientViT_M4
from network.efficientvit import EfficientViT_M5
from torchvision import transforms, models
from tools.utils import AtLocCriterion, AtLocPlusCriterion, AverageMeter, Logger
from dataloaders import SevenScenes, RobotCar, MF
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tools.utils import quaternion_angular_error, qexp
from CameraPoseDataset import CameraPoseDataset
#from torchstat import stat

# Config
opt = Options().parse()
cuda = torch.cuda.is_available()
device = "cuda:" + ",".join(str(i) for i in opt.gpus) if cuda else "cpu"
logfile = osp.join(opt.runs_dir, 'log.txt')
stdout = Logger(logfile)
print('Logging to {:s}'.format(logfile))
sys.stdout = stdout
#from network.moat import tiny_moat_2
# Model
#feature_extractor = EfficientViT_M3()
# #models.resnet34(pretrained=True) #resnet34 EfficientViT_M0()#
#feature_extractor = models.resnet18(pretrained=True)
# feature_extractor = models.vit_b_32(pretrained=False)

# feature_extractor = EfficientViT_M5() #tiny_moat_2(use_window=False, num_classes=10)
# checkpoint = torch.load("./pretrain/efficientvit_m5.pth", map_location='cpu')
# feature_extractor.load_state_dict(checkpoint['model'])

# Clip 提取特征
import os
import clip
import torch


# Load the model 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
model = model.float()
#model.eval()

Global_text = "Task: Camera Pose Regression"
text = clip.tokenize(Global_text).to(device)
global_feature = model.encode_text(text)
feature_extractor = model

atloc = AtLoc(feature_extractor, droprate=opt.train_dropout, pretrained=True, lstm=opt.lstm,text_feature=global_feature)

#stat(atloc,(3,256,256))

if opt.model == 'AtLoc':
    model = atloc
    train_criterion = AtLocCriterion(saq=opt.beta, learn_beta=True)
    val_criterion = AtLocCriterion()
    param_list = [{'params': model.parameters()}]
elif opt.model == 'AtLocPlus':
    model = AtLocPlus(atlocplus=atloc)
    kwargs = dict(saq=opt.beta, srq=opt.gamma, learn_beta=True, learn_gamma=True)
    train_criterion = AtLocPlusCriterion(**kwargs)
    val_criterion = AtLocPlusCriterion()
else:
    raise NotImplementedError

# Optimizer
param_list = [{'params': model.parameters()}]
if hasattr(train_criterion, 'sax') and hasattr(train_criterion, 'saq'):
    print('learn_beta')
    param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
if opt.gamma is not None and hasattr(train_criterion, 'srx') and hasattr(train_criterion, 'srq'):
    print('learn_gamma')
    param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
optimizer = torch.optim.AdamW(param_list, lr=opt.lr, weight_decay=opt.weight_decay)

stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
stats = np.loadtxt(stats_file)
#tforms = [transforms.Resize(opt.cropsize,antialias=False)]
# add for all camerapose dataset
tforms = [transforms.ToPILImage()]
tforms.append(transforms.Resize(opt.cropsize))
tforms.append(transforms.RandomCrop(opt.cropsize))
if opt.color_jitter > 0:
    assert opt.color_jitter <= 1.0
    print('Using ColorJitter data augmentation')
    tforms.append(transforms.ColorJitter(brightness=opt.color_jitter, contrast=opt.color_jitter, saturation=opt.color_jitter, hue=0.5))
else:
    print('Not Using ColorJitter')
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
data_transform = transforms.Compose(tforms)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# Load the dataset
kwargs = dict(scene=opt.scene, data_path=opt.data_dir, transform=data_transform, target_transform=target_transform, seed=opt.seed)
if opt.model == 'AtLoc':
    if opt.dataset == '7Scenes':
        #train_set = SevenScenes(train=True, **kwargs)
        #val_set = SevenScenes(train=False, **kwargs)
        train_set = CameraPoseDataset("./data/7Scenes/", "./data/7Scenes/7scenes_all_scenes.csv", data_transform=data_transform)
        val_set = CameraPoseDataset("./data/7Scenes/", "./data/7Scenes/abs_7scenes_pose.csv_chess_test.csv", data_transform=data_transform)
    elif opt.dataset == 'RobotCar':
        train_set = RobotCar(train=True, **kwargs)
        val_set = RobotCar(train=False, **kwargs)
    else:
        raise NotImplementedError
elif opt.model == 'AtLocPlus':
    kwargs = dict(kwargs, dataset=opt.dataset, skip=opt.skip, steps=opt.steps, variable_skip=opt.variable_skip)
    train_set = MF(train=True, real=opt.real, **kwargs)
    val_set = MF(train=False, real=opt.real, **kwargs)
else:
    raise NotImplementedError
kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(train_set, batch_size=opt.batchsize, shuffle=True, **kwargs)
#val_loader = DataLoader(val_set, batch_size=opt.batchsize, shuffle=False, **kwargs)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)  #for logs with error in trans and rotation

model.to(device)
train_criterion.to(device)
val_criterion.to(device)
L = len(val_set)
total_steps = opt.steps
writer = SummaryWriter(log_dir=opt.runs_dir)
experiment_name = opt.exp_name
# #test time
# repetitions = 300
#
# dummy_input = torch.rand(1, 3, 256, 256).to(device)
#
# # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
# print('warm up ...\n')
# with torch.no_grad():
#     for _ in range(100):
#         _ = model(dummy_input)
#
# # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
# torch.cuda.synchronize()
#
#
# # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# # 初始化一个时间容器
# timings = np.zeros((repetitions, 1))
#
# print('testing ...\n')
# with torch.no_grad():
#     for rep in tqdm.tqdm(range(repetitions)):
#         starter.record()
#         _ = model(dummy_input)
#         ender.record()
#         torch.cuda.synchronize() # 等待GPU任务完成
#         curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
#         timings[rep] = curr_time
#
# avg = timings.sum()/repetitions
# print('\navg={}\n'.format(avg))
# print("!!!!!!!!!!!!!!!!!!!!S")


lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=330,eta_min=6e-6)

#
# for epoch in range(opt.epochs):
#     if epoch % opt.val_freq == 0 or epoch == (opt.epochs - 1):
#         val_batch_time = AverageMeter()
#         val_loss = AverageMeter()
#         model.eval()
#         end = time.time()
#         val_data_time = AverageMeter()
#
#         for batch_idx, (val_data, val_target) in enumerate(val_loader):
#             val_data_time.update(time.time() - end)
#             val_data_var = Variable(val_data, requires_grad=False)
#             val_target_var = Variable(val_target, requires_grad=False)
#             val_data_var = val_data_var.to(device)
#             val_target_var = val_target_var.to(device)
#
#             with torch.set_grad_enabled(False):
#                 val_output = model(val_data_var)
#                 val_loss_tmp = val_criterion(val_output, val_target_var)
#                 val_loss_tmp = val_loss_tmp.item()
#
#             val_loss.update(val_loss_tmp)
#             val_batch_time.update(time.time() - end)
#
#             writer.add_scalar('val_err', val_loss_tmp, total_steps)
#             if batch_idx % opt.print_freq == 0:
#                 print('Val {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}' \
#                       .format(experiment_name, epoch, batch_idx, len(val_loader) - 1, val_data_time.val, val_data_time.avg, val_batch_time.val, val_batch_time.avg, val_loss_tmp))
#             end = time.time()
#
#         print('Val {:s}: Epoch {:d}, val_loss {:f}'.format(experiment_name, epoch, val_loss.avg))
#
#         if epoch % opt.save_freq == 0:
#             filename = osp.join(opt.models_dir, 'epoch_{:03d}.pth.tar'.format(epoch))
#             checkpoint_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optim_state_dict': optimizer.state_dict(), 'criterion_state_dict': train_criterion.state_dict()}
#             torch.save(checkpoint_dict, filename)
#             print('Epoch {:d} checkpoint saved for {:s}'.format(epoch, experiment_name))
#
#     model.train()
#     train_data_time = AverageMeter()
#     train_batch_time = AverageMeter()
#     end = time.time()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         train_data_time.update(time.time() - end)
#
#         data_var = Variable(data, requires_grad=True)
#         target_var = Variable(target, requires_grad=False)
#         data_var = data_var.to(device)
#         target_var = target_var.to(device)
#
#         with torch.set_grad_enabled(True):
#             output = model(data_var)
#             loss_tmp = train_criterion(output, target_var)
#
#         loss_tmp.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         train_batch_time.update(time.time() - end)
#         writer.add_scalar('train_err', loss_tmp.item(), total_steps)
#         if batch_idx % opt.print_freq == 0:
#             print('Train {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}' \
#                   .format(experiment_name, epoch, batch_idx, len(train_loader) - 1, train_data_time.val, train_data_time.avg, train_batch_time.val, train_batch_time.avg, loss_tmp.item()))
#         end = time.time()
#
# writer.close()


# read mean and stdev for un-normalizing predictions
pose_stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev
# loss functions
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error


for epoch in range(opt.epochs):
    if epoch % opt.val_freq == 0 or epoch == (opt.epochs - 1):
        val_batch_time = AverageMeter()
        val_loss = AverageMeter()
        model.eval()
        end = time.time()
        val_data_time = AverageMeter()
        # add metric for eval:

        pred_poses = np.zeros((L, 7))  # store all predicted poses
        targ_poses = np.zeros((L, 7))  # store all target poses


        #add for test
        for batch_idx, (val_data, val_target, scene) in enumerate(val_loader):
            val_data_time.update(time.time() - end)
            val_data_var = Variable(val_data, requires_grad=False)
            val_target_var = Variable(val_target, requires_grad=False)
            val_data_var = val_data_var.to(device)
            val_target_var = val_target_var.to(device)

            with torch.no_grad():#set_grad_enabled(False):
                val_output = model(val_data_var,scene)
                output = val_output
                val_loss_tmp = val_criterion(val_output, val_target_var)
                val_loss_tmp = val_loss_tmp.item()

            val_loss.update(val_loss_tmp)
            val_batch_time.update(time.time() - end)

            writer.add_scalar('val_err', val_loss_tmp, total_steps)
            if False: #batch_idx % opt.print_freq == 8:
                print('Val {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}' \
                      .format(experiment_name, epoch, batch_idx, len(val_loader) - 1, val_data_time.val, val_data_time.avg, val_batch_time.val, val_batch_time.avg, val_loss_tmp))
            end = time.time()

            s = output.size()
            output = output.cpu().data.numpy().reshape((-1, s[-1]))
            target = val_target.numpy().reshape((-1, s[-1]))

            # normalize the predicted quaternions
            q = [qexp(p[3:]) for p in output]
            output = np.hstack((output[:, :3], np.asarray(q)))
            q = [qexp(p[3:]) for p in target]
            target = np.hstack((target[:, :3], np.asarray(q)))

            # un-normalize the predicted and target translations
            output[:, :3] = (output[:, :3] * pose_s) + pose_m
            target[:, :3] = (target[:, :3] * pose_s) + pose_m

            # take the middle prediction
            pred_poses[batch_idx, :] = output[int(len(output) / 2)]
            targ_poses[batch_idx, :] = target[int(len(target) / 2)]

        print('Val {:s}: Epoch {:d}, val_loss {:f}'.format(experiment_name, epoch, val_loss.avg))
        # calculate losses
        t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
        q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])
        errors = np.zeros((L, 2))

        print('Error in translation: median {:3.2f} m,  mean {:3.2f} m '\
              '\nError in rotation: median {:3.2f} degrees, mean {:3.2f} degree'
              .format(np.median(t_loss), np.mean(t_loss), np.median(q_loss), np.mean(q_loss)))
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs - 1):
            filename = osp.join(opt.models_dir, 'epoch_{:03d}.pth.tar'.format(epoch))
            checkpoint_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optim_state_dict': optimizer.state_dict(), 'criterion_state_dict': train_criterion.state_dict()}
            torch.save(checkpoint_dict, filename)
            print('Epoch {:d} checkpoint saved for {:s}'.format(epoch, experiment_name))

    model.train()
    train_data_time = AverageMeter()
    train_batch_time = AverageMeter()
    end = time.time()
    for batch_idx, (data, target,scene) in enumerate(train_loader): #scene is for test
        train_data_time.update(time.time() - end)

        data_var = Variable(data, requires_grad=True)
        target_var = Variable(target, requires_grad=False)
        data_var = data_var.to(device)
        target_var = target_var.to(device)

        #with torch.set_grad_enabled(True):
        output = model(data_var,scene)
        loss_tmp = train_criterion(output, target_var)

        loss_tmp.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_batch_time.update(time.time() - end)
        writer.add_scalar('train_err', loss_tmp.item(), total_steps)
        if batch_idx % opt.print_freq == 0:
            print('Train {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}' \
                  .format(experiment_name, epoch, batch_idx, len(train_loader) - 1, train_data_time.val, train_data_time.avg, train_batch_time.val, train_batch_time.avg, loss_tmp.item()))
        end = time.time()
    print(train_criterion.sax, train_criterion.saq)
    lr_scheduler.step()
writer.close()


