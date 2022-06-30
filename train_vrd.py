import json
import logging
import os
import torch
import time
from dataloader import Load_dataset
from model_multi import Referring_relationships_Model
from utils import weighted_cross_entropy, iou, cross_entropy


if __name__=='__main__':
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with open('config.json', 'r') as load_f:
        cfg = json.load(load_f)
    torch.cuda.set_device(cfg['device'][0])

    net = Referring_relationships_Model(config=cfg)
    device_out = 'cuda:%d' % (cfg['device'][0])
    net = torch.nn.DataParallel(net, device_ids=cfg['device'])
    net = net.cuda(cfg['device'][0])
    net.train()

    dataset_train = Load_dataset(mode='train', config=cfg)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg['train_batch_size'], shuffle=True)
    dataset_val = Load_dataset(mode='val', config=cfg)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=cfg['val_batch_size'], shuffle=False)
    dataset_test = Load_dataset(mode='test', config=cfg)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=cfg['test_batch_size'], shuffle=False)

    lr_base = cfg['lr_base']
    lr_gamma = cfg['lr_gamma']
    lr_schedule = cfg['lr_schedule']

    opt = torch.optim.RMSprop(net.parameters(),lr = lr_base)

    WARM_UP_ITERS = 500
    WARM_UP_FACTOR = 1.0 / 3.0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.7, patience=3, verbose=True)
    
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    wr_step = 0

    val_s_result_list = []
    val_o_result_list = []
    best_val_result = 0.0
    best_test_s_result = 0.0
    best_test_o_result = 0.0
    for epoch in range(0, cfg['epochs']):
        for i, (input_img, input_subj, input_rel, input_obj, s_regions, o_regions) in enumerate(loader_train):
            time_start = time.time()

            if wr_step < WARM_UP_ITERS:
                lr = lr_base
                alpha = float(wr_step) / WARM_UP_ITERS
                warmup_factor = WARM_UP_FACTOR * (1.0 - alpha) + alpha
                lr = lr*warmup_factor 

                for param_group in opt.param_groups:
                    param_group['lr'] = lr
                wr_step +=1
            
            pred_s_regions, pred_o_regions = net(input_img, input_subj, input_rel, input_obj)
            
            s_regions = s_regions.cuda()
            o_regions = o_regions.cuda()
            
            s_loss = weighted_cross_entropy(pred_s_regions, s_regions , cfg["w1"]) 
            o_loss = weighted_cross_entropy(pred_o_regions, o_regions , cfg["w1"])      

            s_iou = iou(pred_s_regions, s_regions , cfg["heatmap_threshold"]) 
            o_iou = iou(pred_o_regions, o_regions , cfg["heatmap_threshold"])
            loss = s_loss+o_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
            opt.step()
            maxmem = int(torch.cuda.max_memory_allocated(device=cfg["device"][0]) / 1024 / 1024)
            time_end = time.time()
            totaltime = int((time_end - time_start) * 1000)
            for param_group in opt.param_groups:
                lr_now = param_group['lr']
            if i%100 == 0 or wr_step < WARM_UP_ITERS:
                print('epoch:%d, step:%d, loss:%f, s_loss:%f, o_loss:%f, s_iou:%f, o_iou:%f, maxMem:%dMB, time:%dms, lr:%f' % \
                    (epoch, i, loss, s_loss, o_loss, s_iou, o_iou, maxmem, totaltime, lr_now))

        with torch.no_grad():
            net.eval()
            # val
            val_s_result = 0.0
            val_o_result = 0.0
            val_result = 0.0
            for i, (input_img, input_subj, input_rel, input_obj, s_regions, o_regions) in enumerate(loader_val):
                pred_s_regions, pred_o_regions = net(input_img, input_subj, input_rel, input_obj)
                
                s_regions = s_regions.cuda()
                o_regions = o_regions.cuda()
                
                s_iou = iou(pred_s_regions, s_regions , cfg["heatmap_threshold"]) 
                o_iou = iou(pred_o_regions, o_regions , cfg["heatmap_threshold"])
        
                val_s_result += s_iou.item()
                val_o_result += o_iou.item()
        
            val_s_result_list.append(val_s_result/(i+1))
            val_o_result_list.append(val_o_result/(i+1))
            val_result = val_s_result/(i+1)+val_o_result/(i+1)
            scheduler.step(val_result)
            print('val_s_iou:', val_s_result_list)
            print('val_o_iou:', val_o_result_list)

            if val_result>best_val_result:
                torch.save(net.state_dict(),'models/n_vrd/net_base_best.pkl')
                best_val_result = val_result

                # test
                test_s_result = 0.0
                test_o_result = 0.0
                for i, (input_img, input_subj, input_rel, input_obj, s_regions, o_regions) in enumerate(loader_test):
                    pred_s_regions, pred_o_regions = net(input_img, input_subj, input_rel, input_obj)
                    
                    s_regions = s_regions.cuda()
                    o_regions = o_regions.cuda()
                    
                    s_iou = iou(pred_s_regions, s_regions , cfg["heatmap_threshold"]) 
                    o_iou = iou(pred_o_regions, o_regions , cfg["heatmap_threshold"])
            
                    test_s_result += s_iou.item()
                    test_o_result += o_iou.item()

                best_test_s_result = test_s_result/(i+1)
                best_test_o_result = test_o_result/(i+1)
                print('test_s_iou:', best_test_s_result)
                print('test_o_iou:', best_test_o_result)

            net.train()
    print('test_s_iou:', best_test_s_result)
    print('test_o_iou:', best_test_o_result)


