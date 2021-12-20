import os
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataset import TotalText, Ctw1500Text, Icdar15Text, Mlt2017Text, TD500Text
from network.textnet import TextNet
from network.loss import TextLoss
from util.augmentation import BaseTransform, Augmentation, SquarePadding
from cfglib.config import config as cfg, update_config, print_config
from cfglib.option import BaseOptions
from util.visualize import visualize_detection, visualize_gt
from util.misc import to_device, mkdirs,rescale_result, AverageMeter
from util.eval import deal_eval_total_text, deal_eval_ctw1500, deal_eval_icdar15, \
    deal_eval_TD500, data_transfer_ICDAR, data_transfer_TD500, data_transfer_MLT2017

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')


def inference(model, test_loader, output_dir):

    total_time = 0.
    if cfg.exp_name != "MLT2017":
        osmkdir(output_dir)
    else:
        if not os.path.exists(output_dir):
            mkdirs(output_dir)
    for i, (image, meta) in enumerate(test_loader):
        input_dict = dict()

        input_dict['img'] = to_device(image)
        # get detection result
        start = time.time()
        torch.cuda.synchronize()
        output_dict = model(input_dict)
        end = time.time()
        if i > 0:
            total_time += end - start
            fps = (i + 1) / total_time
        else:
            fps = 0.0
        idx = 0  # test mode can only run with batch_size == 1
        print('detect {} / {} images: {}.'.format(i + 1, len(test_loader), meta['image_id'][idx]))

        # visualization
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        show_boundary, heat_map = visualize_detection(img_show, output_dict, meta=meta)

        contours = output_dict["py_preds"][-1].int().cpu().numpy()

        gt_contour = []
        label_tag = meta['label_tag'][idx].int().cpu().numpy()
        for annot, n_annot in zip(meta['annotation'][idx], meta['n_annotation'][idx]):
            if n_annot.item() > 0:
                gt_contour.append(annot[:n_annot].int().cpu().numpy())

        gt_vis = visualize_gt(img_show, gt_contour, label_tag)

        show_map = np.concatenate([heat_map, gt_vis], axis=1)
        show_map = cv2.resize(show_map, (320 * 3, 320))
        im_vis = np.concatenate([show_map, show_boundary], axis=0)

        path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name), meta['image_id'][idx].split(".")[0]+".jpg")
        cv2.imwrite(path, im_vis)

        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        img_show, contours = rescale_result(img_show, contours, H, W)

        # write to file
        if cfg.exp_name == "Icdar2015":
            fname = "res_" + meta['image_id'][idx].replace('jpg', 'txt')
            contours = data_transfer_ICDAR(contours)
            write_to_file(contours, os.path.join(output_dir, fname))
        elif cfg.exp_name == "MLT2017":

            out_dir = os.path.join(output_dir, str(cfg.checkepoch))
            if not os.path.exists(out_dir):
                mkdirs(out_dir)
            fname = meta['image_id'][idx].split("/")[-1].replace('ts', 'res')
            fname = fname.split(".")[0] + ".txt"
            data_transfer_MLT2017(contours, os.path.join(out_dir, fname))
        elif cfg.exp_name == "TD500":
            fname = "res_" + meta['image_id'][idx].split(".")[0]+".txt"
            data_transfer_TD500(contours, os.path.join(output_dir, fname))

        else:
            fname = meta['image_id'][idx].replace('jpg', 'txt')
            write_to_file(contours, os.path.join(output_dir, fname))


def main(vis_dir_path):

    osmkdir(vis_dir_path)
    if cfg.exp_name == "Totaltext":
        trainset = TotalText(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        testset = TotalText(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
            #transform=SquarePadding()
        )

    elif cfg.exp_name == "Ctw1500":
        trainset = Ctw1500Text(
            data_root='data/ctw1500',
            is_training=True,
            transform=Augmentation(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
        testset = Ctw1500Text(
            data_root='data/ctw1500',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
            #transform=SquarePadding()
        )
    elif cfg.exp_name == "Icdar2015":
        trainset = Icdar15Text(
            data_root='data/Icdar2015',
            is_training=True,
            transform=Augmentation(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
        testset = Icdar15Text(
            data_root='data/Icdar2015',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
            #transform=SquarePadding()
        )
    elif cfg.exp_name == "MLT2017":
        trainset = Mlt2017Text(
            data_root='data/MLT2017',
            is_training=True,
            transform=Augmentation(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
        testset = Mlt2017Text(
            data_root='data/MLT2017',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
            #transform=SquarePadding()
        )
    elif cfg.exp_name == "TD500":
        trainset = TD500Text(
            data_root='data/TD500',
            is_training=True,
            transform=Augmentation(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
        testset = TD500Text(
            data_root='data/TD500',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
            #transform=SquarePadding()
        )
    else:
        print("{} is not justify".format(cfg.exp_name))
    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    # Model
    model = TextNet(is_training=True, backbone=cfg.net)
    #model_path = os.path.join(cfg.save_dir, cfg.exp_name,
    #                          'TextBPN_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
    #model.load_model(model_path)
    #model_path = os.path.join(cfg.save_dir, cfg.exp_name, 'Totaltext_resnet51_660.pth')
    #model = torch.load(model_path)
    # Create the loss criterion
    criterion = TextLoss()
    model = model.to(cfg.device) # copy to cuda
    criterion = criterion.to(cfg.device)
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
    #                            weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
    # Create the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    #loss_meter = AverageMeter()
    #batch_time = AverageMeter()
    #data_time = AverageMeter()
    for epoch in range(cfg.max_epoch):
        current_epoch = epoch + 1
        model.train(True)
        model.is_training = True
        model.BPN.is_training = True
        losses_meters = {
            'total_loss': AverageMeter(),
            'cls_loss': AverageMeter(),
            'distance_loss': AverageMeter(),
            'dir_loss': AverageMeter(),
            'point_loss': AverageMeter(),
            'norm_loss': AverageMeter(),
            'angle_loss': AverageMeter(),
        }
        train_one_epoch(model, optimizer, scheduler, train_loader, criterion, losses_meters, current_epoch)
        scheduler.step()
        model.is_training = False
        model.BPN.is_training = False
        model.eval()
        if cfg.cuda:
            cudnn.benchmark = True
        val_one_epoch(model, test_loader, current_epoch, cfg)
        if not os.path.exists(cfg.save_dir):
            mkdirs(cfg.save_dir)
        if current_epoch % cfg.save_freq == 0 or current_epoch == cfg.max_epoch:
            if current_epoch != cfg.max_epoch:
                model_path = os.path.join(cfg.save_dir, cfg.exp_name, f"{cfg.exp_name}_{cfg.net}_{current_epoch}.pth")
            else:
                model_path = os.path.join(cfg.save_dir, cfg.exp_name, f"{cfg.exp_name}_{cfg.net}_final.pth")
            checkpoints = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoints, model_path)
    print("Training End")

def train_one_epoch(model, optimizer, scheduler, train_loader, criterion, losses_meters, current_epoch):
    for i, (img, train_mask, tr_mask, distance_field,
            direction_field, weight_matrix, gt_points,
            proposal_points, ignore_tags) in enumerate(train_loader):

        img = to_device(img)
        
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, gt_points, \
        proposal_points, ignore_tags = to_device(img, 
                                                train_mask, tr_mask, distance_field,
                                                direction_field, weight_matrix, gt_points,
                                                proposal_points, ignore_tags)
        input_dict = dict()
        input_dict['img'] = to_device(img)
        input_dict['train_mask'] = to_device(train_mask)
        input_dict['tr_mask'] = to_device(tr_mask)
        input_dict['distance_field'] = to_device(distance_field)
        input_dict['direction_field'] = to_device(direction_field)
        input_dict['weight_matrix'] = to_device(weight_matrix)
        input_dict['gt_points'] = to_device(gt_points)
        input_dict['proposal_points'] = to_device(proposal_points)
        input_dict['ignore_tags'] = to_device(ignore_tags)
        output = model(input_dict)
        lossT = criterion(input_dict, output, current_epoch)
        loss = lossT['total_loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for loss_key_tmp in lossT.keys():
            losses_meters[loss_key_tmp].update(lossT[loss_key_tmp].item())

        if (i + 1) % cfg.display_freq == 0:
            print(f"Epoch [{current_epoch}] [{i + 1}/{len(train_loader)}]  lr: {scheduler.get_last_lr()}, total_loss: {losses_meters['total_loss'].avg:.4f}, cls_loss: {losses_meters['cls_loss'].avg:.4f}, distance_loss: {losses_meters['distance_loss'].avg:.4f}, dir_loss: {losses_meters['dir_loss'].avg:.4f}, point_loss: {losses_meters['point_loss'].avg:.4f}, norm_loss: {losses_meters['norm_loss'].avg:.4f}, angle_loss: {losses_meters['angle_loss'].avg:.4f}")

def val_one_epoch(model, test_loader, current_epoch, cfg):
    output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    if cfg.exp_name != "MLT2017":
        osmkdir(output_dir)
    else:
        if not os.path.exists(output_dir):
            mkdirs(output_dir)
    print(f"Start Evaluating Epoch(val) [{current_epoch}][{len(test_loader)}]")
    for i, (image, meta) in enumerate(test_loader):
        input_dict = dict()
        input_dict['img'] = to_device(image).to(device=cfg.device, dtype=torch.float)
        # get detection result
        torch.cuda.synchronize()
        output_dict = model(input_dict)
        idx = 0  # test mode can only run with batch_size == 1
        #print('detect {} / {} images: {}.'.format(i + 1, len(test_loader), meta['image_id'][idx]))
        # visualization
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
        contours = output_dict["py_preds"][-1].int().cpu().numpy()
        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        img_show, contours = rescale_result(img_show, contours, H, W)
        # write to file
        if cfg.exp_name == "Icdar2015":
            fname = "res_" + meta['image_id'][idx].replace('jpg', 'txt')
            contours = data_transfer_ICDAR(contours)
            write_to_file(contours, os.path.join(output_dir, fname))
        elif cfg.exp_name == "MLT2017":
            out_dir = os.path.join(output_dir, str(cfg.checkepoch))
            if not os.path.exists(out_dir):
                mkdirs(out_dir)
            fname = meta['image_id'][idx].split("/")[-1].replace('ts', 'res')
            fname = fname.split(".")[0] + ".txt"
            data_transfer_MLT2017(contours, os.path.join(out_dir, fname))
        elif cfg.exp_name == "TD500":
            fname = "res_" + meta['image_id'][idx].split(".")[0]+".txt"
            data_transfer_TD500(contours, os.path.join(output_dir, fname))
        else:
            fname = meta['image_id'][idx].replace('jpg', 'txt')
            write_to_file(contours, os.path.join(output_dir, fname))
    if cfg.exp_name == "Totaltext":
        deal_eval_total_text(debug=False)
    elif cfg.exp_name == "Ctw1500":
        deal_eval_ctw1500(debug=False)
    elif cfg.exp_name == "Icdar2015":
        deal_eval_icdar15(debug=False)
    elif cfg.exp_name == "TD500":
        deal_eval_TD500(debug=False)
    else:
        print("{} is not justify".format(cfg.exp_name))

if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    vis_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))

    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)
    # main
    main(vis_dir)
