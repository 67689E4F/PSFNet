import time, os, torch
from core.evaluate import accuracy
from utils.utils import save_batch_heatmaps

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def lossFun(output, target):
    return None

def run_epoch(config, data_loader, model_defocusMap, model_discriminator, criterions, optimizers, epoch,
          output_dir, writer_dict, device, logger, isTrain):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_defocusMap = AverageMeter()
    losses_D = AverageMeter()
    losses_perceptual = AverageMeter()
    acc = AverageMeter()
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.


    if isTrain:
        model_defocusMap.train()
        model_discriminator.train()
    else:
        model_defocusMap.eval()
        model_discriminator.eval()

    end = time.time()
    for i, (img_syndof, label_syndof, img_ded, label_ded) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img_syndof = img_syndof.to(device)
        label_syndof = label_syndof.to(device)

        img_ded = img_ded.to(device)
        label_ded = label_ded.to(device)

        
        defocusMap_ded = model_defocusMap(img_ded)

        model_discriminator.zero_grad()
        b_size = img_syndof.size(0)
        label_D = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = model_discriminator(defocusMap_ded).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterions[0](output, label_D)
        # Calculate gradients for D in backward pass
        if isTrain:
            errD_real.backward(retain_graph=True)
        D_x = output.mean().item()

        defocusMap_syndof = model_defocusMap(img_syndof)
        ## Train with all-fake batch
        label_D.fill_(fake_label)
        # Classify all fake batch with D
        output = model_discriminator(defocusMap_syndof.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterions[0](output, label_D)
        if isTrain:
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward(retain_graph=True)
            # Update D
            optimizers[0].step()

        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        # optimizers[0].step()


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        model_defocusMap.zero_grad()

        label_D.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = model_discriminator(defocusMap_syndof).view(-1)
        # Calculate G's loss based on this output
        errG = criterions[0](output, label_D)
        errMapded = criterions[1](defocusMap_ded, label_ded)
        errMapSyndof = criterions[1](defocusMap_syndof, label_syndof)

        perceptual_sy = criterions[2](defocusMap_syndof.repeat_interleave(3, dim=1), label_syndof.repeat_interleave(3, dim=1))
        perceptual_real = criterions[2](defocusMap_ded.repeat_interleave(3, dim=1), label_ded.repeat_interleave(3, dim=1))


        # edge_sy = criterions[2](defocusMap_syndof, label_syndof)
        # edge_real = criterions[2](defocusMap_syndof, label_syndof)

        errAll = (errG+ perceptual_sy+perceptual_real) * 1e-3 + errMapded + errMapSyndof
        
        # Calculate gradients for G
        if isTrain:
            errAll.backward()
            # Update G
            optimizers[1].step()

        D_G_z2 = output.mean().item()
        

        # measure accuracy and record loss
        losses_defocusMap.update(errAll.item(), b_size)
        losses_D.update(errD.item(), b_size)
        losses_perceptual.update((perceptual_sy+perceptual_real).item(), b_size)

        avg_acc = accuracy(defocusMap_ded.detach().cpu().numpy(),
                                          label_ded.detach().cpu().numpy())
        acc.update(avg_acc)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            if isTrain:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                    'Loss_D {loss_D.val:.5f} ({loss_D.avg:.5f})\t' \
                    'Loss_p {loss_p.val:.5f} ({loss_p.avg:.5f})\t' \
                    'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, len(data_loader), batch_time=batch_time,
                        speed=b_size/batch_time.val,
                        data_time=data_time, loss=losses_defocusMap,loss_D = losses_D, loss_p = losses_perceptual,
                        acc=acc
                        )
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_losses_defocusMap', losses_defocusMap.val, global_steps)
                writer.add_scalar('train_losses_discriminator', losses_D.val, global_steps)
                writer.add_scalar('train_losses_perceptual', losses_perceptual.val, global_steps)

                

                writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

                prefix = '{}_{}_{}.jpg'.format(os.path.join(output_dir, 'train'), epoch, i)
                save_batch_heatmaps(img_ded, defocusMap_ded, prefix)
            else:
                msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Loss_D {loss_D.val:.5f} ({loss_D.avg:.5f})\t' \
                        'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                            i, len(data_loader), batch_time=batch_time,
                            loss=losses_defocusMap,loss_D = losses_D
                            , acc=acc)
                prefix = '{}_{}_{}.jpg'.format(os.path.join(output_dir, 'test'), epoch, i)
                save_batch_heatmaps(img_ded, defocusMap_ded, prefix)

                logger.info(msg)

                if writer_dict:
                    writer = writer_dict['writer']
                    global_steps = writer_dict['valid_global_steps']
                    writer.add_scalar(
                        'valid_loss',
                        losses_defocusMap.avg,
                        global_steps
                    )
                    writer.add_scalar(
                        'valid_acc',
                        acc.avg,
                        global_steps
                    )

                    writer_dict['valid_global_steps'] = global_steps + 1
    return losses_defocusMap.val, losses_D.val, acc.val


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, device, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses= AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, coc_label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = img.to(device)
        # compute output
        outputs_coc = model(img)

        coc_label = coc_label.to(device)
        
        loss = criterion(outputs_coc, coc_label)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), img.size(0))

        avg_acc = accuracy(outputs_coc.detach().cpu().numpy(),
                                          coc_label.detach().cpu().numpy())
        acc.update(avg_acc)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=img.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, 
                      acc=acc
                      )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}_{}.jpg'.format(os.path.join(output_dir, 'train'), epoch, i)
            save_batch_heatmaps(img, outputs_coc, prefix)
def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, device, logger, writer_dict=None,epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (img, coc_label) in enumerate(val_loader):
            # compute output
            img = img.to(device)
            coc_label = coc_label.to(device)

            outputs_coc = model(img)
        
            loss = criterion(outputs_coc, coc_label)
            num_images = img.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            avg_acc = accuracy(outputs_coc.detach().cpu().numpy(),
                                            coc_label.detach().cpu().numpy())
            acc.update(avg_acc)
            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time,
                            loss=losses
                            , acc=acc)
                prefix = '{}_{}_{}.jpg'.format(os.path.join(output_dir, 'test'), epoch, i)
                save_batch_heatmaps(img, outputs_coc, prefix)

                logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )

            writer_dict['valid_global_steps'] = global_steps + 1

    return losses.avg, acc.avg

class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



