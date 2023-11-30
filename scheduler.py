
def adjustlr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
        
def adjustlr_exp(optimizer, decay_ratio, epoch, lr):
    lr_ = lr * decay_ratio ** epoch 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_