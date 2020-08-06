from collections import defaultdict

config = defaultdict(dict)

TRAIN = dict()
VALID = dict()

TRAIN['lr'] = 1e-4

TRAIN['batch_size'] = 16
VALID['batch_size'] = 1

TRAIN['n_epoch_init'] = 1000000
TRAIN['n_epoch'] = 200000

TRAIN['HR_img_path'] = 'DIV2K_train_HR/*.png'
TRAIN['LR_img_path'] = 'DIV2K_train_LR_bicubic/X4/*.png'

VALID['HR_img_path'] = 'DIV2K_valid_HR/*.png'
VALID['LR_img_path'] = 'DIV2K_valid_LR_bicubic/X4/*.png'
