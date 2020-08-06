import glob
import os

from conf import *
from data import *
from model import generator, discriminator
from train import GeneratorTrainer, SrganTrainer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, 'dataset')

"""
DATA
"""
train_hr_img_path = glob.glob(os.path.join(DATASET, TRAIN['HR_img_path']))
train_lr_img_path = glob.glob(os.path.join(DATASET, TRAIN['LR_img_path']))
valid_hr_img_path = glob.glob(os.path.join(DATASET, VALID['HR_img_path']))
valid_lr_img_path = glob.glob(os.path.join(DATASET, VALID['LR_img_path']))

train_data_loader = make_dataset(train_lr_img_path, train_hr_img_path)
train_dataset = train_data_loader.dataset(TRAIN['batch_size'], random_transform=True, repeat_count=None)

valid_data_loader = make_dataset(valid_lr_img_path, valid_hr_img_path)
valid_dataset = train_data_loader.dataset(VALID['batch_size'], random_transform=False, repeat_count=1)


"""
TRAINS
"""
# training context for the generator
pre_trainer = GeneratorTrainer(model=generator(), checkpoint_dir='./ckpt/pre', learning_rate=TRAIN['lr'])

pre_trainer.train(train_dataset, valid_dataset.take(10), steps=TRAIN['n_epoch_init'], eval_every=1000)

pre_trainer.model.save_weights('weights/generator.h5')


# init with pre-trained weights
gan_generator = generator()
gan_generator.load_weights('weights/generator.h5')

gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())

gan_trainer.train(train_dataset, steps=TRAIN['n_epoch'])

gan_trainer.generator.save_weights('weights/gan_generator.h5')
gan_trainer.discriminator.save_weights('weights/gan_discriminator.h5')