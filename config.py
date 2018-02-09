import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', True):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', False):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser(description='')

#Image setting
parser.add_argument('--input_width', dest='input_width', default=128, help='input image width')
parser.add_argument('--input_height', dest='input_height', default=128, help='input image height')
parser.add_argument('--input_channel', dest='input_channel', default=3, help='input image channel')

parser.add_argument('--input_dim', dest='input_dim', default=100, help='input z size')

#Training Settings
parser.add_argument('--continue_training', dest='continue_training', default=False, type=str2bool, help='flag to continue training')

parser.add_argument('--data', dest='data', default='data', help='cats image train path')

parser.add_argument('--epochs', dest='epochs', default=25, help='total number of epochs')
parser.add_argument('--batch_size', dest='batch_size', default=64, help='batch size')

parser.add_argument('--learning_rate', dest='learning_rate', default=0.0001, help='learning rate of the optimizer')
parser.add_argument('--momentum', dest='momentum', default=0.5, help='momentum of the optimizer')
parser.add_argument('--alpha', dest='alpha', default=1.0, help='alpha')

#Measurement model setting
parser.add_argument('--patch_size', dest='patch_size', default=32, help='patch size')
parser.add_argument('--margin', dest='margin', default=5, help='margin')



#Extra folders setting
parser.add_argument('--checkpoints_path', dest='checkpoints_path', default='./checkpoints/', help='saved model checkpoint path')
parser.add_argument('--graph_path', dest='graph_path', default='./graphs/', help='tensorboard graph')
parser.add_argument('--images_path', dest='images_path', default='./images/', help='result images path')
args = parser.parse_args()