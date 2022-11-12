import argparse
import torch
from lib import VideoModel_pvtv2 as Network


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str, default='MoCA')
parser.add_argument('--testsplit',  type=str, default='TestDataset_per_sq')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='testing size')
parser.add_argument('--pretrained_cod10k', default=None, help='path to the pretrained Resnet')

parser.add_argument('--pth_path', type=str, \
    default='/Users/mac/Downloads/snapshot/Net_epoch_MoCA_short_term_pseudo.pth')
parser.add_argument('--onnx_path', type=str, \
    default='/Users/mac/Downloads/sltnet.onnx')

opt = parser.parse_args()


if __name__ == '__main__':
    model = Network(opt)

    model.load_state_dict(torch.load(opt.pth_path, map_location=torch.device('cpu')))
    model.eval()

    input_names = ["image"]  
    output_names = ["pred"]  
    dynamic_axes = {'image': {0: '-1'}, 'pred': {0: '-1'}} 
    dummy_input = torch.randn(1, 9, 352, 352)
    torch.onnx.export(model, dummy_input, opt.onnx_path, input_names=input_names, dynamic_axes=dynamic_axes, output_names=output_names, opset_version=11, verbose=True) 
