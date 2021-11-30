import argparse
import yaml
import torch
import os

from backbone import EfficientDetBackbone
import onnx
import onnxsim
from torch.autograd import Variable
from utils.load_state import load_state


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    '''
    Get command line parameters.

    Returns: None

    '''
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training '
                             'will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load '
                             'last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--output-name', type=str, default='efficient-det-d0-mindxsdk-order.onnx',
                        help='name of output model in ONNX format')

    args = parser.parse_args()
    return args


def convert_to_onnx(net, output_name, input_size):
    f'''
    Convert pytorch model to onnx model.

    Args:
        net: pytorch model object
        output_name: final output onnx model name is simplified-{output_name}
        input_size: input image size of the pytorch model

    Returns: None

    '''
    input = Variable(torch.randn(1, 3, input_size, input_size))
    input_names = ['input']
    output_names = ['p3_out', 'p4_out', 'p5_out', 'p6_out', 'p7_out', 'regression', 'classification', 'anchors']
    output_path = 'onnx_models/' + output_name
    torch.onnx.export(net, input, output_path, verbose=True, opset_version=11, input_names=input_names,
                      output_names=output_names)
    
    print("====> check onnx model...")
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)

    print("====> Simplifying...")
    model_opt, check = onnxsim.simplify(output_path)
    # print("model_opt", model_opt)
    simplified_output_path = os.path.join('onnx_models', 'simplified-' + output_name)
    onnx.save(model_opt, simplified_output_path)
    print("onnx model simplify Ok!")


if __name__ == '__main__':
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    opt = get_args()
    params = Params(f'projects/{opt.project}.yml')
    print('compound_coef: ', opt.compound_coef)
    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales),
                                 onnx_export=True)
    checkpoint = torch.load(opt.load_weights, map_location=torch.device('cpu'))
    model.load_state_dict(torch.load(opt.load_weights), strict=False)
    convert_to_onnx(model, opt.output_name, input_sizes[opt.compound_coef])
