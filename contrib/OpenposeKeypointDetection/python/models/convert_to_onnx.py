import argparse
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
import onnx
import onnxsim
from torch.autograd import Variable


def convert_to_onnx(net, output_name):
    """
    Store weight of pytorch model to a .onnx weight file

    Args:
        net: pytorch model with weight loaded
        output_name: output name of the onnx weight file

    Returns:
        None

    """
    net_input = Variable(torch.randn(1, 3, 560, 560))
    input_names = ['data']
    output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                    'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']

    torch.onnx.export(net, net_input, output_name, verbose=True, opset_version=11, input_names=input_names, output_names=output_names)
    print("====> check onnx model...")

    model = onnx.load(output_name)
    onnx.checker.check_model(model)

    print("====> Simplifying...")
    model_opt, check = onnxsim.simplify(output_name)
    onnx.save(model_opt, output_name)
    print("onnx model simplify Ok!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--output-name', type=str, default='human-pose-estimation.onnx',
                        help='name of output model in ONNX format')
    args = parser.parse_args()
    net_trained = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    load_state(net_trained, checkpoint)
    convert_to_onnx(net_trained, args.output_name)
