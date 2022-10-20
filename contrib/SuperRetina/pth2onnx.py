import torch
import torch.onnx
from model.super_retina import SuperRetina

def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    checkpoint = torch.load(checkpoint, map_location='cpu')
    model = SuperRetina()
    model.load_state_dict(checkpoint['net'], strict=True)
    model.eval()

    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names, opset_version=11) #指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")
    return 0

if __name__ == '__main__':
    checkpoint_path = './SuperRetina.pth'
    onnx = './SuperRetina.onnx'
    inputs = torch.randn(2, 1, 768, 768)
    pth_to_onnx(inputs, checkpoint_path, onnx)

