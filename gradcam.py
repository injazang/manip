import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from models import SRNet, ZhuNet
import glob, os
from image import run
def mix(cover, stego, prob):
    cover = np.reshape(cover[...,0], newshape=[1,256,256])
    stego = np.reshape(stego[...,0], newshape=[1,256,256])

    cover_ = torch.from_numpy(cover).float().unsqueeze(0).requires_grad_(True)
    stego_ = torch.from_numpy(stego).float().unsqueeze(0).requires_grad_(True)
    size = int(np.sqrt(256*256 * prob))
    rand_sizes = torch.rand(2)
    rx = (rand_sizes[0] * (256 - size)).int()
    ry = (rand_sizes[1] * (256 - size)).int()
    mask = np.zeros([1,256, 256], dtype='float32')
    mask[:,rx:rx + size, ry:ry + size] = 1

    total_bits_changed = np.sum(np.abs(cover - stego))
    part_bits_changed = np.sum(np.abs(cover * mask - stego * mask))
    bits_changeds_portion = part_bits_changed / total_bits_changed

    manip_image = cover * mask + stego * (1 - mask)
    manip_image_ = cover * (1 - mask) + stego * mask
    manip_image = torch.from_numpy(manip_image).unsqueeze(0).requires_grad_(True)
    manip_image_ = torch.from_numpy(manip_image_).unsqueeze(0).requires_grad_(True)

    return [cover_, stego_, manip_image_, manip_image],np.abs(cover - stego),  np.array([1 - prob, prob]), mask
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            elif 'fc' in name.lower():
                x = x.view(x.size(0), -1)
                x = module(x)
            else:
                x = module(x)

        return target_activations, torch.cat([1-x, x], dim=1)




def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img/255)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), np.uint8(heatmap / np.max(heatmap) * 255)


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, output


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)
        output = torch.cat([1 - output, output], dim=1)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


""" python grad_cam.py <path_to_image>
1. Loads an image with opencv.
3. Makes a forward pass to find the category index with the highest score,
and computes intermediate activations.
Makes the visualization. """


# Can work with any model, but it assumes that the model has a
# feature method, and a classifier method,
# as in the VGG models in torchvision.

# If None, returns the map for the highest scoring category.
# Otherwise, targets the requested index.
imgdirs =['6105_boss.pgm']#, '', ]#'']
#[,
#'437_boss.pgm' '3238_boss.pgm''6105_boss.pgm'5205_boss.pgm7410_boss.pgm
target_index = None
#dirs = ['bit_0.2', 'un_0.2'] #'srnet_bitmix', 'srnet_unmix']
dirs = ['zhu_mixup','zhunet_mix_s-uniward_0.4_0.5_0.25_20-05-26_13-41', 'zhu_bitmix' ]#'srnet_unmix','mixup', 'srnet_cutmix', 'srnet_mix_s-uniward_0.4_0.5_0.25_20-05-23_16-29']
outputs=[]
for im in imgdirs:
    run(im)
    cover = cv2.imread('./im/cover.png')
    stego = cv2.imread('./im/s-uniward_0.4.png')
    dif = cv2.imread('./im/s-uniward_0.4_diff.png')

    imgs, diff, _, mask_ = mix(cover, stego, 0.5)
    for dir in dirs:
        for i in range(2):
            for j in range(2):
                graddir= f'trained/{dir}/grad/{im}'

                os.makedirs(graddir, exist_ok=True)
                cv2.imwrite(f'{graddir}/mask.png', np.uint8(mask_[0, ...] * 255))
                cv2.imwrite(f'{graddir}/stego.png', stego)
                cv2.imwrite(f'{graddir}/dif.png', dif)


                os.makedirs(graddir, exist_ok=True)
                model = ZhuNet.ZhuNet()
                last_checkpoint_path = glob.glob(os.path.join(f'trained/{dir}', '*.tar'))[-1]
                checkpoint = torch.load(last_checkpoint_path,map_location='cuda:0')
                model.load_state_dict(checkpoint['model_state_dict'])
                grad_cam = GradCam(model=model, feature_module=model.blocks, target_layer_names=['0'], use_cuda=True)

                mask, output = grad_cam(imgs[i]/255, index=j)
                outputs.append(output)
                cammed, heatmap = show_cam_on_image(cover, mask)
                cv2.imwrite(f'{graddir}/cam_{i}_{j}.png', cammed)
                cv2.imwrite(f'{graddir}/heat_{i}_{j}.png', heatmap)
                ''''''
                gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
                print(model._modules.items())
                gb = gb_model(imgs[i], index=i)
                gb = gb.transpose((1, 2, 0))
                cam_mask = cv2.merge([mask, mask, mask])
                cam_gb = deprocess_image(cam_mask * gb)
                gb = deprocess_image(gb)

                cv2.imwrite(f'{graddir}/gb_{i}.jpg', gb)
                cv2.imwrite(f'{graddir}/cam_gb_{i}.jpg', cam_gb)

    [print(out) for out in outputs]