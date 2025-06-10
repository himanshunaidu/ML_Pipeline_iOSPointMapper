import torch
import glob
import os
from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
from utilities.print_utils import *
from transforms.semantic_segmentation.data_transforms import MEAN, STD
from utilities.utils import model_parameters, compute_flops
from data_loader.semantic_segmentation.cityscapes import CITYSCAPE_TRAIN_CMAP

def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


def data_transform(img, im_size, mean, std):
    img = img.resize(im_size, Image.BILINEAR)
    img = F.to_tensor(img)  # convert to tensor (values between 0 and 1)
    img = F.normalize(img, mean, std)  # normalize the tensor
    return img


def evaluate(args, model, image_list, device):
    im_size = tuple(args.im_size)

    # get color map for pascal dataset
    if args.dataset == 'pascal':
        from utilities.color_map import VOCColormap
        cmap = VOCColormap().get_color_map_voc()
    elif args.dataset == 'city' or args.dataset == 'edge_mapping':
        cmap = CITYSCAPE_TRAIN_CMAP
        if args.dataset == 'edge_mapping' and args.is_custom:
            from data_loader.semantic_segmentation.edge_mapping import edge_mapping_to_custom_cocoStuff_dict
            cmap = {}
            for k, v in CITYSCAPE_TRAIN_CMAP.items():
                if k in edge_mapping_to_custom_cocoStuff_dict.keys():
                    cmap[edge_mapping_to_custom_cocoStuff_dict[k]] = v
                else:
                    cmap[k] = v
    else:
        cmap = None

    model.eval()
    for index, imgName in tqdm(enumerate(image_list)):
        img = Image.open(imgName).convert('RGB')
        w, h = img.size

        img = data_transform(img, im_size, args.mean, args.std)
        img = img.unsqueeze(0)  # add a batch dimension
        img = img.to(device)
        img_out = model(img)
        img_out = img_out.squeeze(0)  # remove the batch dimension
        img_out = img_out.max(0)[1].byte()  # get the label map
        img_out = img_out.to(device='cpu').numpy()

        if args.dataset == 'city':
            # cityscape uses different IDs for training and testing
            # so, change from Train IDs to actual IDs
            img_out = relabel(img_out)
        elif args.dataset == 'edge_mapping': # MARK: edge mapping dataset
            # edge mapping dataset uses different IDs for training and testing
            # so, change from Train IDs to actual IDs
            # Same as cityscape dataset
            img_out = relabel(img_out)

        img_out = Image.fromarray(img_out)
        # resize to original size
        img_out = img_out.resize((w, h), Image.NEAREST)

        # pascal dataset accepts colored segmentations
        # if args.dataset == 'pascal':
        img_out.putpalette(cmap)

        # save the segmentation mask
        name = imgName.split('/')[-1]
        img_extn = imgName.split('.')[-1]
        name = '{}/{}'.format(args.savedir, name.replace(img_extn, 'png'))
        img_out.save(name)


def main(args):
    # read all the images in the folder
    if args.dataset == 'city':
        # image_path = os.path.join(args.data_path, "leftImg8bit", args.split, "*", "*.png")
        image_path = os.path.join(args.data_path, "*.jpg")
        image_list = glob.glob(image_path)
        from data_loader.semantic_segmentation.cityscapes import CITYSCAPE_CLASS_LIST
        seg_classes = len(CITYSCAPE_CLASS_LIST)
    # elif args.dataset == 'edge_mapping': # MARK: edge mapping dataset
    #     image_path = os.path.join(args.data_path, "rgb", "*.png")
    #     image_list = glob.glob(image_path)
    #     from data_loader.semantic_segmentation.edge_mapping import EDGE_MAPPING_CLASS_LIST
    #     seg_classes = len(EDGE_MAPPING_CLASS_LIST)
    elif args.dataset == 'edge_mapping': # MARK: edge mapping dataset
        from data_loader.semantic_segmentation.edge_mapping import EdgeMappingSegmentation, EDGE_MAPPING_CLASS_LIST, edge_mapping_to_custom_cocoStuff_dict
        if args.is_custom and args.custom_mapping_dict is None:
            args.custom_mapping_dict = edge_mapping_to_custom_cocoStuff_dict
        image_path = os.path.join(args.data_path, "*.jpg")
        image_list = glob.glob(image_path)
        seg_classes = len(EDGE_MAPPING_CLASS_LIST)
        if args.is_custom: seg_classes = 53
    elif args.dataset == 'pascal':
        from data_loader.semantic_segmentation.voc import VOC_CLASS_LIST
        seg_classes = len(VOC_CLASS_LIST)
        data_file = os.path.join(args.data_path, 'VOC2012', 'list', '{}.txt'.format(args.split))
        if not os.path.isfile(data_file):
            print_error_message('{} file does not exist'.format(data_file))
        image_list = []
        with open(data_file, 'r') as lines:
            for line in lines:
                rgb_img_loc = '{}/{}/{}'.format(args.data_path, 'VOC2012', line.split()[0])
                if not os.path.isfile(rgb_img_loc):
                    print_error_message('{} image file does not exist'.format(rgb_img_loc))
                image_list.append(rgb_img_loc)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    if len(image_list) == 0:
        print_error_message('No files in directory: {}'.format(image_path))

    print_info_message('# of images for testing: {}'.format(len(image_list)))

    if args.model == 'espnetv2':
        from model.semantic_segmentation.espnetv2.espnetv2 import espnetv2_seg
        args.classes = seg_classes
        model = espnetv2_seg(args)
        args.mean = MEAN
        args.std = STD
    elif args.model == 'bisenetv2':
        from model.semantic_segmentation.bisenetv2.bisenetv2 import BiSeNetV2
        seg_classes = seg_classes - 1 if ((args.dataset == 'city' or args.dataset == 'edge_mapping') and not args.is_custom) else seg_classes # Because the background class is not used in the model
        args.classes = seg_classes
        model = BiSeNetV2(n_classes=args.classes, aux_mode='eval')
        # Eventually, the following mean and std related info should be moved to global config
        if args.dataset == 'city' or args.dataset == 'edge_mapping':
            args.mean = (0.3257, 0.3690, 0.3223)
            args.std = (0.2112, 0.2148, 0.2115)
        elif args.dataset == 'coco_stuff':
            args.mean = (0.46962251, 0.4464104,  0.40718787)
            args.std = (0.27469736, 0.27012361, 0.28515933)
        else:
            args.mean = MEAN
            args.std = STD
    else:
        print_error_message('{} network not yet supported'.format(args.model))
        exit(-1)

    # mdoel information
    num_params = model_parameters(model)
    flops = compute_flops(model, input=torch.Tensor(1, 3, args.im_size[0], args.im_size[1]))
    print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(args.im_size[0], args.im_size[1], flops))
    print_info_message('# of parameters: {}'.format(num_params))

    if args.weights_test:
        print_info_message('Loading model weights')
        weight_dict = torch.load(args.weights_test, map_location=torch.device('cpu'))
        model.load_state_dict(weight_dict, strict=False if args.model == 'bisenetv2' else True)
        print_info_message('Weight loaded successfully')
    else:
        print_error_message('weight file does not exist or not specified. Please check: {}', format(args.weights_test))

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    model = model.to(device=device)

    evaluate(args, model, image_list, device=device)


if __name__ == '__main__':
    from config.general_details import segmentation_models, segmentation_datasets

    parser = ArgumentParser()
    # mdoel details
    parser.add_argument('--model', default="espnetv2", choices=segmentation_models, help='Model name')
    parser.add_argument('--weights-test', default='', help='Pretrained weights directory.')
    parser.add_argument('--s', default=2.0, type=float, help='scale')
    # dataset details
    parser.add_argument('--data-path', default="", help='Data directory')
    parser.add_argument('--dataset', default='city', choices=segmentation_datasets, help='Dataset name')
    # input details
    parser.add_argument('--im-size', type=int, nargs="+", default=[512, 256], help='Image size for testing (W x H)')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='data split')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--is-custom', default=False, type=bool, help='Use custom mapping dictionary')
    parser.add_argument('--custom-mapping-dict', default=None, type=dict, help='Custom mapping dictionary')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')

    args = parser.parse_args()

    if not args.weights_test:
        from model.semantic_segmentation.espnetv2.weight_locations import model_weight_map

        model_key = '{}_{}'.format(args.model, args.s)
        dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size[0], args.im_size[1])
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
        args.weights_test = model_weight_map[model_key][dataset_key]['weights']
        if not os.path.isfile(args.weights_test):
            print_error_message('weight file does not exist: {}'.format(args.weights_test))

    # set-up results path
    if args.dataset == 'city':
        args.savedir = 'results/{}_{}_{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'edge_mapping': # MARK: edge mapping dataset
        args.savedir = 'results/{}_{}/{}'.format('results', args.dataset, args.split)
    elif args.dataset == 'pascal':
        args.savedir = 'results/{}_{}/VOC2012/Segmentation/comp6_{}_cls'.format('results', args.dataset, args.split)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    args.weights = ''

    main(args)
