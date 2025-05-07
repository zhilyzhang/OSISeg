import sys
import torch


def get_network(args, net, device, distribution = True):
    """ return given network
    """

    if net == 'sam':
        from models.sam import SamPredictor, sam_model_registry
        from models.sam.utils.transforms import ResizeLongestSide
        net = sam_model_registry['vit_b'](args, checkpoint=args.sam_ckpt).to(device)

    # elif net == 'mobile':
    #     from mobile_sam import SamPredictor, sam_model_registry
    #     from mobile_sam.utils.transforms import ResizeLongestSide
    #
    #     net = sam_model_registry['vit_t'](checkpoint=args.sam_ckpt).to(device)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if distribution != 'none':
        net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
        net = net.to(device)
    else:
        net = net.to(device)

    return net
