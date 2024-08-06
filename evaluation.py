import torch
import argparse
from PIL import Image
from nerf.provider import NeRFDataset
from nerf.network import NeRFNetwork
from nerf.utils import *
from torchvision import models as model_load
import torchvision.transforms as transforms
from torchvision.transforms import Resize


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(self.device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(self.device)
        

    def forward(self, x):
        return (x - self.mean) / self.std
    
def get_evaluation_model(model_name):
    resnet101 = model_load.resnet101(pretrained=True)
    densenet121 = model_load.densenet121(pretrained=True)
    networks = {
            'resnet': resnet101,
            'densenet': densenet121,
        }
    resize_transform = transforms.Resize((224, 224))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = NormalizeByChannelMeanStd(mean, std)
    return nn.Sequential(resize_transform, normalize, networks[model_name].eval()).to(device)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    if 'model' not in checkpoint_dict:
        model.load_state_dict(checkpoint_dict)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_dict['model'], strict=False)
    
    if model.cuda_ray and 'mean_density' in checkpoint_dict:
        model.mean_density = checkpoint_dict['mean_density']
    model.eval()

def extract_target_label(workspace_path, object_name):
    log_file_pattern = f"logs/adv_optimization/{object_name}_*.log"
    # Find the first log file matching the pattern
    log_file = glob.glob(log_file_pattern)
    target_label = int(log_file[0].split(":")[0].split('_')[-1])    
    return target_label

def setup_model(opt, device):
    model = NeRFNetwork(opt)
    model.to(device)
    checkpoint = opt.workspace + "checkpoints/ngp_stage1_ep0400.pth"
    load_checkpoint(model, checkpoint, device)
    return model

def main(opt):
    device = 'cuda' if opt.cuda_ray else 'cpu'
    opt.refine_steps = [int(round(x * opt.iters)) for x in opt.refine_steps_ratio]
    loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()
    model = setup_model(opt, device)
    
    object_name = opt.path.split(r'/')[-2]
    target_label = extract_target_label(opt.workspace, object_name)

    cnt, num = 0, 0
    net = get_evaluation_model(opt.evaluation_model)
    print(target_label)
    
    for data in loader:
        num += process_data_batch(data, model, net, target_label)
        cnt += 1
    
    print(f"The ASR is: {num/cnt}")

def process_data_batch(data, model, net, target_label):
    rays_o, rays_d = data['rays_o'], data['rays_d']
    index = data['index']
    cam_near_far = data.get('cam_near_far')
    
    images = data['images']
    N, C = images.shape
    bg_color = 1
    shading = 'full'
    mvp = data['mvp'].squeeze(0)
    H, W = data['H'], data['W']
    
    outputs = model.render_stage1(rays_o, rays_d, mvp, H, W, index=index, bg_color=bg_color, shading=shading, **vars(opt))
    pred_rgb = outputs['image']
    
    return evaluate_prediction(pred_rgb, H, W, net, target_label)

def evaluate_prediction(pred_rgb, H, W, net, target_label):
    resize_transform = Resize((224, 224))
    pred_image = pred_rgb.reshape(H, W, 3)
    image_predicted = pred_image.permute(2, 0, 1).unsqueeze(0)
    output = net(image_predicted)
    pred = nn.functional.softmax(output, dim=1).topk(1)
    print(pred.indices.item())

    return pred.indices.item() == target_label
        

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="recommended settings")
    parser.add_argument('--workspace', type=str, default='/mnt/data1/huangyao/trial_airliner_01')
    parser.add_argument('--evaluation_model', type=str, default='resnet', help="Type of surrogate model to use, default is 'resnet'.")
    parser.add_argument('--stage', type=int, default=0, help="training stage")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--sdf', action='store_true', help="use sdf instead of density for nerf")
    parser.add_argument('--tcnn', action='store_true', help="use tcnn's gridencoder")
    parser.add_argument('--progressive_level', action='store_true', help="progressively increase max_level")

    ### dataset options
    parser.add_argument('--data_format', type=str, default='nerf', choices=['nerf', 'colmap', 'dtu'])
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'all'])
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--random_image_batch', action='store_true', help="randomly sample rays from all images per step in training stage 0, incompatible with enable_sparse_depth")
    parser.add_argument('--downscale', type=int, default=2, help="downscale training images")
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=-1, help="scale camera location into box[-bound, bound]^3, -1 means automatically determine based on camera poses..")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--mesh', type=str, default='', help="template mesh for phase 2")
    parser.add_argument('--enable_cam_near_far', action='store_true', help="colmap mode: use the sparse points to estimate camera near far per view.")
    parser.add_argument('--enable_cam_center', action='store_true', help="use camera center instead of sparse point center (colmap dataset only)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--enable_sparse_depth', action='store_true', help="use sparse depth from colmap pts3d, only valid if using --data_formt colmap")
    parser.add_argument('--enable_dense_depth', action='store_true', help="use dense depth from omnidepth calibrated to colmap pts3d, only valid if using --data_formt colmap")

    ### training options
    parser.add_argument('--iters', type=int, default=25000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--lr_vert', type=float, default=1e-4, help="initial learning rate for vert optimization")
    parser.add_argument('--pos_gradient_boost', type=float, default=1, help="nvdiffrast option")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--grid_size', type=int, default=128, help="density grid resolution")
    parser.add_argument('--mark_untrained', action='store_true', help="mark_untrained grid")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--diffuse_step', type=int, default=1000, help="training iters that only trains diffuse color for better initialization")
    parser.add_argument('--diffuse_only', action='store_true', help="only train diffuse color by overriding --diffuse_step")
    parser.add_argument('--background', type=str, default='white', choices=['white', 'random'], help="training background mode")
    parser.add_argument('--enable_offset_nerf_grad', action='store_true', help="allow grad to pass through nerf to train vertices offsets in stage 1, only work for small meshes (e.g., synthetic dataset)")
    parser.add_argument('--n_eval', type=int, default=5, help="eval $ times during training")
    parser.add_argument('--n_ckpt', type=int, default=50, help="save $ times during training")

    # batch size related
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--adaptive_num_rays', action='store_true', help="adaptive num rays for more efficient training")
    parser.add_argument('--num_points', type=int, default=2 ** 18, help="target num points for each training step, only work with adaptive num_rays")

    # stage 1 regularizations
    parser.add_argument('--wo_smooth', action='store_true', help="disable all smoothness regularizations")
    parser.add_argument('--lambda_lpips', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_offsets', type=float, default=0.1, help="loss scale")
    parser.add_argument('--lambda_lap', type=float, default=0.001, help="loss scale")
    parser.add_argument('--lambda_normal', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_edgelen', type=float, default=0, help="loss scale")

    # unused
    parser.add_argument('--contract', action='store_true', help="apply L-INF ray contraction as in mip-nerf, only work for bound > 1, will override bound to 2.")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")
    parser.add_argument('--trainable_density_grid', action='store_true', help="update density_grid through loss functions, instead of directly update.")
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--ind_dim', type=int, default=0, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=500, help="number of individual codes, should be larger than training dataset size")

    # stage 1
    parser.add_argument('--ssaa', type=int, default=2, help="super sampling anti-aliasing ratio")
    parser.add_argument('--texture_size', type=int, default=1024, help="exported texture resolution")
    parser.add_argument('--refine', action='store_true', help="track face error and do subdivision")
    parser.add_argument("--refine_steps_ratio", type=float, action="append", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
    parser.add_argument('--refine_size', type=float, default=0.01, help="refine trig length")
    parser.add_argument('--refine_decimate_ratio', type=float, default=0.1, help="refine decimate ratio")
    parser.add_argument('--refine_remesh_size', type=float, default=0.02, help="remesh trig length")

    ### GUI options
    parser.add_argument('--vis_pose', action='store_true', help="visualize the poses")
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1000, help="GUI width")
    parser.add_argument('--H', type=int, default=1000, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    opt = parser.parse_args()
    opt.cuda_ray = True
    main(opt)