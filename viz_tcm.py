import os
import cv2
import utils
import torch
import imageio
import argparse
import numpy as np
from tcm import compute_TCM
from rich.progress import track
import pdb


def main(args):
    ## prepare dirs    
    frame_dir = os.path.join(args.input_dir, args.approach)
    flow_dir = os.path.join(args.meta_dir, args.flow_dir)
    mask_dir = os.path.join(args.meta_dir, args.occu_dir)
    
    image_names = [x for x in os.listdir(frame_dir) if utils.is_image_file(x) and args.sub_name in x]
    image_names = sorted(list(set([x.split('-')[0] for x in image_names if args.type in x])))
    # print(len(image_names))
    # pdb.set_trace()
    
    os.makedirs(args.result_dir, exist_ok=True)
    out_name = args.approach + '-' + args.sub_name + '-' + args.type

    out_base_dir = os.path.join(args.result_dir, args.approach, args.type)
    os.makedirs(out_base_dir, exist_ok=True)
    
    ## flow warping layer
    image_size = (436, 1024)
    device = torch.device(args.device)
    wrap_op = utils.Warper2d(image_size).to(device) 
    
    ### start evaluation
    TCM_lst = []
    
    writer = imageio.get_writer(os.path.join(out_base_dir, out_name + '.mp4'), fps=10)
    
    for i in track(range(len(image_names) - 1), description='Processing'):
        img1 = utils.load_img_and_resize(os.path.join(frame_dir, image_names[i] + '-pred.png'))
        img2 = utils.load_img_and_resize(os.path.join(frame_dir, image_names[i+1] + '-pred.png'))
        
        base_name = image_names[i].split('_' + args.type)[0].split(args.sub_name+'_')[-1]
        flow_wpf = os.path.join(flow_dir, args.sub_name +'_'+ base_name + '.h5')
        # print(flow_wpf)
        flow = utils.h5_reader(flow_wpf)
        mask_wpf = os.path.join(mask_dir, args.sub_name, base_name + '.png')
        mask = utils.load_img_and_resize(mask_wpf)
        
        # deal with boundary effect, boundary erosion
        erosion_w = 20
        mask[:erosion_w,:,:] = 1
        mask[-erosion_w:,:,:] = 1
        mask[:,:erosion_w,:] = 1
        mask[:,-erosion_w:,:] = 1
        
        img1_ref = utils.load_img_and_resize(os.path.join(frame_dir, image_names[i] + '-real.png'))
        img2_ref = utils.load_img_and_resize(os.path.join(frame_dir, image_names[i+1] + '-real.png'))
        
        tcm, tcm_map, tcm_map_opt, err, err_ref = compute_TCM(img1, img1_ref, img2, img2_ref, flow, mask, wrap_op, device, False)
        TCM_lst.append(tcm)
        
        err_ave = np.mean(err, axis=-1)
        err_ref_ave = np.mean(err_ref, axis=-1)
        
        error_show, error_htm = utils.overlay_results(img1.copy(), err_ave / err_ave.max())
        tcm_map_opt_show_inv, tcm_map_opt_htm_inv = utils.overlay_results(img1.copy(), 1 - (tcm_map_opt / tcm_map_opt.max()))
        # tcm_map_opt_show, tcm_map_opt_htm = utils.overlay_results(img1.copy(), tcm_map_opt / tcm_map_opt.max())
        tcm_map_opt_show, tcm_map_opt_htm = utils.overlay_results(img1.copy(), tcm_map_opt)
        # for comparison:
        error_ref_show, error_ref_htm = utils.overlay_results(img1_ref.copy(), err_ref_ave)
        
        flow_show = utils.flow_to_color(flow)[..., ::-1]
        
        writer.append_data(cv2.hconcat([cv2.vconcat([error_show, error_htm, tcm_map_opt_show_inv, tcm_map_opt_htm_inv]),
                                        # cv2.vconcat([tcm_map_opt_show, tcm_map_opt_htm, error_cmp_htm, flow_show])
                                        cv2.vconcat([tcm_map_opt_show, tcm_map_opt_htm, np.uint8(img1*255), flow_show])]))
        
    writer.close()

    np.save(os.path.join(out_base_dir, out_name + '.npy'), np.array(TCM_lst))
    print('Mean TCM: ', np.mean(np.array(TCM_lst)))
    print(f'Results are saved at: {os.path.join(out_base_dir, out_name)}.mp4')
    with open(os.path.join(out_base_dir, 'TCM.txt'), 'a+') as f:
        f.write('{}| {}| {}: {:.6f}\n'.format(
            args.approach, args.type, args.sub_name, np.mean(np.array(TCM_lst))
        ))
        

def main_single_frame(args):
    ## prepare dirs    
    frame_dir = os.path.join(args.input_dir, args.approach)
    flow_dir = os.path.join(args.meta_dir, args.flow_dir)
    mask_dir = os.path.join(args.meta_dir, args.occu_dir)
    frame_id = args.number  # 1~49
    
    image_names = [x for x in os.listdir(frame_dir) if utils.is_image_file(x) and args.sub_name in x]
    image_names = sorted(list(set([x.split('-')[0] for x in image_names if args.type in x])))
    # print(len(image_names))
    # pdb.set_trace()
    
    os.makedirs(args.result_dir, exist_ok=True)
    out_name = args.approach + '-' + args.sub_name + '-' + args.type

    out_base_dir = os.path.join(args.result_dir, args.approach, args.type)
    os.makedirs(out_base_dir, exist_ok=True)
    
    ## flow warping layer
    image_size = (436, 1024)
    device = torch.device(args.device)
    wrap_op = utils.Warper2d(image_size).to(device) 
    
    ### start evaluation
    # TCM_lst = []
    
    # writer = imageio.get_writer(os.path.join(out_base_dir, out_name + '.mp4'), fps=10)
    
    i = frame_id - 1
    img1 = utils.load_img_and_resize(os.path.join(frame_dir, image_names[i] + '-pred.png'))
    img2 = utils.load_img_and_resize(os.path.join(frame_dir, image_names[i+1] + '-pred.png'))
    
    base_name = image_names[i].split('_' + args.type)[0].split(args.sub_name+'_')[-1]
    flow_wpf = os.path.join(flow_dir, args.sub_name +'_'+ base_name + '.h5')
    # print(flow_wpf)
    flow = utils.h5_reader(flow_wpf)
    mask_wpf = os.path.join(mask_dir, args.sub_name, base_name + '.png')
    mask = utils.load_img_and_resize(mask_wpf)
    
    # deal with boundary effect, boundary erosion
    erosion_w = 20
    mask[:erosion_w,:,:] = 1
    mask[-erosion_w:,:,:] = 1
    mask[:,:erosion_w,:] = 1
    mask[:,-erosion_w:,:] = 1
    
    img1_ref = utils.load_img_and_resize(os.path.join(frame_dir, image_names[i] + '-real.png'))
    img2_ref = utils.load_img_and_resize(os.path.join(frame_dir, image_names[i+1] + '-real.png'))
    
    tcm, tcm_map, tcm_map_opt, err, err_ref = compute_TCM(img1, img1_ref, img2, img2_ref, flow, mask, wrap_op, device, False)
    # TCM_lst.append(tcm)
    
    err_ave = np.mean(err, axis=-1)
    err_ref_ave = np.mean(err_ref, axis=-1)
    
    error_show, error_htm = utils.overlay_results(img1.copy(), err_ave / err_ave.max())
    tcm_map_opt_show_inv, tcm_map_opt_htm_inv = utils.overlay_results(img1.copy(), 1 - (tcm_map_opt / tcm_map_opt.max()))
    # tcm_map_opt_show, tcm_map_opt_htm = utils.overlay_results(img1.copy(), tcm_map_opt / tcm_map_opt.max())
    tcm_map_opt_show, tcm_map_opt_htm = utils.overlay_results(img1.copy(), tcm_map_opt)
    # for comparison:
    error_ref_show, error_ref_htm = utils.overlay_results(img1_ref.copy(), err_ref_ave)
    
    flow_show = utils.flow_to_color(flow)[..., ::-1]
    
    cv2.imwrite(os.path.join(out_base_dir, out_name+'-pred-'+str(frame_id)+'.png'), np.uint8(img1*255)[..., ::-1]) # (filename, img)
    cv2.imwrite(os.path.join(out_base_dir, out_name+'-gt-'+str(frame_id)+'.png'), np.uint8(img1_ref*255)[..., ::-1])
    cv2.imwrite(os.path.join(out_base_dir, out_name+'-flow-'+str(frame_id)+'.png'), flow_show)
    cv2.imwrite(os.path.join(out_base_dir, out_name+'-tcmMapInv-'+str(frame_id)+'.png'), tcm_map_opt_show_inv[..., ::-1])
    cv2.imwrite(os.path.join(out_base_dir, out_name+'-tcmHtmInv-'+str(frame_id)+'.png'), tcm_map_opt_htm_inv[..., ::-1])
    
    print('saving imgs in {}'.format(os.path.join(out_base_dir, out_name+'*.png')))
    # writer.append_data(cv2.hconcat([cv2.vconcat([error_show, error_htm, tcm_map_opt_show_inv, tcm_map_opt_htm_inv]),
    #                                 # cv2.vconcat([tcm_map_opt_show, tcm_map_opt_htm, error_cmp_htm, flow_show])
    #                                 cv2.vconcat([tcm_map_opt_show, tcm_map_opt_htm, np.uint8(img1*255), flow_show])]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TCM Visualizer')
    parser.add_argument('-i', '--input_dir',  type=str, help='input dir',
                        default='/home/wzj/ws/intrinsic/compare_results/')
    parser.add_argument('-m', '--meta_dir',  type=str, help='meta dir, contains flows, occlusions',
                        default='/home/wzj/ws/intrinsic/data/MPI/')
    parser.add_argument('-f', '--flow_dir',   type=str, help='flow dir',
                        default='flow_hdf5')
    parser.add_argument('-o', '--occu_dir',   type=str, help='mask occlusion dir',
                        default='occlusions')
    parser.add_argument('-s', '--sub_name',   type=str, help='sub name', default='cave_4')   # bandage_2
    parser.add_argument('-r', '--result_dir', type=str, help='output result dir',
                        default='test_results_wzj')
    parser.add_argument('-t', '--type',     choices=['reflect', 'shading'], default='reflect')
    parser.add_argument('-a', '--approach', choices=['DI_flow', 'DI_framewise', 'ours_DVP', 'ours_flow_r10_s5', 'ours_framewise_ep195', 'revisiting'],
                        default='DI_flow')
    parser.add_argument('-n', '--number', type=int, help='set specific frame number', default=None)
    parser.add_argument('--device', default='cuda:0', help='device name')
    
    args = parser.parse_args()
    
    if args.number is None:
        main(args)
    else:
        main_single_frame(args)
    

# RUN commands:
#
# python viz_tcm.py -s cave_4 -t shading -a DI_flow
#
# You can get some results like below:

# Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
# Mean TCM:  0.5582780025356711
# Results are saved at: test_results/DI_flow-cave_4-shading* .
    