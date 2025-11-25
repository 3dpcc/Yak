import numpy as np
import torch
import os, glob, argparse, time
import pandas as pd
from tqdm import tqdm
import math
import MinkowskiEngine as ME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from data_processing.data_utils import load_sparse_tensor, run_cmd, yuv2rgb, rgb2yuv, knn_interpolation
from extension.pc_error import pc_error
from extension.gpcc import gpcc_encode, gpcc_decode, get_points_number

from models.guided_filter import GuidedFilter

def test(filedir, bin_dir, rec_dir, transformType=0, qp=51):
    results_enc = gpcc_encode(filedir, bin_dir, transformType=transformType, qp=qp, posQuantscale=1)
    results_dec = gpcc_decode(bin_dir, rec_dir)

    # record results
    results = {'filename': os.path.split(filedir)[-1].split('.')[0], 'qp': qp}
    for k, v in results_enc.items(): results['Enc_' + k] = v
    for k, v in results_dec.items(): results['Dec_' + k] = v
    num_points = get_points_number(filedir)
    bpp_geo = results_enc['positions bitstream size'] * 8 / num_points
    bpp_att = results_enc['colors bitstream size'] * 8 / num_points
    results['num_points'] = num_points
    results['bpp_geo'] = bpp_geo
    results['bpp_att'] = bpp_att

    return results


def test_geometry_coding(model_list, qs_list, input_rootdir, output_rootdir):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+'ply'), recursive=True))
    for idx_file, input_filedir in enumerate(tqdm(input_filedirs)):
        point_cloud_name = input_filedir[len(input_rootdir):].split('.')[0]
        out_folder = './' + output_rootdir + point_cloud_name
        os.makedirs(out_folder, exist_ok=True)

        print('Testing ', point_cloud_name)
        
        # load data
        x = load_sparse_tensor(input_filedir, order='rgb')

        # preprocess
        downsampler = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        true_down = downsampler(downsampler(x))

        from data_processing.data_utils import write_ply_ascii
        down_dir = out_folder + '/' + point_cloud_name + '_down.ply'
        bin_dir = out_folder + '/' + point_cloud_name + '_down.bin'
        rec_dir = out_folder + '/' + point_cloud_name + '_down_rec.ply'

        write_ply_ascii(down_dir, true_down.C.cpu().numpy()[:, 1:] // 4,
                        (true_down.F.detach().cpu().numpy() * 255).round())
        

        for idx_rate, lambda_G in enumerate(tqdm(qs_list)):
            out_subfolder = out_folder + '/r' + str(idx_rate) + '_' + 'lambda_' + str(lambda_G)
            os.makedirs(out_subfolder, exist_ok=True)
            filename = out_subfolder + '/' + point_cloud_name
            print('-------------------------------------------------------------------------------')
            print(f'Start compressing and decompressing {point_cloud_name}, lambda: {lambda_G}')

            results_one = {'lambda_G': lambda_G}
            
            if lambda_G <= 512.:
                model = model_list[0]
            else:
                lambda_G = lambda_G // 8
                model = model_list[1]

            print(lambda_G)

            # base layer - GPCC
            results_single = test(down_dir, bin_dir, rec_dir, transformType=0)
            gpcc_bytes = results_single['Enc_positions bitstream size']

            # enhancement layer - Neural Network
            
            # encode
            y_Bytes, z_Bytes, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, nums_list = model.encode(x, lambda_G, filename)

            # decode
            out = model.decode(lambda_G, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, nums_list, down_dir,
                                filename, device)

            # save
            coords = out.C.detach().cpu().numpy()[:,1:]

            dec_dir = filename + '_dec.ply'

            from data_processing.data_utils import write_ply_ascii_geo
            write_ply_ascii_geo(dec_dir, coords)

            print('Start evaluating performance')
            # rate
            Bytes = y_Bytes + z_Bytes
            bpp_geo_base = round(gpcc_bytes * 8 / float(x.__len__()), 4)
            bpp_geo_enhance = round(Bytes * 8 / float(x.__len__()), 4)

            results_one['Total_bpp_geo'] = bpp_geo_base + bpp_geo_enhance
            results_one['bpp_geo_base'] = bpp_geo_base
            results_one['bpp_geo_enhance'] = bpp_geo_enhance

            # PSNR
            pc_error_results = pc_error(input_filedir, dec_dir, res=1023, color=False, show=False)
            for k, v in pc_error_results.items(): 
                if k == 'mseF,PSNR (p2point)' or k =='mseF,PSNR (p2plane)':
                    results_one[k] = v

            # PCQM
            cmd = './PCQM' + ' ' + input_filedir + ' ' + dec_dir + ' -r 0.004 -knn 20 -rx 2.0'
            std_out_and_err = run_cmd([cmd])
            std_out = str(std_out_and_err[0])

            pcqm = std_out.split('PCQM value is :')[1].split('\\n')[0]
            geo = std_out.split('Geo value is :')[1].split('\\n')[0]
            attr = std_out.split('Attr value is :')[1].split('\\n')[0]

            results_one['PCQM'] = pcqm
            results_one['Geo'] = geo
            results_one['Attr'] = attr
            results_one['1 - PCQM'] = 1 - float(pcqm)

            if idx_rate==0: results_list = [results_one]
            else: results_list.append(results_one)

            print(results_one)

        # merge all rate
        results = {'filename':point_cloud_name}
        one_keys = []
        multi_keys = ['lambda_G',
                      'Total_bpp_geo', 'bpp_geo_base', 'bpp_geo_enhance', 
                      'mseF,PSNR (p2point)', 'mseF,PSNR (p2plane)',
                      'PCQM', 'Geo', 'Attr', '1 - PCQM']
        for idx_rate, results_one in enumerate(results_list):
            for k, v in results_one.items():
                if k in one_keys and idx_rate==0: results[k] = v
                if k in multi_keys: results['R'+str(idx_rate)+'_'+k] = v
        results = pd.DataFrame([results])
        # merge all file
        if idx_file==0: results_allfile = results.copy(deep=True)
        else: results_allfile = pd.concat([results_allfile, results], ignore_index=True)
        csvfile = os.path.join(output_rootdir, output_rootdir.split('/')[-2]+'.csv')
        results_allfile.to_csv(csvfile, index=False)
        print('save results to ', csvfile)

    return results_allfile

def test_attribute_coding(model, qs_list, input_rootdir, output_rootdir):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+'ply'), recursive=True))
    for idx_file, input_filedir in enumerate(tqdm(input_filedirs)):
        point_cloud_name = input_filedir[len(input_rootdir):].split('.')[0]
        out_folder = './' + output_rootdir + point_cloud_name
        os.makedirs(out_folder, exist_ok=True)

        print('Testing ', point_cloud_name)
        
        # load data
        x = load_sparse_tensor(input_filedir, order='rgb')

        # preprocess
        downsampler = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        true_down = downsampler(downsampler(x))

        from data_processing.data_utils import write_ply_ascii

        down_dir = out_folder + '/' + point_cloud_name + '_down.ply'

        write_ply_ascii(down_dir, true_down.C.cpu().numpy()[:, 1:] // 4,
                        (true_down.F.detach().cpu().numpy() * 255).round())

        x = ME.SparseTensor(features=rgb2yuv(x.F.clone()),
                            coordinates=x.C,
                            device=x.device)

        # guided filter
        guided_model = GuidedFilter(ckpt_u='./models/guided_filter/guided_filter_U.pth',
                      ckpt_v='./models/guided_filter/guided_filter_V.pth').to('cuda')

        for idx_rate, lambda_A in enumerate(tqdm(qs_list)):
            out_subfolder = out_folder + '/r' + str(idx_rate) + '_' + 'lambda_' + str(lambda_A)
            os.makedirs(out_subfolder, exist_ok=True)
            filename = out_subfolder + '/' + point_cloud_name

            print(filename)

            print('-------------------------------------------------------------------------------')
            print(f'Start compressing and decompressing {point_cloud_name}, lambda: {lambda_A}')

            results_one = {'lambda_A': lambda_A}

            # base layer - GPCC
            d = round((math.log(lambda_A) - math.log(64)) * 6)
            if d <= 0:
                qp = 40
            else:
                qp = 40 - d

            bin_dir = out_folder + '/' + point_cloud_name + '_qp_' + str(qp) +'_down.bin'
            rec_dir = out_folder + '/' + point_cloud_name + '_qp_' + str(qp) +'_down_rec.ply'

            print(f'base layer QP: {qp}')
            results_single = test(down_dir, bin_dir, rec_dir, transformType=0, qp=qp)
            gpcc_bytes = results_single['Enc_colors bitstream size']

            # enhancement layer - Neural Network

            # encode
            y_Bytes, z_Bytes, Guided_bytes, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, _ = model.encode(x, lambda_A, rec_dir, filename, guided_model)

            # decode
            out = model.decode(x, lambda_A, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag,
                                rec_dir, filename, guided_model, device)

            out = ME.SparseTensor(features=yuv2rgb(out.F.clone()),
                                    coordinates=out.C,
                                    device=out.device)

            # save
            coords = out.C.cpu().numpy()[:, 1:]
            feats = (out.F.detach().cpu().numpy() * 255).round()
            feats = np.clip(feats, 0, 255).astype('uint8')

            dec_dir = filename + '_dec.ply'

            from data_processing.data_utils import write_ply_ascii
            write_ply_ascii(dec_dir, coords, feats)

            print('Start evaluating performance')
            # rate
            Bytes = y_Bytes + z_Bytes
            bpp_attr_base = round(gpcc_bytes * 8 / float(x.__len__()), 4)
            bpp_attr_enhance = round((Bytes + Guided_bytes) * 8 / float(x.__len__()), 4)

            results_one['Total_bpp_attr'] = bpp_attr_base + bpp_attr_enhance
            results_one['bpp_attr_base'] = bpp_attr_base
            results_one['bpp_attr_enhance'] = bpp_attr_enhance

            # PSNR
            pc_error_results = pc_error(input_filedir, dec_dir, res=1023, color=True, show=False)

            key_list = ['  c[0],PSNRF', '  c[1],PSNRF', '  c[2],PSNRF']
            for k, v in pc_error_results.items(): 
                if k in key_list:
                    results_one[k] = v

            results_one['YUV PSNR'] = (6 * float(results_one['  c[0],PSNRF']) + float(
                results_one['  c[1],PSNRF']) + float(results_one['  c[2],PSNRF'])) / 8

            # PCQM
            cmd = './PCQM' + ' ' + input_filedir + ' ' + dec_dir + ' -r 0.004 -knn 20 -rx 2.0'

            std_out_and_err = run_cmd([cmd])

            std_out = str(std_out_and_err[0])
            pcqm = std_out.split('PCQM value is :')[1].split('\\n')[0]
            geo = std_out.split('Geo value is :')[1].split('\\n')[0]
            attr = std_out.split('Attr value is :')[1].split('\\n')[0]

            results_one['PCQM'] = pcqm
            results_one['Geo'] = geo
            results_one['Attr'] = attr
            results_one['1 - PCQM'] = 1 - float(pcqm)
            print(results_one)

            if idx_rate == 0: results_list = [results_one]
            else: results_list.append(results_one)

            del out
            torch.cuda.empty_cache()  # empty cache.

        # merge all rate
        results = {'filename': os.path.split(filename)[-1]}
        one_keys = []
        multi_keys = ['lambda_A',
                      'Total_bpp_attr', 'bpp_attr_base', 'bpp_attr_enhance',
                      'mseF,PSNR (p2point)', 'mseF,PSNR (p2plane)',
                      '  c[0],PSNRF', '  c[1],PSNRF', '  c[2],PSNRF', 'YUV PSNR',
                      'PCQM', 'Geo', 'Attr', '1 - PCQM']
        for idx_rate, results_one in enumerate(results_list):
            for k, v in results_one.items():
                if k in one_keys and idx_rate == 0: results[k] = v
                if k in multi_keys: results['R' + str(idx_rate) + '_' + k] = v
        results = pd.DataFrame([results])
        # merge all file
        if idx_file == 0:
            results_allfile = results.copy(deep=True)
        else:
            results_allfile = pd.concat([results_allfile, results], ignore_index=True)
        csvfile = os.path.join(output_rootdir, output_rootdir.split('/')[-2] + '.csv')
        results_allfile.to_csv(csvfile, index=False)
        print('save results to ', csvfile)

    return results_allfile


def test_joint_coding(geo_model_list, attr_model, g_qs_list, a_qs_list, input_rootdir, output_rootdir):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+'ply'), recursive=True))
    for idx_file, input_filedir in enumerate(tqdm(input_filedirs)):
        point_cloud_name = input_filedir[len(input_rootdir):].split('.')[0]
        out_folder = './' + output_rootdir + point_cloud_name
        os.makedirs(out_folder, exist_ok=True)

        print('Testing ', point_cloud_name)
        
        filename = out_folder + '/' + point_cloud_name

        # load data
        x = load_sparse_tensor(input_filedir, order='rgb')

        # preprocess
        downsampler = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        true_down = downsampler(downsampler(x))

        from data_processing.data_utils import write_ply_ascii

        down_dir = out_folder + '/' + point_cloud_name + '_down.ply'

        write_ply_ascii(down_dir, true_down.C.cpu().numpy()[:, 1:] // 4,
                        (true_down.F.detach().cpu().numpy() * 255).round())

        x = ME.SparseTensor(features=rgb2yuv(x.F.clone()),
                            coordinates=x.C,
                            device=x.device)

        # guided filter
        guided_model = GuidedFilter(ckpt_u='./models/guided_filter/guided_filter_U.pth',
                      ckpt_v='./models/guided_filter/guided_filter_V.pth').to('cuda')

        for idx_rate, (lambda_G, lambda_A) in enumerate(tqdm(zip(g_qs_list, a_qs_list))):
            out_subfolder = out_folder + '/r' + str(idx_rate) + '_' + 'lambda_' + str(lambda_A)
            os.makedirs(out_subfolder, exist_ok=True)
            filename = out_subfolder + '/' + point_cloud_name

            print('-------------------------------------------------------------------------------')
            print(f'Start compressing and decompressing {point_cloud_name}, lambda_G: {lambda_G} lambda: {lambda_A}')

            results_one = {'lambda_G': lambda_G}
            results_one['lambda_A'] = lambda_A

            if lambda_G <= 512.:
                geo_model = geo_model_list[0]
            else:
                lambda_G = lambda_G // 8
                geo_model = geo_model_list[1]

            # base layer - GPCC
            d = round((math.log(lambda_A) - math.log(64)) * 6)
            if d <= 0:
                qp = 40
            else:
                qp = 40 - d

            bin_dir = out_folder + '/' + point_cloud_name + '_qp_' + str(qp) +'_down.bin'
            rec_dir = out_folder + '/' + point_cloud_name + '_qp_' + str(qp) +'_down_rec.ply'

            print(f'base layer QP: {qp}')
            results_single = test(down_dir, bin_dir, rec_dir, transformType=0, qp=qp)
            
            geo_gpcc_bytes = results_single['Enc_positions bitstream size']
            attr_gpcc_bytes = results_single['Enc_colors bitstream size']

            # enhancement layer - Neural Network

            # geo encode
            y_Bytes, z_Bytes, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, nums_list = geo_model.encode(x, lambda_G, filename)

            # geo decode
            out = geo_model.decode(lambda_G, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, nums_list, down_dir, filename, device)
            
            # rate
            Bytes = y_Bytes + z_Bytes

            bpp_geo_base = round(geo_gpcc_bytes * 8 / float(x.__len__()), 4)
            bpp_geo_enhance = round(Bytes * 8 / float(x.__len__()), 4)

            # recolor
            attr_in_F = knn_interpolation(x, out, k=1)
            attr_in = ME.SparseTensor(
                features=attr_in_F, coordinates=out.C,
                device=out.device)
            geo_dec = attr_in

            # attr encode
            y_Bytes, z_Bytes, Guided_bytes, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, _ = attr_model.encode(attr_in, lambda_A, rec_dir, filename, guided_model)
            
            # attr decode
            out = attr_model.decode(geo_dec, lambda_A, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, rec_dir, filename, guided_model, device)

            out = ME.SparseTensor(features=yuv2rgb(out.F.clone()),
                                    coordinates=out.C,
                                    device=out.device)

            # save
            coords = out.C.cpu().numpy()[:, 1:]
            feats = (out.F.detach().cpu().numpy() * 255).round()
            feats = np.clip(feats, 0, 255).astype('uint8')

            dec_dir = filename + '_dec.ply'

            from data_processing.data_utils import write_ply_ascii
            write_ply_ascii(dec_dir, coords, feats)

            print('Start evaluating performance')
            # rate
            Bytes = y_Bytes + z_Bytes
            bpp_attr_base = round(attr_gpcc_bytes * 8 / float(x.__len__()), 4)
            bpp_attr_enhance = round((Bytes + Guided_bytes) * 8 / float(x.__len__()), 4)

            results_one['Total_bpp'] = bpp_geo_base + bpp_geo_enhance + bpp_attr_base + bpp_attr_enhance

            results_one['Total_bpp_geo'] = bpp_geo_base + bpp_geo_enhance
            results_one['bpp_geo_base'] = bpp_geo_base
            results_one['bpp_geo_enhance'] = bpp_geo_enhance

            results_one['Total_bpp_attr'] = bpp_attr_base + bpp_attr_enhance
            results_one['bpp_attr_base'] = bpp_attr_base
            results_one['bpp_attr_enhance'] = bpp_attr_enhance

            # PSNR
            pc_error_results = pc_error(input_filedir, dec_dir, res=1023, color=True, show=False)

            key_list = ['mseF,PSNR (p2point)', 'mseF,PSNR (p2plane)', '  c[0],PSNRF', '  c[1],PSNRF', '  c[2],PSNRF']
            for k, v in pc_error_results.items(): 
                if k in key_list:
                    results_one[k] = v

            results_one['YUV PSNR'] = (6 * float(results_one['  c[0],PSNRF']) + float(
                results_one['  c[1],PSNRF']) + float(results_one['  c[2],PSNRF'])) / 8

            # PCQM
            cmd = './PCQM' + ' ' + input_filedir + ' ' + dec_dir + ' -r 0.004 -knn 20 -rx 2.0'
            # print(cmd)
            std_out_and_err = run_cmd([cmd])

            std_out = str(std_out_and_err[0])
            pcqm = std_out.split('PCQM value is :')[1].split('\\n')[0]
            geo = std_out.split('Geo value is :')[1].split('\\n')[0]
            attr = std_out.split('Attr value is :')[1].split('\\n')[0]

            results_one['PCQM'] = pcqm
            results_one['Geo'] = geo
            results_one['Attr'] = attr
            results_one['1 - PCQM'] = 1 - float(pcqm)
            print(results_one)

            if idx_rate == 0:
                results_list = [results_one]
            else:
                results_list.append(results_one)

            del out
            torch.cuda.empty_cache()  # empty cache.

        # merge all rate
        results = {'filename': point_cloud_name}
        one_keys = []
        multi_keys = ['lambda_G', 'lambda_A',
                      'Total_bpp',
                      'Total_bpp_geo', 'bpp_geo_base', 'bpp_geo_enhance', 
                      'Total_bpp_attr', 'bpp_attr_base', 'bpp_attr_enhance', 
                      'mseF,PSNR (p2point)', 'mseF,PSNR (p2plane)',
                      '  c[0],PSNRF', '  c[1],PSNRF', '  c[2],PSNRF', 'YUV PSNR',
                      'PCQM', 'Geo', 'Attr', '1 - PCQM']
        for idx_rate, results_one in enumerate(results_list):
            for k, v in results_one.items():
                if k in one_keys and idx_rate == 0: results[k] = v
                if k in multi_keys: results['R' + str(idx_rate) + '_' + k] = v
        results = pd.DataFrame([results])
        # merge all file
        if idx_file == 0:
            results_allfile = results.copy(deep=True)
        else:
            results_allfile = pd.concat([results_allfile, results], ignore_index=True)
        csvfile = os.path.join(output_rootdir, output_rootdir.split('/')[-2] + '.csv')
        results_allfile.to_csv(csvfile, index=False)
        print('save results to ', csvfile)

    return results_allfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", default='attribute')
    parser.add_argument("--input_rootdir", default='../data/static/8iVFB/')
    parser.add_argument("--model_name", default='attribute_coding')
    parser.add_argument('--g_lambda_list', type=float, nargs='+', default=[64., 1024.], help="64. ~ 4096.")
    parser.add_argument('--a_lambda_list', type=float, nargs='+', default=[64., 2048.], help="64. ~ 2048.")
    args = parser.parse_args()

    model_name = args.model_name
    args.output_rootdir = 'output/' + model_name + '/'
    os.makedirs(args.output_rootdir, exist_ok=True)
    import shutil

    shutil.copy('test_static.py', os.path.join(args.output_rootdir))
    shutil.copy('./models/basic_module.py', os.path.join(args.output_rootdir))
    shutil.copy('model_static.py', os.path.join(args.output_rootdir))
    print('dbg:\t output_rootdir:\t', args.output_rootdir)

    g_lambda_list = args.g_lambda_list
    a_lambda_list = args.a_lambda_list

    if args.mode == 'geometry' or args.mode == 'joint':
        from model_static import PCGCModel
        geo_static_low_model = PCGCModel().to(device)
        ckptsdir = 'ckpts/static/geo/geo_64_512_low.pth'
        ckpt = torch.load(ckptsdir, map_location='cuda:0')
        geo_static_low_model.load_state_dict(ckpt['model'])

        geo_static_high_model = PCGCModel().to(device)
        ckptsdir = 'ckpts/static/geo/geo_64_512_high.pth'
        ckpt = torch.load(ckptsdir, map_location='cuda:0')
        geo_static_high_model.load_state_dict(ckpt['model'])

    if args.mode == 'attribute' or args.mode == 'joint': 
        from model_static import PCACModel
        attr_model = PCACModel().to(device)
        ckptsdir = 'ckpts/static/attr/attr_64_2048.pth'
        ckpt = torch.load(ckptsdir, map_location='cuda:0')
        attr_model.load_state_dict(ckpt['model'])


    if args.mode == 'geometry':
        results = test_geometry_coding([geo_static_low_model, geo_static_high_model], g_lambda_list, args.input_rootdir, args.output_rootdir)

    if args.mode == 'attribute':
        results = test_attribute_coding(attr_model, a_lambda_list, args.input_rootdir, args.output_rootdir)

    if args.mode == 'joint':
        results = test_joint_coding([geo_static_low_model, geo_static_high_model], attr_model, g_lambda_list, a_lambda_list, 
                                   args.input_rootdir, args.output_rootdir)
        
    print(results.mean())
    print(results)
