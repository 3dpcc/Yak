import numpy as np
import torch
import os, glob, argparse, time
import pandas as pd
from tqdm import tqdm
import math
import MinkowskiEngine as ME

from data_processing.data_utils import load_sparse_tensor, yuv2rgb, rgb2yuv, run_cmd
from extension.pc_error import pc_error
from extension.gpcc import gpcc_encode_inter, gpcc_decode_inter
from data_processing.data_utils import knn_interpolation

from models.guided_filter import GuidedFilter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_geometry_coding(static_model, dynamic_model, g_lambda, input_rootdir, output_rootdir, first, count):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+'ply'), recursive=True))

    # preprocess
    os.makedirs(output_rootdir + '/down/', exist_ok=True)

    for idx_file, input_filedir in enumerate(tqdm(input_filedirs)):
        if idx_file == int(count):
            break
        x = load_sparse_tensor(input_filedir, order='rgb')
        downsampler = torch.nn.Sequential(*[ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)] * 2).to(device)

        x_true_down = downsampler(x)

        from data_processing.data_utils import write_ply_ascii

        x_true_down_dir = output_rootdir + '/down/' + input_filedir[len(input_rootdir):].split('.')[
            0] + '_down.ply'

        write_ply_ascii(x_true_down_dir, x_true_down.C.cpu().numpy()[:, 1:] // (2 ** 2),
                        (x_true_down.F.detach().cpu().numpy() * 255).round())
    
    point_cloud_rootdir = output_rootdir + '/down/' + input_filedirs[0].split('/')[-1].split('.')[0].split('vox10')[
        0] + 'vox10_%d_down.ply'
    
    
    # base layer - GPCC
    os.makedirs(output_rootdir + '/bin/', exist_ok=True)
    os.makedirs(output_rootdir + '/rec/', exist_ok=True)

    rec_dir = os.path.join(output_rootdir + '/rec/',
                            input_filedirs[0].split('/')[-1].split('.')[0].split('vox10')[
                                0] + 'vox10_%d_down' + '_qp' + str(51) + '.ply')

    bin_dir = os.path.join(output_rootdir + '/bin',
                            input_filedirs[0].split('/')[-1].split('.')[0].split('vox10')[
                                0] + 'vox10_down' + '_qp' + str(51) + '.bin')

    results_enc = gpcc_encode_inter(point_cloud_rootdir, first, count, bin_dir, posQuantscale=1,
                                    transformType=0, qp=51, show=False)
    
    results_dec = gpcc_decode_inter(rec_dir, first, count, bin_dir, show=False)

    gpcc_result = results_enc
    base_file_list = sorted(glob.glob(os.path.join(output_rootdir + '/rec/', f'*' + 'ply'), recursive=True))

    # enhancement layer - Neural Network
    previous_x = None
    for idx_file, input_filedir in enumerate(tqdm(input_filedirs)):
        if idx_file == int(count):
            break
        if idx_file == 0:
            input_filedir = input_filedirs[idx_file]
            filename = './' + output_rootdir + os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0])
            point_cloud_name = filename.split('/')[-1]

            # load data
            x = load_sparse_tensor(input_filedir, order='rgb')

            print('-------------------------------------------------------------------------------')
            print(f'Start compressing and decompressing {point_cloud_name}, lambda: {g_lambda}')

            results_one = {'lambda_G': g_lambda}

            if g_lambda > 512.:
                lambda_G = g_lambda // 8
            else:
                lambda_G = g_lambda

            # encode 
            y_Bytes, z_Bytes, index_gp1, index_gp2, z_gp1_flag, \
                z_gp2_flag, nums_list = static_model.encode(x, lambda_G, filename)

            # decode
            out = static_model.decode(lambda_G, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, nums_list,
                                base_file_list[idx_file], filename, device)

            # save
            coords = out.C.detach().cpu().numpy()[:, 1:]
            dec_dir = filename + '_dec.ply'
            from data_processing.data_utils import write_ply_ascii_geo
            write_ply_ascii_geo(dec_dir, coords)

            print('Start evaluating performance')
            # rate
            Bytes = y_Bytes + z_Bytes
            gpcc_bytes = gpcc_result['positions bitstream size_' + str(idx_file)]
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
            print(results_one)

            results_list = [results_one]

            previous_x = ME.SparseTensor(features=out.F, coordinates=out.C, device=out.device)

            # del out, out_set, currnet_gpcc
            torch.cuda.empty_cache()  # empty cache.

        else:
            input_filedir = input_filedirs[idx_file]
            filename = './' + output_rootdir + os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0])
            point_cloud_name = filename.split('/')[-1]

            # load data
            x = load_sparse_tensor(input_filedir, order='rgb')

            print('-------------------------------------------------------------------------------')
            print(f'Start compressing and decompressing {point_cloud_name}, lambda: {g_lambda}')

            results_one = {'lambda_G': g_lambda}

            if g_lambda > 512.:
                lambda_G = g_lambda // 8
            else:
                lambda_G = g_lambda

            # encode
            y_Bytes, z_Bytes, index_gp1, index_gp2, z_gp1_flag, \
                z_gp2_flag, nums_list = dynamic_model.encode(x, previous_x, lambda_G, filename)
            
            # decode
            out = dynamic_model.decode(previous_x, lambda_G, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, nums_list,
                                base_file_list[idx_file], filename, device)
            
            # save
            coords = out.C.detach().cpu().numpy()[:, 1:]
            dec_dir = filename + '_dec.ply'
            from data_processing.data_utils import write_ply_ascii_geo
            write_ply_ascii_geo(dec_dir, coords)

            print('Start evaluating performance')
            # rate
            Bytes = y_Bytes + z_Bytes
            gpcc_bytes = gpcc_result['positions bitstream size_' + str(idx_file)]
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
            print(results_one)

            results_list = [results_one]

            previous_x = ME.SparseTensor(features=out.F, coordinates=out.C, device=out.device)

            # del out, out_set, currnet_gpcc
            torch.cuda.empty_cache()  # empty cache.

        # merge all rate
        results = {'filename': point_cloud_name}
        one_keys = []
        multi_keys = ['lambda_G',
                      'Total_bpp_geo', 'bpp_geo_base', 'bpp_geo_enhance', 
                      'mseF,PSNR (p2point)', 'mseF,PSNR (p2plane)',
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


def test_attribute_coding(static_model, dynamic_model, a_lambda, input_rootdir, output_rootdir, first, count):
    # guided filter
    guided_model = GuidedFilter(ckpt_u='./models/guided_filter/guided_filter_U.pth',
                    ckpt_v='./models/guided_filter/guided_filter_V.pth').to('cuda')
    
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+'ply'), recursive=True))

    # preprocess
    os.makedirs(output_rootdir + '/down/', exist_ok=True)

    for idx_file, input_filedir in enumerate(tqdm(input_filedirs)):
        if idx_file == int(count):
            break
        x = load_sparse_tensor(input_filedir, order='rgb')
        downsampler = torch.nn.Sequential(*[ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)] * 2).to(device)

        x_true_down = downsampler(x)

        from data_processing.data_utils import write_ply_ascii

        x_true_down_dir = output_rootdir + '/down/' + input_filedir[len(input_rootdir):].split('.')[
            0] + '_down.ply'

        write_ply_ascii(x_true_down_dir, x_true_down.C.cpu().numpy()[:, 1:] // (2 ** 2),
                        (x_true_down.F.detach().cpu().numpy() * 255).round())
    
    point_cloud_rootdir = output_rootdir + '/down/' + input_filedirs[0].split('/')[-1].split('.')[0].split('vox10')[
        0] + 'vox10_%d_down.ply'
    
    
    # base layer - GPCC
    os.makedirs(output_rootdir + '/bin/', exist_ok=True)
    os.makedirs(output_rootdir + '/rec/', exist_ok=True)

    d = round((math.log(a_lambda) - math.log(64)) * 6)
    if d <= 0:
        qp = 40
    else:
        qp = 40 - d

    rec_dir = os.path.join(output_rootdir + '/rec/',
                            input_filedirs[0].split('/')[-1].split('.')[0].split('vox10')[
                                0] + 'vox10_%d_down' + '_qp' + str(qp) + '.ply')

    bin_dir = os.path.join(output_rootdir + '/bin',
                            input_filedirs[0].split('/')[-1].split('.')[0].split('vox10')[
                                0] + 'vox10_down' + '_qp' + str(qp) + '.bin')

    results_enc = gpcc_encode_inter(point_cloud_rootdir, first, count, bin_dir, posQuantscale=1,
                                    transformType=0, qp=qp, show=False)
    
    results_dec = gpcc_decode_inter(rec_dir, first, count, bin_dir, show=False)

    gpcc_result = results_enc
    base_file_list = sorted(glob.glob(os.path.join(output_rootdir + '/rec/', f'*' + 'ply'), recursive=True))

    previous_x = None
    for idx_file, input_filedir in enumerate(tqdm(input_filedirs)):
        lambda_A = a_lambda
        if idx_file == int(count):
            break
        if idx_file == 0:
            input_filedir = input_filedirs[idx_file]
            filename = './' + output_rootdir + os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0])
            point_cloud_name = filename.split('/')[-1]

            # load data
            x = load_sparse_tensor(input_filedir, order='rgb')
            x = ME.SparseTensor(features=rgb2yuv(x.F.clone()),
                                coordinates=x.C,
                                device=x.device)
            
            print('-------------------------------------------------------------------------------')
            print(f'Start compressing and decompressing {point_cloud_name}, lambda: {lambda_A}')

            results_one = {'lambda_A': lambda_A}

            # encode
            y_Bytes, z_Bytes, Guided_bytes, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, \
                enc_out = static_model.encode(x, lambda_A, base_file_list[idx_file], filename, guided_model)

            # decode
            out = static_model.decode(x, lambda_A, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag,
                                base_file_list[idx_file], filename, guided_model, device)

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
            gpcc_bytes = gpcc_result['colors bitstream size_' + str(idx_file)]
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

            results_list = [results_one]

            previous_x = ME.SparseTensor(features=out.F, coordinates=out.C, device=out.device)

            # del out, out_set, currnet_gpcc
            torch.cuda.empty_cache()  # empty cache.

        else:
            input_filedir = input_filedirs[idx_file]
            filename = './' + output_rootdir + os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0])
            point_cloud_name = filename.split('/')[-1]

            # load data
            x = load_sparse_tensor(input_filedir, order='rgb')
            x = ME.SparseTensor(features=rgb2yuv(x.F.clone()),
                                coordinates=x.C,
                                device=x.device)
            
            print('-------------------------------------------------------------------------------')
            print(f'Start compressing and decompressing {point_cloud_name}, lambda: {lambda_A}')

            results_one = {'lambda_A': lambda_A}

            # encode
            y_Bytes, Guided_bytes, index_gp1, index_gp2 = \
                dynamic_model.encode(x, previous_x, lambda_A, base_file_list[idx_file], filename, guided_model)
            
            # decode
            out = dynamic_model.decode(x, previous_x, lambda_A, index_gp1, index_gp2,
                                base_file_list[idx_file], filename, guided_model, device)

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
            gpcc_bytes = gpcc_result['colors bitstream size_' + str(idx_file)]
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

            results_list = [results_one]

        # merge all rate
        results = {'filename': point_cloud_name}
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



def test_joint_coding(geo_static_model, geo_dynamic_model, attr_static_model, attr_dynamic_model, g_lambda, a_lambda, 
                                   input_rootdir, output_rootdir, first, count):
    # guided filter
    guided_model = GuidedFilter(ckpt_u='./models/guided_filter/guided_filter_U.pth',
                    ckpt_v='./models/guided_filter/guided_filter_V.pth').to('cuda')
    
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+'ply'), recursive=True))

    # preprocess
    os.makedirs(output_rootdir + '/down/', exist_ok=True)

    for idx_file, input_filedir in enumerate(tqdm(input_filedirs)):
        if idx_file == int(count):
            break
        x = load_sparse_tensor(input_filedir, order='rgb')
        downsampler = torch.nn.Sequential(*[ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)] * 2).to(device)

        x_true_down = downsampler(x)

        from data_processing.data_utils import write_ply_ascii

        x_true_down_dir = output_rootdir + '/down/' + input_filedir[len(input_rootdir):].split('.')[
            0] + '_down.ply'

        write_ply_ascii(x_true_down_dir, x_true_down.C.cpu().numpy()[:, 1:] // (2 ** 2),
                        (x_true_down.F.detach().cpu().numpy() * 255).round())
    
    point_cloud_rootdir = output_rootdir + '/down/' + input_filedirs[0].split('/')[-1].split('.')[0].split('vox10')[
        0] + 'vox10_%d_down.ply'
    
    
    # base layer - GPCC
    os.makedirs(output_rootdir + '/bin/', exist_ok=True)
    os.makedirs(output_rootdir + '/rec/', exist_ok=True)

    d = round((math.log(a_lambda) - math.log(64)) * 6)
    if d <= 0:
        qp = 40
    else:
        qp = 40 - d

    rec_dir = os.path.join(output_rootdir + '/rec/',
                            input_filedirs[0].split('/')[-1].split('.')[0].split('vox10')[
                                0] + 'vox10_%d_down' + '_qp' + str(qp) + '.ply')

    bin_dir = os.path.join(output_rootdir + '/bin',
                            input_filedirs[0].split('/')[-1].split('.')[0].split('vox10')[
                                0] + 'vox10_down' + '_qp' + str(qp) + '.bin')

    results_enc = gpcc_encode_inter(point_cloud_rootdir, first, count, bin_dir, posQuantscale=1,
                                    transformType=0, qp=qp, show=False)
    
    results_dec = gpcc_decode_inter(rec_dir, first, count, bin_dir, show=False)

    gpcc_result = results_enc
    base_file_list = sorted(glob.glob(os.path.join(output_rootdir + '/rec/', f'*' + 'ply'), recursive=True))

    previous_x = None
    for idx_file, input_filedir in enumerate(tqdm(input_filedirs)):
        lambda_A = a_lambda
        if idx_file == int(count):
            break
        if idx_file == 0:
            input_filedir = input_filedirs[idx_file]
            filename = './' + output_rootdir + os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0])
            point_cloud_name = filename.split('/')[-1]

            # load data
            x = load_sparse_tensor(input_filedir, order='rgb')
            x = ME.SparseTensor(features=rgb2yuv(x.F.clone()),
                                coordinates=x.C,
                                device=x.device)
            
            print('-------------------------------------------------------------------------------')
            print(f'Start compressing and decompressing {point_cloud_name}, lambda_G: {g_lambda} lambda_A: {a_lambda}')

            if g_lambda > 512.:
                lambda_G = g_lambda // 8
            else:
                lambda_G = g_lambda

            results_one = {'lambda_G': g_lambda}
            results_one['lambda_A'] = a_lambda

            # geo encode
            y_Bytes, z_Bytes, index_gp1, index_gp2, z_gp1_flag, \
                z_gp2_flag, nums_list = geo_static_model.encode(x, lambda_G, filename)
            out = geo_static_model.decode(lambda_G, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag,
                                        nums_list, base_file_list[idx_file], filename, device)
            # rate
            Bytes = y_Bytes + z_Bytes
            gpcc_bytes = gpcc_result['positions bitstream size_' + str(idx_file)]
            bpp_geo_base = round(gpcc_bytes * 8 / float(x.__len__()), 4)
            bpp_geo_enhance = round(Bytes * 8 / float(x.__len__()), 4)

            # recolor
            attr_in_F = knn_interpolation(x, out, k=1)
            attr_in = ME.SparseTensor(
                features=attr_in_F, coordinates=out.C,
                device=out.device)
            geo_dec = attr_in

            # attr encode
            y_Bytes, z_Bytes, Guided_bytes, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, \
                enc_out = attr_static_model.encode(attr_in, lambda_A, base_file_list[idx_file], filename, guided_model)

            # attr decode
            out = attr_static_model.decode(geo_dec, lambda_A, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag,
                                base_file_list[idx_file], filename, guided_model, device)

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
            gpcc_bytes = gpcc_result['colors bitstream size_' + str(idx_file)]
            bpp_attr_base = round(gpcc_bytes * 8 / float(x.__len__()), 4)
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

            results_list = [results_one]

            previous_x = ME.SparseTensor(features=out.F, coordinates=out.C, device=out.device)

            del out, x, attr_in, geo_dec
            torch.cuda.empty_cache()  # empty cache.

        else:
            input_filedir = input_filedirs[idx_file]
            filename = './' + output_rootdir + os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0])
            point_cloud_name = filename.split('/')[-1]

            # load data
            x = load_sparse_tensor(input_filedir, order='rgb')
            x = ME.SparseTensor(features=rgb2yuv(x.F.clone()),
                                coordinates=x.C,
                                device=x.device)
            
            print('-------------------------------------------------------------------------------')
            print(f'Start compressing and decompressing {point_cloud_name}, lambda_G: {g_lambda} lambda_A: {a_lambda}')

            if g_lambda > 512.:
                lambda_G = g_lambda // 8
            else:
                lambda_G = g_lambda

            results_one = {'lambda_G': g_lambda}
            results_one['lambda_A'] = lambda_A

            # geo encode
            y_Bytes, z_Bytes, index_gp1, index_gp2, z_gp1_flag, \
                z_gp2_flag, nums_list = geo_dynamic_model.encode(x, previous_x, lambda_G, filename)
            
            # geo decode
            out = geo_dynamic_model.decode(previous_x, lambda_G, index_gp1, index_gp2, z_gp1_flag,
                                        z_gp2_flag, nums_list, base_file_list[idx_file], filename, device)

            # rate
            Bytes = y_Bytes + z_Bytes
            gpcc_bytes = gpcc_result['positions bitstream size_' + str(idx_file)]
            bpp_geo_base = round(gpcc_bytes * 8 / float(x.__len__()), 4)
            bpp_geo_enhance = round(Bytes * 8 / float(x.__len__()), 4)

            # recolor
            attr_in_F = knn_interpolation(x, out, k=1)
            attr_in = ME.SparseTensor(
                features=attr_in_F, coordinates=out.C,
                device=out.device)
            geo_dec = attr_in

            # attr encode
            y_Bytes, Guided_bytes, index_gp1, index_gp2 = \
                attr_dynamic_model.encode(attr_in, previous_x, lambda_A, base_file_list[idx_file], filename , guided_model)

            # attr decode
            out = attr_dynamic_model.decode(geo_dec, previous_x, lambda_A, index_gp1, index_gp2,
                                base_file_list[idx_file], filename , guided_model, device)

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
            gpcc_bytes = gpcc_result['colors bitstream size_' + str(idx_file)]
            bpp_attr_base = round(gpcc_bytes * 8 / float(x.__len__()), 4)
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

            results_list = [results_one]

            del out, x, attr_in, geo_dec
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
    parser.add_argument("--mode", default='geometry')
    parser.add_argument("--input_rootdir", default='../data/dynamic/basketball_player_vox10')
    parser.add_argument("--model_name", default='inter_geometry_coding/basketball_player_vox10')
    parser.add_argument('--g_lambda', type=float, default=64., help="64. ~ 4096.")
    parser.add_argument('--a_lambda', type=float, default=64., help="64. ~ 2048.")
    parser.add_argument("--first", type=str, default='10000001')
    parser.add_argument("--count", type=str, default='2')

    args = parser.parse_args()

    model_name = args.model_name
    args.output_rootdir = 'output/' + model_name + '/'

    if args.mode == 'geometry':
        args.output_rootdir = args.output_rootdir + 'lambda_G_' + str(args.g_lambda)
    elif args.mode == 'attribute':
        args.output_rootdir = args.output_rootdir + 'lambda_A_' + str(args.a_lambda)
    else:
        args.output_rootdir = args.output_rootdir + 'lambda_G_' + str(args.g_lambda) + '_'  + 'lambda_A_' + str(args.a_lambda)

    os.makedirs(args.output_rootdir, exist_ok=True)
    import shutil

    shutil.copy('test_static.py', os.path.join(args.output_rootdir))
    shutil.copy('./models/basic_module.py', os.path.join(args.output_rootdir))
    shutil.copy('model_static.py', os.path.join(args.output_rootdir))
    print('dbg:\t output_rootdir:\t', args.output_rootdir)

    g_lambda = args.g_lambda
    a_lambda = args.a_lambda

    if args.mode == 'geometry' or args.mode == 'joint':
        if g_lambda <= 512.:
            # low rate model
            from model_static import PCGCModel
            geo_static_model = PCGCModel().to(device)
            ckptsdir = 'ckpts/static/geo/geo_64_512_low.pth'
            ckpt = torch.load(ckptsdir)
            geo_static_model.load_state_dict(ckpt['model'])

            from model_dynamic import PCGCInterModel
            geo_dynamic_model = PCGCInterModel().to(device)
            ckptsdir = 'ckpts/dynamic/geo/geo_dynamic_64_512_low.pth'
            ckpt = torch.load(ckptsdir)
            geo_dynamic_model.load_state_dict(ckpt['model'])
        else:
            # high rate model
            from model_static import PCGCModel
            geo_static_model = PCGCModel().to(device)
            ckptsdir = 'ckpts/static/geo/geo_64_512_high.pth'
            ckpt = torch.load(ckptsdir)
            geo_static_model.load_state_dict(ckpt['model'])

            from model_dynamic import PCGCInterModel
            geo_dynamic_model = PCGCInterModel().to(device)
            ckptsdir = 'ckpts/dynamic/geo/geo_dynamic_64_512_high.pth'
            ckpt = torch.load(ckptsdir)
            geo_dynamic_model.load_state_dict(ckpt['model'])

    if args.mode == 'attribute' or args.mode == 'joint': 
        from model_static import PCACModel
        attr_static_model = PCACModel().to(device)
        ckptsdir = 'ckpts/static/attr/attr_64_2048.pth'
        ckpt = torch.load(ckptsdir)
        attr_static_model.load_state_dict(ckpt['model'])

        from model_dynamic import PCACInterModel
        attr_dynamic_model = PCACInterModel().to(device)
        ckptsdir = 'ckpts/dynamic/attr/attr_dynamic_64_2048.pth'
        ckpt = torch.load(ckptsdir)
        attr_dynamic_model.load_state_dict(ckpt['model'])

    if args.mode == 'geometry':
        results = test_geometry_coding(geo_static_model, geo_dynamic_model, g_lambda, args.input_rootdir, args.output_rootdir, args.first, args.count)

    if args.mode == 'attribute':
        results = test_attribute_coding(attr_static_model, attr_dynamic_model, a_lambda, args.input_rootdir, args.output_rootdir, args.first, args.count)

    if args.mode == 'joint':
        results = test_joint_coding(geo_static_model, geo_dynamic_model, attr_static_model, attr_dynamic_model, g_lambda, a_lambda, 
                                   args.input_rootdir, args.output_rootdir, args.first, args.count)
        
    print(results.mean())
    print(results)