import os, time
import numpy as np
import subprocess

rootdir = os.path.split(__file__)[0]
if rootdir == '': rootdir = '.'

def get_points_number(filedir):
    plyfile = open(filedir)
    line = plyfile.readline()
    while line.find("element vertex") == -1:
        line = plyfile.readline()
    number = int(line.split(' ')[-1][:-1])

    return number


def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item)
        except ValueError:
            continue

    return number


def gpcc_encode_inter(filedir, first, count, bin_dir, posQuantscale=1, transformType=0, qp=22, test_time=True,
                      show=False):
    """Compress point cloud losslessly using MPEG GeS-TM.
    You can download and install TMC13 from https://github.com/MPEGGroup/mpeg-pcc-tmc13
    """

    config = ' --trisoupNodeSizeLog2=0' + \
             ' --neighbourAvailBoundaryLog2=8' + \
             ' --qtbtEnabled=0' + \
             ' --inferredDirectCodingMode=0' + \
             ' --positionQuantizationScale=' + str(posQuantscale) + \
             ' --randomAccessPeriod=32' + \
             ' --interPredictionEnabled=1' + \
             ' --motionParamPreset=2' + \
             ' --entropyContinuationEnabled=1' + \
             ' --GoFGeometryEntropyContinuationEnabled=1' + \
             ' --autoSeqBbox=0' + \
             ' --seqOrigin=0,0,0' + \
             ' --seqSizeWhd=256,256,256' + \
             ' --rahtInterPredictionEnabled=1' + \
             ' --randomAccessPeriod=100'


    # lossless
    if posQuantscale == 1:
        config += ' --mergeDuplicatedPoints=0'
    else:
        config += ' --mergeDuplicatedPoints=1'
    # attr (raht)
    if qp is not None:
        config += ' --convertPlyColourspace=1'
        if transformType == 0:
            config += ' --transformType=0'
        else:
            raise ValueError('transformType=' + str(transformType))
        config += ' --qp=' + str(qp) + \
                  ' --qpChromaOffset=0' + \
                  ' --bitdepth=8' + \
                  ' --attrOffset=0' + \
                  ' --attrScale=1' + \
                  ' --attribute=color'
    # headers
    headers = ['positions bitstream size', 'Total bitstream size']
    if test_time: headers += ['positions processing time (user)', 'Processing time (user)', 'Processing time (wall)']
    if qp is not None: headers += ['colors bitstream size']
    if qp is not None and test_time:  headers += ['colors processing time (user)']

    subp = subprocess.Popen(rootdir + '/tmc3 --mode=0' + config + \
                            ' --firstFrameNum=' + first + \
                            ' --frameCount=' + count + \
                            ' --uncompressedDataPath=' + filedir + \
                            ' --compressedStreamPath=' + bin_dir,
                            shell=True, stdout=subprocess.PIPE)

    index = 0
    results = {}
    for _, key in enumerate(headers):
        results[key] = 0

    c = subp.stdout.readline()
    while c:
        if show: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key + '_' + str(index)] = value
                if key == 'colors processing time (user)':
                    index += 1

        c = subp.stdout.readline()

    return results


def gpcc_decode_inter(rec_dir, first, count, bin_dir, attr=True, test_geo=True, test_attr=True, show=False):
    if attr:
        config = ' --convertPlyColourspace=1'
    else:
        config = ''

    subp = subprocess.Popen(rootdir + '/tmc3 --mode=1' + config + \
                            ' --firstFrameNum=' + first + \
                            ' --frameCount=' + count + \
                            ' --compressedStreamPath=' + bin_dir + \
                            ' --reconstructedDataPath=' + rec_dir + \
                            ' --outputBinaryPly=0',
                            shell=True, stdout=subprocess.PIPE)

    # headers
    if test_geo: headers = ['positions bitstream size', 'positions processing time (user)',
                            'Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    if test_attr: headers += ['colors bitstream size', 'colors processing time (user)']
    # headers = []
    results = {}

    for _, key in enumerate(headers):
        results[key] = 0

    index = 0
    c = subp.stdout.readline()
    while c:
        if show: print(c)
        line = c.decode(encoding='utf-8')
        # print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key + '_' + str(index)] = value
                if key == 'colors processing time (user)':
                    index += 1

        c = subp.stdout.readline()

    return results

def gpcc_encode(filedir, bin_dir, posQuantscale=1, transformType=0, qp=22, test_time=False, show=False):
    """Compress point cloud losslessly using MPEG G-PCC.
    You can download and install TMC13 from https://github.com/MPEGGroup/mpeg-pcc-tmc13
    """
    config = ' --trisoupNodeSizeLog2=0' + \
             ' --neighbourAvailBoundaryLog2=8' + \
             ' --maxNumQtBtBeforeOt=4' + \
             ' --minQtbtSizeLog2=0' + \
             ' --autoSeqBbox=0' + \
             ' --seqOrigin=0,0,0' + \
             ' --seqSizeWhd=256,256,256' + \
             ' --positionQuantizationScale=' + str(posQuantscale)

    # lossless
    if posQuantscale == 1:
        config += ' --mergeDuplicatedPoints=0' + \
                  ' --inferredDirectCodingMode=1'
    else:
        config += ' --mergeDuplicatedPoints=1'
    # attr (raht)
    if qp is not None:
        config += ' --convertPlyColourspace=1'
        if transformType == 0:
            config += ' --transformType=0'
        elif transformType == 2:
            print('dbg:\t transformType=', transformType)
            config += ' --transformType=2' + \
                      ' --numberOfNearestNeighborsInPrediction=3' + \
                      ' --levelOfDetailCount=12' + \
                      ' --lodDecimator=0' + \
                      ' --adaptivePredictionThreshold=64'
        else:
            raise ValueError('transformType=' + str(transformType))
        config += ' --qp=' + str(qp) + \
                  ' --qpChromaOffset=0' + \
                  ' --bitdepth=8' + \
                  ' --attrOffset=0' + \
                  ' --attrScale=1' + \
                  ' --attribute=color'
    # headers
    headers = ['positions bitstream size', 'Total bitstream size']
    if test_time: headers += ['positions processing time (user)', 'Processing time (user)', 'Processing time (wall)']
    if qp is not None: headers += ['colors bitstream size']
    if qp is not None and test_time:  headers += ['colors processing time (user)']

    #
    subp = subprocess.Popen(rootdir + '/tmc3 --mode=0' + config + \
                            ' --uncompressedDataPath=' + filedir + \
                            ' --compressedStreamPath=' + bin_dir,
                            shell=True, stdout=subprocess.PIPE)
    results = {}
    for _, key in enumerate(headers):
        results[key] = 0

    c = subp.stdout.readline()
    while c:
        if show: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] += value

        c = subp.stdout.readline()

    return results


def gpcc_decode(bin_dir, rec_dir, attr=True, test_geo=False, test_attr=False, show=False):
    if attr:
        config = ' --convertPlyColourspace=1'
    else:
        config = ''
    subp = subprocess.Popen(rootdir + '/tmc3 --mode=1' + config + \
                            ' --compressedStreamPath=' + bin_dir + \
                            ' --reconstructedDataPath=' + rec_dir + \
                            ' --outputBinaryPly=0',
                            shell=True, stdout=subprocess.PIPE)
    # headers
    if test_geo: headers = ['positions bitstream size', 'positions processing time (user)',
                            'Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    if test_attr: headers += ['colors bitstream size', 'colors processing time (user)']
    headers = []
    results = {}

    for _, key in enumerate(headers):
        results[key] = 0

    c = subp.stdout.readline()
    while c:
        if show: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] += value
        c = subp.stdout.readline()

    return results