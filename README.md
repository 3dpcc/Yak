# **Neural Compression System for Point Cloud Video Streaming**

Point cloud video streaming is promising for immersive media applications, which urges the development of efficient compression methods. However, existing approaches either suffer from poor performance or lack effective coder control mechanisms, making them impractical for networked point cloud services, where bandwidth is often constrained and fluctuates over time. Therefore, this paper proposes a system-level solution -- a layered point cloud compressor, called Yak, to address these issues. Yak offers comprehensive support for both intra and inter-frame coding of geometry and attribute components in point cloud sequences. It consists of three layers: the Base Layer uses the standard G-PCC to encode a thumbnail counterpart downscaled from the input point cloud; the Enhancement Layer devises the end-to-end variational autoencoder to compress the original input conditioned on the base layer reconstruction, and the Dynamic Layer generates feature-space predictions as the temporal prior for conditional inter-frame coding. In addition, Yak devises the Content Analysis module to dynamically determine the optimal encoding parameters of each frame, by which bit budget is intelligently allocated for geometry and attribute components to maximize the overall rate-distortion (R-D) performance. Such accurate rate control relies on the parametric rate/distortion models whose parameters are initialized through one-pass template matching and frame-wise delta updating constrained by R-D optimization. Following standard evaluation guidelines, Yak has notably outperformed traditional rules-based methods such as MPEG G-PCC and V-PCC, as well as other learning-based approaches, while offering flexible networked adaption and affordable complexity.

## News

- 2025.11.21 The paper was accpeted by IEEE Transactions on Image Processing. (Junteng Zhang, Tong Chen, Dandan Ding, and Zhan Ma, "Neural Compression System for Point Cloud Video Streaming")

## Requirments

- python >=3.7


- cuda >= 10.2


- pytorch >= 1.7


- MinkowskiEngine 0.54


- pytorch3d 0.6.1


- Test dataset: [8iVFB & Owlii](https://box.nju.edu.cn/d/4a3745da59284f1eba81/)


- Pretrained Models: [Pretrained Models](https://box.nju.edu.cn/f/0d2c5e87fb604043b405/)

## Usage

### Testing
```shell
sudo chmod 777 tmc3 pc_error_d PCQM

# static geometry coding
python test_static.py --mode='geometry' --input_rootdir='../data/static/' --model_name='geometry_coding' --g_lambda_list 64. 128. 256. 512. 1024. 2048. 4096.

# static attribute coding
python test_static.py --mode='attribute' --input_rootdir='../data/static/' --model_name='attribute_coding' --a_lambda_list 64. 128. 256. 512. 1024. 1536. 2048.

# static geometry & attribute coding
python test_static.py --mode='joint' --input_rootdir='../data/static/' --model_name='joint_coding' --g_lambda_list 64. 128. 256. 512. 1024.  --a_lambda_list 256. 512. 1024. 1536. 2048. 

# dynamic geometry coding
python test_dynamic.py --mode='geometry' --input_rootdir='../data/dynamic/basketball_player_vox10' --model_name='inter_geometry_coding/basketball_player_vox10' --g_lambda 1024. --first 10000001 --count 10

# dynamic attribute coding
python test_dynamic.py --mode='attribute' --input_rootdir='../data/dynamic/basketball_player_vox10' --model_name='inter_attribute_coding/basketball_player_vox10' --a_lambda 2048. --first 10000001 --count 10

# dynamic geometry & attribute coding
python test_dynamic.py --mode='joint' --input_rootdir='../data/dynamic/basketball_player_vox10' --model_name='inter_joint_coding/basketball_player_vox10' --g_lambda 256. --a_lambda 512. --first 10000001 --count 100
```
The testing rusults can be found in `output`.

## Authors
These files are provided by Nanjing University Vision Lab. Please contact us (zhangjunteng@smail.nju.edu.cn) if you have any questions.
