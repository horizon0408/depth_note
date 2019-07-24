# Depth estimation research papers
paper notes, keep updating......

# Table of Contents
1. [Monocular depth estimation](#monocular)
2. [Stereo depth estimation](#stereo)
3. [Multi-view depth estimation](#mvs)


## Monocular depth estimation<a name="monocular"></a>
<a href="http://fastdepth.mit.edu/">FastDepth: Fast Monocular Depth Estimation on Embedded Systems (ICRA 2019)</a>


## Stereo depth estimation<a name="stereo"></a>

<a href = "http://vision.middlebury.edu/stereo/taxonomy-IJCV.pdf">A Taxonomy and Evaluation of Dense Two-Frame
Stereo Correspondence Algorithms(2002)</a>
* Traditional stereo methods generally perform 4 steps: matching cost computation; cost aggregation; disparity computation / optimization; disparity refinement.

<a href = "https://arxiv.org/pdf/1703.04309.pdf">GC-Net: End-to-End Learning of Geometry and Context for Deep Stereo Regression (ICCV 2017)</a>
* cost volume
    + not simply concatenate left and right features, but concat across each disparity level (H, W, maxD+1, F)
    + use distance metric restricts the network to only learn relative representations between features, and cannot carry absolute feature representations through to cost volume.
* use 3D convolutions to regularize the cost volume over height×width×disparity dimensions, get final regularized cost volume with size H×W×D.
* Differentiable soft argmin
    + traditional argmin's results are discrete, no sub-pixel estimates and not differentiable.
    + convert cost volume to probability volume, firstly take the negative value and then use softmax
    + take the sum of each disparity, weighted by its normalized probability
    + rely on network’s regularization to produce probability distribution which is predominantly unimodal
    



<a href = "https://github.com/JiaRenChang/PSMNet">PSMNet: Pyramid Stereo Matching Network(CVPR 2018)</a>
* Spatial Pyramid Pooling(<a href = "https://arxiv.org/pdf/1406.4729.pdf">SPP</a>) Module
    + aims to incorporate context information by learning the relationship between an object and its sub-region.
* 4D cost volume
    + concat left and right SPP feature maps across each disparity level (H, W, D, F)
* Stacked hour-glass architecture for cost volume regularization
    + repeat top-down/bottom-up processing with intermediate supervision
    
Code details notes:
I trained ScenesFlow 10 epochs with batch size = 4(A pair of images in size 256x512 consumed about 4GB GPU memory.), the training takes 24 hours; tried finetune with KITTI2015 300 epochs with batch size = 4, there are 160 training pairs so each epochs have 40 iters, which takes 4.44 hours.

<details>
 <summary><a href = "https://arxiv.org/pdf/1712.01039.pdf">Learning for Disparity Estimation through Feature Constancy (CVPR 2018)</a> </summary>
    
*  iResNet (iterative residual prediction network), incorporate all steps into a single network
*  use feature consistency to identify the correctness of the initial disparity and then refine
*  refined disparity map considered as a new initial map，repeated until the improvement is small
*  implemented in CAFFE (https://github.com/leonzfa/iResNet)
</details>


<details>
<summary><a href = "https://arxiv.org/pdf/1803.06641.pdf">Zoom and Learn: Generalizing Deep Stereo Matching to Novel Domains (CVPR 2018)</a> </summary>
Proposed a self-adaptation method to generalize a pre-trained deep stereo model to novel scenes 

* Scale diversity
    + passing an up-sampled stereo pair then down-sampling the result lead to more high-frequency details, but the performance won't keep improving with the increase of the up-sample rate.
    + up-sampling enable matching with sub-pixel accuracy, more details are taken into consideration
    + meanwhile, finer-scale input means smaller receptive filed, which leads to lack of non-local info
* <a href = "https://arxiv.org/pdf/1604.07948.pdf">graph laplacian regularization</a>
    + an adaptive metric as smoothness term
* iterative regularization
    + given pretrained model and a set combine both synthetic dataset w/ GT and real pairs wo/ GT.
    + create 'gt' for real pairs by zooming, minimize the difference between current prediction and fine-grain prediction.
* daily scenes from smartphones
    + 1900 pairs for training, 320 for validation, 320 for testing. collected resolution is 768×768.
    + the disparity is small
</details>
    

    

<details>
<summary><a href = "https://arxiv.org/pdf/1903.04025.pdf">GwcNet: Group-wise Correlation Stereo Network(CVPR 2019)</a></summary>
    
* Construct cost volume by group-wise correlation
    + full correlation(DispNetC) lose information because only produces a single-channel correlation map for each disparity level
    + Divide channel into multiple groups, split features along channel dimension
    + left ith left group is cross-correlated with ith right group over all disparity levels(compute inner product as DispNetC)
    + packed correlations into matching cost volume (D/4, H/4, W/4, N_g)

* Modified the stacked 3D hourglass networks 
    + add extra output module and the extra loss lead to better features at lower layers
    + remove residual connection between output modules.
    + connections within each hourglass are added 1×1×1 3D conv
 </details>
   
 <details>
 <summary><a href = "http://openaccess.thecvf.com/content_CVPR_2019/papers/Nie_Multi-Level_Context_Ultra-Aggregation_for_Stereo_Matching_CVPR_2019_paper.pdf">Multi-Level Context Ultra-Aggregation for Stereo Matching(CVPR 2019)</a> </summary>
  
 * Formulate two aggregation schemes(<a href = "https://arxiv.org/pdf/1608.06993.pdf">DenseNet</a>, <a href = "https://arxiv.org/pdf/1707.06484.pdf">DLA</a>) with Higher Order RNNS.
    + DenseNets cannot merge features across scales and resolutions.
    + the fusion in DLA only refers to the intra-level combination.
 * Intra-level combination (divide into two groups according to the size of feature maps(1/2 or 1/4), fuse features in each group)
    + use 1×1 conv to match with each other, integrated by element-wise summation and pre-activated
 * Inter-level combination
    + an independent child module, firstly avg pooling to reduce the size by half(1/4), same architecture with the 1st group(1/2)
    + obtain large receptive fields at shallow stages
 * EMCUA
    + firstly train the model that MCUA is applied on the matching cost computation in PSMNet (2D-CNNs after SPP??)
    + secondly train EMCUA where a residual module is added at the end of MCUA
 </details>
 
<details>
<summary> <a href = "https://arxiv.org/pdf/1904.06587.pdf">GA-Net: Guided Aggregation Net for End-to-end Stereo Matching(CVPR 2019)</a></summary>
    
 * Semi-global guided aggregation(SGA) layer
    + aims to solve occluded regions or large textureless/reflective regions
    + differentiable approximation of semi-global matching (<a href="https://core.ac.uk/download/pdf/11134866.pdf">SGM</a>), which aggregates matching cost iteratively in four directions.
    + replace min selection with a weighted sum, internal min changed to max(aims max the probabilities at the ground truth depths instead of min the matching costs), keeps the best from only one direction
    + The SGA layer are much faster and more effective than 3D convolutions.
   
 * Local guided aggregation(LGA) layer
    + aims to refine the thin structures and object edges which may be blured by down-sampling and up-sampling easily
    + compare to traditional <a href="http://wwwpub.zih.tu-dresden.de/~cvweb/publications/papers/2012/FastCost-VolumeFiltering.pdf">cost filter</a>, it aggregates with a K×K×3 weight matrix in a K×K local region for each pixel.
</details>

<details>
<summary> <a href = "https://arxiv.org/pdf/1905.09265.pdf">Bridging Stereo Matching and Optical Flow via Spatiotemporal Correspondence(CVPR 2019)</a> </summary>
    
 * learn joint representations for tasks that are highly-related unsupervisedly with given stereo videos
    + share a single network for both flow estimation and stereo matching
 * forward-backward consistency check to find occluded regions for optical flow
 * 2-Warp consistency loss
    + warp image twice by both optical flow and stereo disparity
    + training in unsupervised setting, no gt optical flow, disparity map and camera poses provided.
    
Code notes: Looks rely on CUDA9.2: after download cuda9.2 toolkit and export LD_LIBRARY_PATH="/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH" export PATH="/usr/local/cuda-9.2/bin:$PATH" it works.
 
 
</details>

<details>
<summary> <a href="https://arxiv.org/pdf/1904.02251.pdf">StereoDRNet: Dilated Residual Stereo Net (CVPR 2019)</a></summary>

 * 3D Dilated Convolution in Cost Filtering
    + combine information fetched from varing receptive fields
 * Disparity refinement
    + warp right image to left view via D_r (photometric consistency)
    + warp right disparity D_r to left view via left disparity D_l (geometric consistency)
    + use error maps as parts of input of refinement network rather than as loss function.
 * <a href="https://arxiv.org/pdf/1804.06242.pdf">Vortex Pooling</a> better than SPP
 </details>
 
 <a href = "http://openaccess.thecvf.com/content_CVPR_2019/papers/Poggi_Guided_Stereo_Matching_CVPR_2019_paper.pdf">Guided Stereo Matching (CVPR 2019)</a> <a href="https://github.com/mattpoggi/guided-stereo">[demo code only]</a>
 * use external sparse(< 5%) depth cues
    + to simulate the cues, randomly sample pixels from the ground truth disparity maps for both training and testing
 * feature enhancement 
    + given a disparity value k, enhance the k-th channel output of a correlation layer or the k-th slice of a 4D volume.
    + to avoid replace a lot zero values, use a Gaussian function.
    + the Gaussian modulation applied after concatenating L/R features (2F, D, H, W)
   

 
## Multi-view depth estimation<a name="mvs"></a>
<a href="https://arxiv.org/pdf/1901.02571.pdf">Neural RGB→D Sensing: Depth and Uncertainty from a Video Camera (CVPR 2019)</a>  <a href = "https://github.com/NVlabs/neuralrgbd">[code]</a>
* use D-Net to learn the depth probability volume (DPV)
    + pre-define dmin, dmax and neighbour window size to learn DPV
    + warps the features from neighbour frames to the reference frame and compute a cost volume (L1/L2).
    + the confidence maps can be obtained from DPV
* Apply Bayesian filter to integrate DPV over time
    + warp current DPV to 'predict' the DPV at t+1
    + to prevent wrong information propagate but also encourage correct information to be integrated, use K-Net to change the weight of 'prediction' adaptively
* R-Net
    + upsample and refine the DPV to original resolution (1/4 before)

## others
<a href = "http://www.cvlibs.net/publications/Janai2018ECCV.pdf"> Unsupervised Learning of Multi-Frame Optical Flow with Occlusions (ECCV 2018)</a>
* three-frame temporal window
    + consider both past and future frames
* occlusion estimation
    + consider three occlusion cases: visible in all frame, only occlusion in past frame, only occlusion is future frame
    + enforce the norm of occlusion variable for each pixel to be 1 with softmax function
    + use it to weight the contribution of future and past estimates
* Two separate cost volumes
    + one for past and one for future, to detect occlusions
    + stack two cost volume as the input for all separate decoders
* Two flow decoders
    + encourage constant velocity as a soft constraint
    + Under the constant velocity assumption, the future and past flow should be equal in length but differ in direction.
    
