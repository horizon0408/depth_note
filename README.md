# Depth estimation research papers
12.2.2019: add iccv19 to-read list

# Table of Contents
1. [Monocular depth estimation](#monocular)
2. [Stereo depth estimation](#stereo)
3. [Multi-view depth estimation](#mvs)


## Monocular depth estimation<a name="monocular"></a>
<a href="http://fastdepth.mit.edu/">FastDepth: Fast Monocular Depth Estimation on Embedded Systems (ICRA 2019)</a>
<details>
<summary><a href="https://arxiv.org/pdf/1908.03127.pdf">Enhancing self-supervised monocular depth estimation with traditional visual odometry (3DV 2019)</a></summary>
    
* VO to obtained sparse 3D points
    + reproject 3D points onto both L/R camera planes to get sparse disparity map.
    + deploy two VO methods for training that exploit stereo and monocular sequences respectively
    + use ORB-SLAM2 for stereo VO (correct scale), Zenuity's pipeline for monocular VO(need scale recovery)
    
* sparsity-invariant autoencoder (also check paper <a href="https://arxiv.org/pdf/1708.06500.pdf">Sparsity Invariant CNNs</a>)
    + sparse disparity map (SD) to denser priors (DD) for further estimation  
    + final prediction d is a sum of DD'(from skip module) and D(from depth estimator)
  
* self-supervised loss
    + use stereo image only for training (symmetric scheme), the inference use monocular input
    + apperance matching loss,  disparity smoothness loss, left-right consistency loss
    + occlusion loss: minimize the sum of all disparities
    + inner loss: enforce DD to be consistent with SD (use L1 here)
    + outer loss: to preserve the info from VO, enforce final prediction d to be consistent with SD
</details>


<a href="https://research.dshin.org/iccv19/multi-layer-depth/">3D Scene Reconstruction with Multi-layer Depth and Epipolar Transformers (ICCV 2019)</a>


<a href="https://arxiv.org/pdf/1908.09521.pdf">Object-Driven Multi-Layer Scene Decomposition From a Single Image (ICCV 2019)</a>

<a href="http://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Spatial_Correspondence_With_Generative_Adversarial_Network_Learning_Depth_From_Monocular_ICCV_2019_paper.pdf">Spatial Correspondence with Generative Adversarial Network:
Learning Depth from Monocular Videos (ICCV 2019)</a>


<a href="https://arxiv.org/pdf/1909.09051.pdf">Self-Supervised Monocular Depth Hints</a>

<a href="https://arxiv.org/pdf/1806.01260.pdf">Digging Into Self-Supervised Monocular Depth Estimation</a>

## Stereo depth estimation<a name="stereo"></a>

<a href="http://www.nlpr.ia.ac.cn/2011papers/gjhy/gh75.pdf">On Building an Accurate Stereo Matching System on Graphics Hardware</a>


<a href = "http://vision.middlebury.edu/stereo/taxonomy-IJCV.pdf">A Taxonomy and Evaluation of Dense Two-Frame
Stereo Correspondence Algorithms(2002)</a>
* Traditional stereo methods generally perform 4 steps: matching cost computation; cost aggregation; disparity computation / optimization; disparity refinement.

<a href = "https://arxiv.org/pdf/1512.02134.pdf">(DispNet)A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation</a>
* follow the archietecture of <a href="https://arxiv.org/pdf/1504.06852.pdf">FlowNet</a>
* DispNetCorr
    + two images processed separately to conv2 and resulting features are correlated horizontally(1D).
    + compute the dot product, lead to single-channel correlation map for each disparity level
    

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
<summary><a href = "https://arxiv.org/pdf/1807.08865.pdf">StereoNet: Guided Hierarchical Refinement for Real-Time Edge-Aware Depth Prediction (ECCV 2018)</a></summary>
    
* low resolution cost volume
    + low resolution(1/8 or 1/16) lead to bigger receptive filed and compact feature vectors
    + most time is spent with higher resolutions while most performance gain from lower resolutions.
* edge-aware hierarchical refinement
    + upsample the disparity bilinearly and concatenate with color
    + output is a 1D residual to be added to previous coarse prediction
* real-time(60 fps)
</details>

<details>
<summary><a href="https://arxiv.org/pdf/1807.06009.pdf">ActiveStereoNet: End-to-End Self-Supervised Learning for Active Stereo Systems (ECCV 2018)</a></summary>
* active stereo
    + a textured is projected into the scene with an IR(红外) projector, and cameras are augmented to perceive IR and visible spectra.
    
* photometric loss is poor
    + brighter pixels are closer(passive stereo won't suffer as the intensity and disparity won't have correlation)
    + brighter pixels are likely to have bigger residual than dark pixels.
    
* Weighted Local Contrast Normalization(LCN)
    + remove the dependency between intensity and disparity, give better residual in occluded region
    + compute local mean and std in 9×9 patch to normalize the intensity
    + before re-weight, suffer in low texture regions(have small std that can amplify residual)
    + re-weight using std estimated on the reference image

* adaptive support weight cost aggregation
    + traditional adaptive support sceheme, effective but slow
    + only integrate in training with 32×32 windlow
    
* invalidation network
    + left-right check occlusion mask with enforcing regularization on the number of valid pixel
    + invalidation network also produce mask, which make inference faster
    
* dataset
    + real dataset: collected from Intel Realsense D435 camera(10000/100 train/test), the camera have IR light source.
    + synthetic dataset: rendered by Blender(10000/1200)
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
   
 <a href = "https://arxiv.org/pdf/1810.05424.pdf">Real-time self-adaptive deep stereo (CVPR)</a> <a href='https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo'>[code]</a>
 * a fast modular architecture
    + at the lowest resoltion (F6), forward features from left to right into correlation layer(DispNetC here), decoder D6 get disparity map at lowest resolution.
    + upsample D6 to level 5, used for warping right features to left before computing correlation.
    + then the decoder D_k is to refine and correct the up-scaled disparity prediction
    + the correlation scores computed between original left and aligned right features guides the refinement process.
    
 * modular adaption
    + model is always in training mode and continuously fine-tuning to the sensed environment
    + grouping layers at the same resolution into a single module
    + optimize module independently, compute loss with prediction y_i and excute shorter backprop only across Module i
    
 * Reward/punishment selection (sample)
    + when deploying, need to sample a portion (from [1, p]) of the network to optimize for each incoming pair
    + create a histogram with p bins and apply softmax to obtain probability distribution to sample
    + To update the histogram, compute noisy expected value L_exp according to previous loss(L_{t-1}, L_{t-2})
    + change the value of histogram according to L_{exp} - L_t (>0 means effect)
    + the loss is based on photometric consistency loss, combination of L1 and SSIM
    
  <a href = "https://arxiv.org/pdf/1909.05845.pdf">DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch(ICCV 19)</a> 
  
<a href = "https://arxiv.org/pdf/1807.11699.pdf">SegStereo: Exploiting Semantic Information for Disparity Estimation (ECCV18)</a> 
* Model Specification
    + shallow part of ResNet-50 model to extract image features
    + PSPNet-50 as segmentation net
    + The weights in the shallow part and segmentation network are fixed when training
    + Disparity encoder behind hybrid volume contains 12 residual blocks
* given shared representation, use segmentation network to compute semantic features for left/right respectively
* concatenate both transformed left features(to preserve details), correlated features, left semantic features as hybrid volume
* warped semantic consistency via semantic loss regularization
    + warp right semantic features to left based on predicted disparity map and use left segmentation GT to guide
    + propagates softmax loss back to disparity branch by warping
* framework is appliable for both supervised/unsupervised training
    + the unsupervised loss introduce a mask indicator to avoid outlier, setting a threshold for resulting photometric diff
    + chabonier function for spatial smoothness penalty 


<a href = "http://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Semantic_Stereo_Matching_With_Pyramid_Cost_Volumes_ICCV_2019_paper.pdf">Semantic Stereo Matching with Pyramid Cost Volumes (ICCV 19)</a> 
* pyramid cost volumes for both semantic and spatial info 
    + Unlike PSMNet use single cost volume with multiscale features, it construct multilevel cost volumes directly (btw the figure for spatial cost volume via spatial pooling is clear) (However, should lead to much higher complexity?)
    + semantic cost volume follows <a href="https://arxiv.org/pdf/1612.01105.pdf">PSPNet</a>
        + single, upsample feature maps to the same size and concatenate
* 3D multi-cost aggregation with hourglass and 3D feature fusion module(FFM)
    + for each spatial cost volume, firstly a hourglass then upsampled for following fusion
    + fuse 4D spatial cost volumes from low to high level in a recursive way 
    + FFM employ <a href="https://arxiv.org/pdf/1709.01507.pdf">SE-block structure</a>
* boundary loss
    + disparity discontinuity point is always on the semantic boundaries
    + compute intensity gradient for GT segmentation labels and predicted disparity (align edges)
* two-step training, train the segementation subnet firstly and then joint training the whole
    + For Scene Flow, have object-level segmentation GT, transform to segmentation labels
    + For KITTI2015/12, the semantic segmentation first trained with KITTI15 (have GT for left images)
    
<a href="https://arxiv.org/pdf/1910.00541.pdf">Real-Time Semantic Stereo Matching</a> 
* Both segmentation and disparity are fully computed only at the lowest resolution and progressively refined through the higher resolution residual stages(residual disparity), also applied in final refinement
    + by building cost volume at the lowest reso, dmax=12 is enough(correspond to 192 at full reso)
* Synergy Disparity Refinement
    + previous work(<a href="https://arxiv.org/pdf/1807.11699.pdf">SegStereo</a>) use concatenate the two embeddings into a hybrid volume
    + perform a cascade of residual concatenations between semantic class probabilities and disparity volumes
* since only GT instance segmentation in SceneFlow, initialize network on the CityScapes(disparity maps obtained via SGM, noisy)
        

<a href="https://arxiv.org/pdf/1904.09099.pdf">AMNet: Deep Atrous Multiscale Stereo Disparity Estimation Networks</a>
*  use an AM module after the D-ResNet backbone to form the feature extractor (similar purpose like SPP)
    + a set of 3×3 dilated convolutions with increasing dilation factors (1,2,2,4,4,...,k/2,k/2,k), two 1×1 conv with dilation factor one are added at the end
    + dilated convs provide denser features than pooling 
    + increase the receptive field and get denser multiscale contextual w/o losing the spatial resolution
    
* Extended Cost Volume Aggregation
    + unlike others only use single volume, it concatenate three different cost volumes (kind of encode several distance metric res here), final size will be H×W×(D+1)×4C 
    + disparitylevel feature concatenation
        + just concatenate L/R features lile GC-Net, PSMNet, get volume of size H×W×(D+1)×2C
    + Disparity-level feature distance
        + compute the point-wise absolute difference between L/R features at all disparity levels, get volume of size H×W×(D+1)×C
    + Disparity-level depthwise correlation
        + compute scalar product(like DispNetC) between L/R patches, get volume of size H×W×(D+1)×1
        + to make the size comparable(now the channel is only 1 here), implement depthwise correlation, which means compute the patch correlation for each channel (t is 0 in practical, which means the size of patch is 1×1 actually, so it's just two number's product for every channel), which finally get a volume of size H×W×(D+1)×C.
        
* stacked AM modules to aggregate ECV (3D convs here because of the size of ECV)



 
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


<a href="https://arxiv.org/pdf/1908.04422.pdf">Point-Based Multi-View Stereo Network</a>

<a href="https://arxiv.org/pdf/1908.03706.pdf">Exploiting temporal consistency for real-time video depth estimation</a>

## others
<details>
<summary> <a href = "http://www.cvlibs.net/publications/Janai2018ECCV.pdf"> Unsupervised Learning of Multi-Frame Optical Flow with Occlusions (ECCV 2018)</a> </summary>
    
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
</details>

<details>
<summary><a href="https://arxiv.org/pdf/1910.12361.pdf">SENSE: a Shared Encoder Network for Scene-flow Estimation </a></summary>
    
* shared encoder for 4 related tasks: optical flow/ disparity from stereo/ occlusion/ semantic segementation
    + inputsL two stereo images pairs (no camera pose needed)
    + build on top pf PWC-Net, encoder extracts features at different hierarchies, reduce pyramid level from 6 to 5 
    + decoder for disparity
        + Pyramid Pooling Module(PPM) to aggregate learned features
        + add a hourglass, take twice up-sampled disparity, feature map, warped feature map to predict residual disparity
    + decoder for segmentation, use <a href="https://arxiv.org/pdf/1807.10221.pdf">UPerNet</a>
    + occlusion,add sibling branches to flow/disparity decoders to do pixel-wise binary classification
* semi-supervised
    + distillation loss
        + pseudo GT for occlusion/segmentation provided by pre-trained model on other data 
    + self-supervision loss
        + corresponding pixels have photometric consistency and semantic consistency(similar posterior probability)
        + add regularization terms for occlusion study
* rigidity-based warped disparity refinement
    + select pixel as static by removing semantic level vehicle/pedestrian/cyclist/sky
    + estimate rigid flow induced by camera motion
        + use estimated transformation(motion) to estimate rigid flow which should be consistent with estimated flow in the background rigion
    + estimate warped second frame rigid disparity
        + use estimated transformation to get warped disparity of 2nd frame from 1st frame
        + then use the estimated forward flow to compute warped disparity of 2nd frame (suspect Eq.18 in Appendix B??)
</details>

<a href="https://papers.nips.cc/paper/6502-surge-surface-regularized-geometry-estimation-from-a-single-image.pdf">SURGE: Surface Regularized Geometry Estimation from a Single Image</a>
For single image depth estimation might depends on appearance information alone, so the surface geometry should help a lot here
* a fourstream CNN, predict depths, surface normals, and likelihoods of planar region and planar boundary 
* a DCRF integrate 4 predictions
    +  the field of variables to be optimized are depths and normals

<a href="https://arxiv.org/pdf/1804.06278.pdf">PlaneNet: Piece-wise Planar Reconstruction from a Single RGB Image</a>    
* piece-wise planar depthmap reconstruction requires a structured geometry representation
* directly produce a set of plane parameters and probabilistic plane segmentation masks 
    + Plane parameters, predict a fixed number of planar surfaces(K) for each scene, depth can be inferred from paramteres
        + don't know the number of planes, enable the corresponding probabilistic segmentation masks to be 0
        + order-agnostic loss function based on the Chamfer distance
    + Non-planar depthmap, regard as (K+1) th surface
    + Segmentation masks, probabilistic segmentation masks
        + joint train a <a href= "https://arxiv.org/pdf/1210.5644.pdf">DCRF</a> module

    

<a href="https://arxiv.org/pdf/1812.06264.pdf">Hierarchical Discrete Distribution Decomposition for Match Density Estimation</a>

<a href="https://arxiv.org/pdf/1710.01020.pdf">Learning Affinity via Spatial Propagation Networks (SPN)</a>
<a href="https://arxiv.org/pdf/1707.06484.pdf">Deep Layer Aggregation</a>
<a href="https://arxiv.org/pdf/1706.05587.pdf">Rethinking Atrous Convolution for Semantic Image Segmentation</a>
<a href="https://arxiv.org/pdf/1811.01791.pdf">Confidence Propagation through CNNs for Guided Sparse Depth Regression</a>
    
    
        
