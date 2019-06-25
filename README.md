# Depth estimation research papers

# Table of Contents
1. [Monocular depth estimation](#monocular)
2. [Stereo depth estimation](#stereo)
3. [Multi-view depth estimation](#mvs)


## Monocular depth estimation<a name="monocular"></a>
<a href="https://arxiv.org/pdf/1901.02571.pdf">Neural RGB→D Sensing: Depth and Uncertainty from a Video Camera</a>

<a href="http://fastdepth.mit.edu/">FastDepth: Fast Monocular Depth Estimation on Embedded Systems (ICRA 2019)</a>


## Stereo depth estimation<a name="stereo"></a>

<a href = "https://github.com/JiaRenChang/PSMNet">PSMNet: Pyramid Stereo Matching Network(CVPR 2018)</a>
* Spatial Pyramid Pooling(<a href = "https://arxiv.org/pdf/1406.4729.pdf">SPP</a>) Module
    + aims to incorporate context information by learning the relationship between an object and its sub-region.
* 4D cost volume
    + concat left and right SPP feature maps across each disparity level (H, W, D, F)
* Stacked hour-glass architecture for cost volume regularization
    + repeat top-down/bottom-up processing with intermediate supervision
    
 <a href = "https://arxiv.org/pdf/1903.04025.pdf">GwcNet: Group-wise Correlation Stereo Network(CVPR 2019)</a>
 * Construct cost volume by group-wise correlation
    + full correlation(DispNetC) lose information because only produces a single-channel correlation map for each disparity level
    + Divide channel into multiple groups, split features along channel dimension
    + left ith left group is cross-correlated with ith right group over all disparity levels(compute inner product as DispNetC)
    + packed correlations into matching cost volume (D/4, H/4, W/4, N_g)

* Modified the stacked 3D hourglass networks 
    + add extra output module and the extra loss lead to better features at lower layers
    + remove residual connection between output modules.
    + connections within each hourglass are added 1×1×1 3D conv
    
 <a href = "http://openaccess.thecvf.com/content_CVPR_2019/papers/Nie_Multi-Level_Context_Ultra-Aggregation_for_Stereo_Matching_CVPR_2019_paper.pdf">Multi-Level Context Ultra-Aggregation for Stereo Matching(CVPR 2019)</a>
 * Formulate two aggregation schemes(<a href = "https://arxiv.org/pdf/1608.06993.pdf">DenseNet</a>, <a href = "https://arxiv.org/pdf/1707.06484.pdf">DLA</a>) with Higher Order RNNS.
    + DenseNets cannoy merge features across scales and resolutions.
    + the fusion in DLA only refers to the intra-level combination.
 * Intra-level combination (divide into two groups according to the size of feature maps(1/2 or 1/4), fuse features in each group)
    + use 1×1 conv to match with each other, integrated by element-wise summation and pre-activated
 * Inter-level combination
    + an independent child module, firstly avg pooling to reduce the size by half(1/4), same architecture with the 1st group(1/2)
    + obtain large receptive fields at shallow stages
 * EMCUA
    + firstly train the model that MCUA is applied on the matching cost computation in PSMNet (2D-CNNs after SPP??)
    + secondly train EMCUA where a residual module is added at the end of MCUA
    
 <a href = "https://arxiv.org/pdf/1904.06587.pdf">GA-Net: Guided Aggregation Net for End-to-end Stereo Matching(CVPR 2019)</a>
 * Semi-global guided aggregation(SGA) layer
    + aims to solve occluded regions or large textureless/reflective regions
    + differentiable approximation of semi-global matching (<a href="https://core.ac.uk/download/pdf/11134866.pdf">SGM</a>), which aggregates matching cost iteratively in four directions.
    + replace min selection with a weighted sum, internal min changed to max(aims max the probabilities at the ground truth depths instead of min the matching costs), keeps the best from only one direction
    + The SGA layer are much faster and more effective than 3D convolutions.
   
 * Local guided aggregation(LGA) layer
    + aims to refine the thin structures and object edges which may be blured by down-sampling and up-sampling easily
    + compare to traditional <a href="http://wwwpub.zih.tu-dresden.de/~cvweb/publications/papers/2012/FastCost-VolumeFiltering.pdf">cost filter</a>, it aggregates with a K×K×3 weight matrix in a K×K local region for each pixel.
    
 <a href = "https://arxiv.org/pdf/1905.09265.pdf">Bridging Stereo Matching and Optical Flow via Spatiotemporal Correspondence(CVPR 2019)</a>
    
## Multi-view depth estimation<a name="mvs"></a>
<a href = "https://arxiv.org/pdf/1904.08103.pdf">Multi-Scale Geometric Consistency Guided Multi-View Stereo(CVPR 2019)</a>
