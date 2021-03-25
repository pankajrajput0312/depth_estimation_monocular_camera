# Depth_Estimation_Using_Monocular_Camera

## About
Depth estimation is an important technique in computer vision for applications such as pose modelling, activity recognition, etc. Common methods of depth estimation include stereo vision which relies on generating a disparity map using images from two cameras to triangulate and estimate distances. To reduce the reliance on stereo vision systems, learning-based methods have been utilized to generate depth maps using RGB images from a single camera. In this paper, we present a novel method to predict depth data from RGB images by focussing on distant depth regions in the image. We propose a dual branched model for utilizing the semantic information from an image and use the data for improving the depth estimates using data sharing layers between the depth estimation branch and semantic segmentation branch. Experiments on standard datasets signal that the proposed approach achieves state of the art performance to estimate depth maps from RGB images from a monocular camera.

