---
layout: posts
title: 3-D scanning with water
---

Scholars from Sandong University have developed a new methods for 3D construction: $$water$$. The benefits of this method, compared with current optical based method is that, it is able to reconstruct the 3D strucutre of invisible parts, e.g., inner space, concave region.

![3D scanning using a dip scanner. The object is dipped using a robot arm in a bath of water (left), acquiring a dip transform. The quality of the reconstruction is improving as the number of dipping orientations is increased (from left to right)](http://irc.cs.sdu.edu.cn/3dshape/files/dip_1.png)
# Abstract

he paper presents a novel three-dimensional shape acquisition and reconstruction method based on the well-known Archimedes equality between fluid displacement and the submerged volume. By repeatedly dipping a shape in liquid in different orientations and measuring its volume displacement, we generate the dip transform: a novel volumetric shape representation that characterizes the object’s surface. The key feature of our method is that it employs fluid displacements as the shape sensor. Unlike optical sensors, the liquid has no line-of-sight requirements, it penetrates cavities and hidden parts of the object, as well as transparent and glossy materials, thus bypassing all visibility and optical limitations of conventional scanning devices. Our new scanning approach is implemented using a dipping robot arm and a bath of water, via which it measures the water elevation. We show results of reconstructing complex 3D shapes and evaluate the quality of the reconstruction with respect to the number of dips.

# Results

![3D dip reconstructions comparison. (a) Picture of the objects during the dipping (b) Profile picture of the printed objects (c) Structured light scanner reconstruction (d) Our 3D reconstruction using the dipping robot. Occluded parts of the body have no line-of-sight to the scanner sensor, while the dipping robot, using water, is able to reconstruct these hidden parts.
](http://irc.cs.sdu.edu.cn/3dshape/files/result_1.png)

![Video of actual operation](http://irc.cs.sdu.edu.cn/3dshape/files/dip_final.mp4)

# Links

  - [Website](http://irc.cs.sdu.edu.cn/3dshape/)
  - [Paper](http://irc.cs.sdu.edu.cn/3dshape/files/fermat_spirals.pdf)
  - [Presentation](http://irc.cs.sdu.edu.cn/3dshape/files/fermat_spirals.pptx)
