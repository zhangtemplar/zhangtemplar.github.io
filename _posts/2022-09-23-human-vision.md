---
layout: post
title: Human Vision Specification
tags:  pupil field-of-view fov human eyeball iris resolution focal-length vision eye
---
This document describes the specifications of typical human vision system: resolution 576 megapixels with eye movement or 324 megapixels at a single glint; angle of view is around 180°; light sensitivity is about ISO 800; dynamic range is 1 billion to 1 with adjustment or 10000 to 1 with a single glint; focal length at 22mm and aperture size at F/3.2.

# Dimensions

The image below shows antomoy of human's eyeall. The size of a human adult eye is approximately **24.2 mm (transverse) × 23.7 mm (sagittal) × 22.0–24.8 mm (axial)** with no significant difference between sexes and age groups. Pupil size changes according to environment and/or mental state, which could be 2~8mm in diameter.

![image-20220923140633554](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_23_14_06_33_image-20220923140633554.png)

# Resolution

![preview](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_23_13_50_42_v2-0bc0c0212183cffa585997368432dc55_r.jpg)

According to [ClarkVision.com](https://clarkvision.com/imagedetail/eye-resolution.html), human eye could distinguish object as tiny as 0.3 arc-minute. The eye is not a single frame snapshot camera. It is more like a video stream. The eye moves rapidly in small angular amounts and continually updates the image in one's brain to "paint" the detail. We also have two eyes, and our brains combine the signals to increase the resolution further. We also typically move our eyes around the scene to gather more information. Because of these factors, the eye plus brain assembles a higher resolution image than possible with the number of photoreceptors in the retina. So the megapixel equivalent numbers below refer to the spatial detail in an image that would be required to show what the human eye could see when you view a scene.

Based on the above data for the resolution of the human eye, let's try a "small" example first. Consider a view in front of you that is 90 degrees by 90 degrees, like looking through an open window at a scene. The number of pixels would be 90 degrees * 60 arc-minutes/degree * 1/0.3 * 90 * 60 * 1/0.3 = 324,000,000 pixels (324 megapixels). At any one moment, you actually do not perceive that many pixels, but your eye moves around the scene to see all the detail you want. But the human eye really sees a larger field of view, close to 180 degrees. Let's be conservative and use 120 degrees for the field of view. Then we would see 120 * 120 * 60 * 60 / (0.3 * 0.3) = 576 megapixels. The full angle of human vision would require even more megapixels. This kind of image detail requires A large format camera to record.

# **The Sensitivity of the Human Eye (ISO Equivalent)**

> At low light levels, the human eye integrates up to about 15 seconds (Blackwell, J. Opt. Society America, v 36, p624-643, 1946). The ISO changes with light level by increasing rhodopsin in the retina. This process takes a half hour our so to complete, and that assumes you haven't been exposed to bright sunlight during the day. Assuming you wear sunglasses and dark adapt well, You can see pretty faint stars away from a city. Based on that a reasonable estimate of the dark adapted eye can be done.
>
> In a test exposure I did with a Canon 10D and 5-inch aperture lens, the DSLR can record magnitude 14 stars in 12 seconds at ISO 400. You can see magnitude 14 stars in a few seconds with the same aperture lens. (Clark, R.N., Visual Astronomy of the Deep Sky, Cambridge U. Press and Sky Publishing, 355 pages, Cambridge, 1990.)
>
> So I would estimate the dark adapted eye to be about ISO 800.
>
> Note that at ISO 800 on a 10D, the gain is 2.7 electrons/pixel (reference: http://clarkvision.com/articles/digital.signal.to.noise) which would be similar to the eye being able to see a couple of photons for a detection.
>
> During the day, the eye is much less sensitive, over 600 times less (Middleton, Vision Through the Atmosphere, U. Toronto Press, Toronto, 1958), which would put the ISO equivalent at about 1.

# The Sensitivity of the Human Eye (ISO Equivalent)

> The Human eye is able to function in bright sunlight and view faint starlight, a range of more than 100 million to one. The Blackwell (1946) data covered a brightness range of 10 million and did not include intensities brighter than about the full Moon. The full range of adaptability is on the order of a billion to 1. But this is like saying a camera can function over a similar range by adjusting the ISO gain, aperture and exposure time.
>
> In any one view, the eye eye can see over a 10,000 range in contrast detection, but it depends on the scene brightness, with the range decreasing with lower contrast targets. The eye is a contrast detector, not an absolute detector like the sensor in a digital camera, thus the distinction. (See Figure 2.6 in Clark, 1990; Blackwell, 1946, and references therein). The range of the human eye is greater than any film or consumer digital camera.

# Focal Length

This chart shows the relationship of diagnoal angle of view vs focal length for 35mm (diagonal) image sensor.

![preview](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_23_13_52_31_v2-29ac4de04066f5dfb7bbad0c2852e76c_r.jpg)

> So this explains the commonly cited ~17mm focal length, but the correct value is ~22 mm focal length
>
> This then makes more sense for the f/ratio: with an aperture of 7 mm, the f/ratio = 22.3/7 = 3.2.

# Spectrum

Obviously human's vision is more sensitive to green light.

![The cones and rods in the eye dictate human's photopic response.](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_23_14_09_46_EyeResp1.png)
