# QR-code-detection
Method for isolating QR-code from image:
- Convert image to greyscale (NTSC formula)
- Normalize pixel intensity
- Highlight edges (sobel filter)
- Highlight large patches of white pixels(mean filter)
- Create binary image, remove non-white pixels (threshold filter)
- Dilate/expand image to fill holes(dilation filter)
- erosion filter to shrink edges of remaining "shapes"
- On processed image run Connected components algorithm
- pyplot to display final result

