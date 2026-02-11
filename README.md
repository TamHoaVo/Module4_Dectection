Thermal Animal Boundary Detection
This project implements a classical image processing pipeline to detect the boundary of an animal in a thermal image using OpenCV.

**Algo**
1. Enhances contrast
2. Applies intensity thresholding
3. Performs morphological cleanup
4. Detects connected components (blobs)
5. Selects the most central object
6. Extracts and draws its boundary
The output is the detected contour of the animal.

**Method Pipeline**
1. Grayscale Conversion
Thermal segmentation relies on intensity (temperature), not color.
2. Histogram Equalization
Improves contrast between hot animal regions and background.
3. Gaussian Blur
Reduces noise before thresholding.
4. Thresholding
Applies a fixed threshold (175) to isolate high-temperature pixels.
6. Morphological Closing
Dilation followed by erosion to:
Fill small holes
Connect nearby hot regions
Smooth boundaries
7. Contour Detection
Finds connected components (blobs) in the binary mask.
8. Contour Selection
Selects the contour closest to the image center (assumes subject is centered).

**How to Run**
Install: pip install opencv-python numpy
Run:
seg_image, seg_mask, seg_contours = segment_animal_classical(
    "dog.jpg",
    "output_boundary.png"
)










