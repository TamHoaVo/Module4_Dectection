import cv2
import numpy as np

#define the function 
def segment_animal(img_path, output_path=None):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #load into grayscale for brightness value
    if img is None:
        raise FileNotFoundError("Image not found")

    img_eq = cv2.equalizeHist(img) #contrast enhancement improve separation between dog and background

    #Slight blur helps remove noise
    blurred = cv2.GaussianBlur(img_eq, (5,5), 0)

    #Threshold convert to binary
    _, thresh = cv2.threshold(
        blurred, 175, 255, cv2.THRESH_BINARY #175 was the best to separate the hottest region(dog) from the rest of the image
    )

    #Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) #define shape for dilation/erosion, ellipse(natural smoothing), 7x7 (moderate)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3) #connect white region, fit hole, smooth boundary, repeat closing by 3(best fit)

    #Find contours
    #Only outer boundaries, Ignore internal holes
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,  
        cv2.CHAIN_APPROX_SIMPLE
    )

    #Remove very small contours to reduce noise blobs
    min_area = 3000
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    #Keep contour closest to image center (likely the dog)
    h, w = img.shape
    center = np.array([w//2, h//2])

    best_contour = None
    min_dist = float('inf')

    for c in contours:
        #compute image moment 
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        #compute centroid
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        #compute distance to image center 
        dist = np.linalg.norm(np.array([cx, cy]) - center) #Euclidean distance
        #keep the closest contours
        if dist < min_dist:
            min_dist = dist
            best_contour = c

    #Draw result
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, [best_contour], -1, (0,255,0), 2)
    
    if output_path:
        cv2.imwrite(output_path, output)

    return output, cleaned, contours

seg_image, seg_mask, seg_contours = segment_animal("dog.jpg", "output_boundary.png")
print(f"Detected {len(seg_contours)} contour(s)")
