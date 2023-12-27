# **Lane Detection**

## TO-DO
1. ~~Perspective transformation~~
2. ~~Change the order of the preprocessing steps to be (Crop, Perspective Transformation - Edge detection (Sobel or canny), Morphological operations)~~
3. ~~Integrate Hough~~ and sliding window algorithm
4. Check real-time performance
5. Test on different road videos
6. **UI**

## Problems
1. Accuracy of Hough is low, also it is slow
2. 

# **GUI**
## TO-DO
1. Integrate function (upload) with uploading the video
2. Function start - starts lane detection and stops it
3. Function pause - pause processing (can be deleted)
4. Video frames:
    1. Frame for the main lane detection video
    2. Frame for modifying PT points
    3. Frame for Transformed image (PT + morphology + binary)
5. Buttons (Hough / sliding window) : depending on which algorithm will we choose to work with
6. Sliders: modify PT points **Before or Due Processing**