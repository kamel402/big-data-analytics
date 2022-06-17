# FaceRay

This project is about Facial Recognition with the distributed compute framework called Ray.

------

## Prerequisites
This project uses Anaconda as the package manager with these versions of packages.

Python: 3.9.12

face_recognition: 1.3.0

opencv: 4.5.5.64

ray: 1.13.0

## Run
You can run the code using the following command:
```
python recognize_faces.py --encodings encodings.pickle --video  dataset/4_faces.mp4 --resize 240 
```

## How It Works
The `recognize_faces.py` code process each frame of the video using these three functions:

1. detection function accepts the image, and return the bounding boxes and encodings of the faces.

2. recognize function is a Ray remote function which takes the encoding of a face and returns the name of detected face. this function will be executed 
asynchronously on separate Python workers using Ray and return the promises. Then the result of promises can be retrieved with ``ray.get`` which returns the bounding boxes of recognized faces and associated names.

3. Finally draw function will draw bounding boxes and names on the image.
