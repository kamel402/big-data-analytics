# FaceRay

This project is about Facial Recognition with the distributed compute framework called Ray.

------

## Prerequisites
This project uses Anaconda as the package manager with these versions of packages.

Python: 3.9.10

face_recognition: 1.3.0

opencv: 4.5.5.64

ray: 1.12.0

## Run
You can run the code using the following command:
```
python recognize_faces.py --encodings encodings.pickle --image ben_afflek.jpg 
```

## How It Works
The `recognize_faces.py` code has two functions, recognize and draw.

recognize function is a remote function which takes the image and 128-d face embeddings. this function will be executed 
asynchronously on separate Python workers using Ray and return the promises. 

Then the result of promises can be retrieved with ``ray.get`` which returns the bounding boxes of recognized faces and associated names.

Finally draw function will draw bounding boxes and names on the image.