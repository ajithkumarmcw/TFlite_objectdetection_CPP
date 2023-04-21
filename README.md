# TFlite_objectdetection_CPP

**Prerequistes**

    - tflite installation
    - opencv installation

**Build script**

I ran this inside /~/tensorflow-2.4.2/tensorflow/lite/examples/ path
```
mkdir build
cd build
cmake ..
cmake --build . -j
```

Issue: boxes, classes, labels are not correct they are zeros always