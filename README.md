# Examples for tflitego


# how to use

```
go get github.com/nbortolotti/tflitego

```

## Examples

1. Iris model using tflitego: this example propose an inference using tflitego.

included: 
* iris.go: all the example code to implement an inference using tflitego
* iris_lite.tflite: tflite generic model to run the example. 

2. Image Categorization: this example propose inferences using images and with the objective to provide categorization or labeling.

included: 
* image.go:
* image_test.go
* images: folder with image placerholders.
* models: folder with tf models.
  * mobilenet_v1_1.0_224_quant.tflite
  * mobilenet_v2_1.0_224_quant.tflite
  * labels_mobilenet_224.txt

---
**Note:**

remember to navigate into examples folder.

---


```shell script
go build
```

```shell script
go run iris.go
```

<img src="https://storage.googleapis.com/tflitego/iris3.gif?raw=true" width="600px">
