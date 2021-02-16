![Subreddit subscribers](https://img.shields.io/reddit/subreddit-subscribers/tflitego?style=social)

# Examples for tflitego


# how to use

```shell script
go get github.com/nbortolotti/tflitego
```

## Examples

### Iris Model
This example propose an inference using tflitego.

included: 
* iris.go: all the example code to implement an inference using tflitego
* iris_lite.tflite: tflite generic model to run the example. 

### Image Categorization
This example propose inferences using images and with the objective to provide categorization or labeling.

included: 
* image.go:
* image_test.go
* images: folder with image placerholders.
* models: folder with tf models.
  * mobilenet_v1_1.0_224_quant.tflite
  * mobilenet_v2_1.0_224_quant.tflite
  * labels_mobilenet_224.txt

Note: ioncluded a **flag** to customize an image capture into Raspberry Pi device. --rasp=true. This option provide th eoportunity to use a Raspberry Pi Camera Module to provide an interative demostration of image categorization. 

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
