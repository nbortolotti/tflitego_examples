# Examples for tflitego


# how to use

```shell script
go get github.com/nbortolotti/tflitego
```

## Examples

1. Iris model using tflitego: this example propose an inference using tflitego.

included: 
* iris.go: all the example code to implement an inference using tflitego
* iris_lite.tflite: tflite generic model to run the example. 

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
