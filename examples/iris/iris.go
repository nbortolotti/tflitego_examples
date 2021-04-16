package main

import (
	"fmt"
	"log"

	"github.com/nbortolotti/tflitego"
)

func topSpecie(results []float32) string {
	pos := 0
	var max = results[0]
	for i := range results {
		if results[i] > max {
			max = results[i]
			pos = i
		}
	}

	specie := ""
	switch position := pos; {
	case position == 0:
		specie = "Setosa"
	case position == 1:
		specie = "Versicolor"
	case position == 2:
		specie = "Virginica"
	}

	return specie

}

func main() {
	model, err := tflite.NewModelFromFile("iris_lite.tflite")
	defer model.Delete()
	if err != nil {
		if model == nil {
			log.Fatal("cannot load model")
		}
	}

	options, err := tflite.NewInterpreterOptions()
	defer options.Delete()
	if err != nil {
		log.Fatal("cannot initialize interpreter options", err)
	}
	options.SetNumThread(4)

	interpreter, err := tflite.NewInterpreter(model, options)
	defer interpreter.Delete()
	if err != nil {
		log.Fatal("cannot create interpreter", err)
	}

	status := interpreter.AllocateTensors()
	if status != tflite.StatusOk {
		log.Println("allocate Tensors failed")
	}

	newspecie := []float32{7.9, 3.8, 6.4, 2.0}
	input, err := interpreter.GetInputTensor(0)
	if err != nil {
		log.Fatal("cannot get input tensor", err)
	}
	input.SetFloat32(newspecie)

	status = interpreter.Invoke()
	if status != tflite.StatusOk {
		log.Println("invoke interpreter failed")
	}

	output,err := interpreter.GetOutputTensor(0)
	if err != nil {
		log.Fatal("cannot get output tensor", err)
	}
	out := output.OperateFloat32()
	fmt.Println(topSpecie(out))

}
