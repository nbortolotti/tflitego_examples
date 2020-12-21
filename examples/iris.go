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
	model, err := tflitego.NewTFLiteModelFromFile("iris_lite.tflite")
	if err != nil {
		if model == nil {
			log.Fatal("cannot load model")
		}
	}
	defer model.Delete()

	options, err := tflitego.NewInterpreterOptions()
	if err != nil {
		options.SetNumThread(4)
		defer options.Delete()
	}

	interpreter, err := tflitego.NewInterpreter(model, options)
	if err != nil {
		if interpreter == nil {
			log.Println("cannot create interpreter")
			return
		}
		defer interpreter.Delete()
	}

	status := interpreter.AllocateTensors()
	if status != tflitego.TfLiteOk {
		log.Println("allocate Tensors failed")
		return
	}

	newspecie := []float32{7.9, 3.8, 6.4, 2.0}
	input, err := interpreter.GetInputTensor(0)
	input.SetFloat32(newspecie)

	status = interpreter.Invoke()
	if status != tflitego.TfLiteOk {
		log.Println("invoke interpreter failed")
		return
	}

	output := interpreter.GetOutputTensor(0)
	out := output.OperateFloat32()
	fmt.Println(topSpecie(out))

}
