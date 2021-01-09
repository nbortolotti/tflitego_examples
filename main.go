package main

import (
	"fmt"

	tflite "github.com/nbortolotti/tflitego"
)

func main() {
	checkTFVersion()
}

func checkTFVersion() {

	version, err := tflite.Version()
	if err != nil {
		fmt.Printf("Error with TF lite version: %s", err)
	}
	fmt.Printf("TF lite version: %s", version)
	fmt.Println()
}
