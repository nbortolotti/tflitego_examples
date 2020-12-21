package main

import (
	"fmt"

	"github.com/nbortolotti/tflitego"
)

func main() {
	checkTFVersion()
}

func checkTFVersion() {

	version, err := tflitego.TFVersion()
	if err != nil {
		fmt.Printf("Error with TF lite version: %s", err)
	}
	fmt.Printf("TF lite version: %s", version)
	fmt.Println()
}
