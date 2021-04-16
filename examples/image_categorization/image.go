package main

import (
	"bufio"
	"flag"
	"fmt"
	"image"
	_ "image/png"
	"log"
	"os"
	"sort"

	"github.com/dhowden/raspicam"
	tflite "github.com/nbortolotti/tflitego"
	"github.com/nfnt/resize"
)

func main() {
	var modelPath, labelsPath, imagePath string
	var raspCapture bool
	flag.StringVar(&modelPath, "model", "models/mobilenet_v2_1.0_224_quant.tflite", "path to model file")
	flag.StringVar(&labelsPath, "label", "models/labels_mobilenet_224.txt", "path to label file")
	flag.StringVar(&imagePath, "image", "images/dog.png", "path to image file")
	flag.BoolVar(&raspCapture, "rasp", false, "useful when is neede to activate image capture in raspberry-pi device")
	flag.Parse()

	labels, err := getLabels(labelsPath)
	if err != nil {
		log.Println("cannot get labels: ", err)
	}

	model, err := tflite.NewModelFromFile(modelPath)
	if err != nil {
		log.Fatal("cannot create new model from file", err)
	}
	defer model.Delete()
	if err != nil {
		log.Fatal("cannot create new model from file", err)
	}

	options, err := tflite.NewInterpreterOptions()
	defer options.Delete()
	if err != nil {
		log.Fatal("cannot create new interpreter options", err)
	}
	options.SetNumThread(4)

	interpreter, err := tflite.NewInterpreter(model, options)
	if err != nil {
		log.Fatal("cannot create new interpreter", err)
	}
	defer interpreter.Delete()
	if err != nil {
		log.Fatal("cannot create new interpreter", err)
	}

	status := interpreter.AllocateTensors()
	if status != tflite.StatusOk {
		log.Println("allocate failed")
		return
	}

	input, err := interpreter.GetInputTensor(0)
	if err != nil {
		log.Fatal("cannot get input tensor:", err)
	}

	if raspCapture {
		imagePath, err = captureImage("images/temp.jpg")
		if err != nil {
			log.Println("cannot capture image: ", err)
		}
	}

	ibuffer, err := imageToBuffer(imagePath, input)
	if err != nil {
		log.Println("cannot apply image to bugger:", err)
	}

	input.FromBuffer(ibuffer)

	status = interpreter.Invoke()
	if status != tflite.StatusOk {
		log.Println("invoke failed")
		return
	}

	output, err := interpreter.GetOutputTensor(0)
	if err != nil {
		log.Println("cannot get output tensor", err)
	}
	outputSize := output.Dim(output.NumDims() - 1)
	b := make([]byte, outputSize)

	status = output.ToBuffer(&b[0])
	if status != tflite.StatusOk {
		log.Println("output failed")
		return
	}

	r := getResults(outputSize, b)

	for i := 0; i < len(r); i++ {
		fmt.Printf("%s: %f\n", labels[r[i].index], r[i].score)
		if i > 7 {
			break
		}
	}
}

type cResult struct {
	score float64
	index int
}

func captureImage(path string) (string, error) {
	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	s := raspicam.NewStill()
	s.Width = 244
	s.Height = 244
	s.Quality = 100
	s.Encoding = 3

	errCh := make(chan error)
	go func() {
		for x := range errCh {
			fmt.Fprintf(os.Stderr, "%v\n", x)
		}
	}()
	log.Println("Capturing image...")
	raspicam.Capture(s, f, errCh)
	return path, nil
}

// loadLabels
func getLabels(filename string) ([]string, error) {
	labels := []string{}
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	return labels, nil
}

// decodeImage ...
func decodeImage(imagePath string) image.Image {
	f, err := os.Open(imagePath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		log.Fatal(err)
	}
	return img
}

// imageToBuffer ...
func imageToBuffer(imagePath string, t *tflite.Tensor) ([]byte, error) {
	imageHeight := t.Dim(1)
	imagewidth := t.Dim(2)
	channels := t.Dim(3)
	wantedType := t.Type()

	img := decodeImage(imagePath)

	resized := resize.Resize(uint(imagewidth), uint(imageHeight), img, resize.NearestNeighbor)
	bounds := resized.Bounds()
	dx, dy := bounds.Dx(), bounds.Dy()

	if wantedType == tflite.TfLiteUInt8 {
		bb := make([]byte, dx*dy*channels)
		for y := 0; y < dy; y++ {
			for x := 0; x < dx; x++ {
				col := resized.At(x, y)
				r, g, b, _ := col.RGBA()
				i := y*dx + x
				bb[(i)*3+0] = byte(float64(r) / 255.0)
				bb[(i)*3+1] = byte(float64(g) / 255.0)
				bb[(i)*3+2] = byte(float64(b) / 255.0)
			}
		}
		return bb, nil
	}
	return nil, fmt.Errorf("incorrect type")
}

func getResults(outputSize int, b []byte) []cResult {

	results := []cResult{}
	for i := 0; i < outputSize; i++ {
		score := float64(b[i]) / 255.0
		if score < 0.4 {
			continue
		}
		results = append(results, cResult{score: score, index: i})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})
	return results
}
