package main

import (
	"testing"
)

func TestGetLabels(t *testing.T) {

	tests := []struct {
		name  string
		index int
		want  string
	}{
		{
			name:  "checking background label",
			index: 0,
			want:  "background",
		},
		{
			name:  "checking koala label",
			index: 106,
			want:  "koala",
		},
	}

	labels, err := getLabels("models/labels_mobilenet_224.txt")
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := labels[tc.index]
			if tc.want != got {
				t.Errorf("LoadLabel got %q , but should be %q", got, tc.want)
			}
		})
	}
}
