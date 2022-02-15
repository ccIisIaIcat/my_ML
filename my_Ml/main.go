package main

import (
	"my_ML/GMM"
	"my_ML/read_excel"
)

func main() {
	data := read_excel.D_ex("./test_data/sample_10.xlsx", "Sheet1")
	u := [][]float64{{4}, {6}}
	z := [][]float64{{1}, {1}}
	gm := GMM.Gmm{X: data["X"], Y: data["Y"], U0: u, Z0: z, Times: 1000}
	gm.Fit()

}
