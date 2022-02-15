package svm

import (
	"fmt"
	"my_ML/smo"
)

type Svm struct {
	X        []float64 //必写
	Y        []float64 //必写
	N_size   int
	P_size   int
	A_set    []float64
	C        float64 //必写
	MAX_iter int     //必写

	// 内部计算的变量
	w []float64
	b float64
}

func (S *Svm) init() {

	S.N_size = len(S.Y)
	S.P_size = len(S.X) / len(S.Y)
	temp_s := smo.Smo{X: S.X, Y: S.Y, C: S.C}
	S.A_set = temp_s.Get_A_set(S.MAX_iter)
	S.w = make([]float64, S.P_size)
	S.b = 0

}

func (S *Svm) get_w() {
	for i := 0; i < S.P_size; i++ {
		for j := 0; j < S.N_size; j++ {
			S.w[i] += S.A_set[j] * S.Y[j] * S.X[j*S.P_size+i]
		}
	}
}

func (S *Svm) get_b() {
	a_max_id := int(0)
	a_max := S.A_set[0]
	for i := 0; i < S.N_size; i++ {
		if S.A_set[i] >= a_max {
			a_max = S.A_set[i]
			a_max_id = i
		}
	}
	temp := float64(0)
	for i := 0; i < S.P_size; i++ {
		temp += S.w[i] * S.X[a_max_id*S.P_size+i]
	}
	S.b = S.Y[a_max_id] - temp
}

func (S *Svm) Fit() ([]float64, float64) {
	S.init()
	S.get_w() // 一定要先算w再算b
	S.get_b()
	ww := S.w
	bb := S.b
	fmt.Println("W的计算结果为：", ww)
	fmt.Println("b的计算结果为：", bb)
	return ww, bb

}

/*一个例子
func main() {

	fmt.Println("HELLO")
	sample_data := read_excel.D_ex("./test_data/sample_6.xlsx", "Sheet1")

	s := svm.Svm{X: sample_data["X"], Y: sample_data["Y"], C: 1, MAX_iter: 1500}
	s.Fit()

}
*/
