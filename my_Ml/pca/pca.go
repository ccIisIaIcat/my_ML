package pca

import (
	"gonum.org/v1/gonum/mat"

	"fmt"
)

type Pca struct {
	X             []float64
	Y             []float64
	N_size        int
	P_size        int
	N_delete_dim  int
	X_matrix      mat.Matrix
	H_matrix      mat.Matrix
	Q_matrix      mat.Matrix
	Answer_matrix mat.Matrix
}

//基本数据的初始化
func (P *Pca) init_() {
	P.N_size = len(P.Y)
	P.P_size = len(P.X) / len(P.Y)
	P.X_matrix = mat.NewDense(P.N_size, P.P_size, P.X)
	temp := make([]float64, P.N_size*P.N_size)
	for i := 0; i < P.N_size; i++ {
		for j := 0; j < P.N_size; j++ {
			if i == j {
				temp[i*P.N_size+j] = 1 - 1/float64(P.N_size)
			} else {
				temp[i+P.N_size+j] = -1 / float64(P.N_size)
			}
		}
	}
	P.H_matrix = mat.NewDense(P.N_size, P.N_size, temp)

}

//计算中心化以后的向量组样本方差转置与自身乘积的特征值和特征向量
//并从中找到绝对值最大的p个方向
func (P *Pca) make_Q() {
	var YY mat.Dense
	YY.Mul(P.H_matrix, P.X_matrix)
	var S mat.Dense
	S.Mul(YY.T(), &YY)
	new_slice := make([]float64, P.P_size*P.P_size)
	for i := 0; i < P.P_size; i++ {
		for j := 0; j < P.P_size; j++ {
			new_slice[i*P.P_size+j] = S.At(i, j)
		}
	}

	SS := mat.NewSymDense(P.P_size, new_slice)
	var eig mat.EigenSym
	eig.Factorize(SS, true)
	r := eig.Values(nil)
	var ev mat.Dense
	eig.VectorsTo(&ev)
	r_chose := make([]int, P.P_size-P.N_delete_dim)
	fmt.Println("正交变换后各维度方差：", r)
	for j := 0; j < P.P_size-P.N_delete_dim; j++ {
		max_ := float64(0)
		r_max := -1
		for i := 0; i < P.P_size; i++ {
			if r[i] >= max_ {
				max_ = r[i]
				r_max = i
			}
		}
		r[r_max] = 0
		r_chose[j] = r_max
	}

	new_data := make([]float64, 0)
	for i := 0; i < P.P_size-P.N_delete_dim; i++ {
		new_data = append(new_data, mat.Col(nil, r_chose[i], &ev)...)
	}
	P.Q_matrix = mat.NewDense(P.P_size, P.P_size-P.N_delete_dim, new_data)
	fmt.Println("所选维度：", "  第", r_chose, "维", "(从零维开始计算)")

}

//进一步计算原数据在新维度上的投影，并返回一个切片用于保存新的数据
func (P *Pca) Back_data() []float64 {
	P.init_()
	P.make_Q()
	var aa mat.Dense
	aa.Mul(P.H_matrix, P.X_matrix)
	var bb mat.Dense
	bb.Mul(&aa, P.Q_matrix)
	P.Answer_matrix = &bb
	back := make([]float64, P.N_size*(P.P_size-P.N_delete_dim))
	for i := 0; i < P.N_size; i++ {
		for j := 0; j < P.P_size-P.N_delete_dim; j++ {
			back[(P.P_size-P.N_delete_dim)*i+j] = bb.At(i, j)
		}
	}
	return back
}

//对新数据进行格式化展示
func (P *Pca) Show_new_data() {
	fmt.Println("New Data!!!!")
	fmt.Println(mat.Formatted(P.Answer_matrix))
}

func (P *Pca) Show_Q_matrix() mat.Matrix {
	fmt.Println("Q_matrix!!!")
	fmt.Println(mat.Formatted(P.Q_matrix))
	return P.Q_matrix
}

/*
一个案例
func main() {

	fmt.Println("HELLO")
	data_sample := read_excel.D_ex("./test_data/sample_5.xlsx", "Sheet1")
	p_a := pca.Pca{X: data_sample["X"], Y: data_sample["Y"], N_delete_dim: 1}
	p_a.Back_data()
	p_a.Show_new_data()

}
*/
