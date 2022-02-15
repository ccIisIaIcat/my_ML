package liner_regression

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

/*使用案例

func main() {

	data := []float64{1, 1, 1, 2, 1, 3, 1, 4}
	data2 := []float64{2, 3, 4, 5}

	rl := liner_regression.ML_re{X: data, Y: data2, N_size: 4, P_size: 2}
	rl.Append_info([]float64{1, 5}, 6)
	rl.Mul_rl()
	// fmt.Println(rl.Predicate([]float64{1, 6}))

}


*/

type ML_re struct {
	X      []float64
	Y      []float64
	N_size int
	P_size int
	P      []float64
}

func (M *ML_re) Append_info(x_new []float64, y_new float64) bool {
	if M.P_size == 0 {
		M.P_size = len(x_new)
	} else {
		if M.P_size != len(x_new) {
			panic("新增数据维度不一致")
		}
	}
	M.N_size++
	M.Y = append(M.Y, y_new)
	M.X = append(M.X, x_new...)
	return true
}

func (M *ML_re) Mul_rl() []float64 {
	x := mat.NewDense(M.N_size, M.P_size, M.X)
	y := mat.NewDense(M.N_size, 1, M.Y)
	answer := get_linerinformation(x, y, M.P_size)
	M.P = answer
	return answer
}

func (M *ML_re) Mul_rl_ridge(k float64) []float64 {
	x := mat.NewDense(M.N_size, M.P_size, M.X)
	y := mat.NewDense(M.N_size, 1, M.Y)
	answer := get_ridge_linerinformation(x, y, M.P_size, k)
	M.P = answer
	return answer
}

func (M *ML_re) Predicate(x []float64) float64 {
	if len(M.P) == 0 {
		fmt.Println("尚未完成训练")
		return 0
	} else if len(x) != M.P_size {
		fmt.Println("输入数据维度错误")
		return 0
	} else {
		answer := float64(0)
		for i := 0; i < M.P_size; i++ {
			answer += M.P[i] * x[i]
		}
		return answer
	}
}

func get_linerinformation(x *mat.Dense, y *mat.Dense, p_size int) []float64 {
	var tool_matric mat.Dense
	var b_information mat.Dense
	tool_matric.Mul(x.T(), x)
	err := tool_matric.Inverse(&tool_matric)
	if err != nil {
		fmt.Println("矩阵求逆过程出现错误")
		fmt.Println("通常出现类似问题原因有：")
		fmt.Println("1、数据量过少或数据维度过大；2、存在数据行或参数列的高线性相关")
		fmt.Println("建议使用岭回归")
	}
	b_information.Mul(x.T(), y)
	b_information.Mul(&tool_matric, &b_information)
	fc := mat.Formatted(&b_information, mat.Prefix(" "), mat.Squeeze())
	fmt.Println("参数计算结果")
	fmt.Printf("贝塔 =\n %v\n", fc)
	fmt.Println("_________________________________")
	new_slice := make([]float64, p_size)
	for i := 0; i < p_size; i++ {
		new_slice[i] = b_information.At(i, 0)
	}
	return new_slice
}

func get_ridge_linerinformation(x *mat.Dense, y *mat.Dense, p_size int, k float64) []float64 {
	var tool_matric mat.Dense
	var b_information mat.Dense
	I := mat.NewDense(p_size, p_size, nil)
	for i := 0; i < p_size; i++ {
		I.Set(i, i, k)
	}
	tool_matric.Mul(x.T(), x)
	tool_matric.Add(&tool_matric, I)
	err := tool_matric.Inverse(&tool_matric)
	if err != nil {
		fmt.Println("矩阵求逆过程出现错误")
	}
	b_information.Mul(x.T(), y)
	b_information.Mul(&tool_matric, &b_information)
	fc := mat.Formatted(&b_information, mat.Prefix(" "), mat.Squeeze())
	fmt.Println("参数计算结果")
	fmt.Printf("贝塔 =\n %v\n", fc)
	fmt.Println("_________________________________")
	new_slice := make([]float64, p_size)
	for i := 0; i < p_size; i++ {
		new_slice[i] = b_information.At(i, 0)
	}
	return new_slice
}
