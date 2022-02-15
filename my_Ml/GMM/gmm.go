package GMM

import (
	"fmt"
	"my_ML/read_excel"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

type Gmm struct {
	X         []float64   //必写
	Y         []float64   //必写
	U0        [][]float64 //必写
	Z0        [][]float64 //必写
	Times     int         //必写
	P0        []float64
	z0        []*mat.SymDense
	k         int
	dimension int
	N_size    int
	P_size    int
}

func (G *Gmm) init() {
	G.N_size = len(G.Y)
	G.P_size = len(G.X) / len(G.Y)
	G.k = len(G.U0)
	G.z0 = make([]*mat.SymDense, G.k)
	G.dimension = len(G.U0[0])
	//生成协方差矩阵
	for i := 0; i < G.k; i++ {
		G.z0[i] = mat.NewSymDense(G.dimension, G.Z0[i])
	}
	//初始化P0
	G.P0 = make([]float64, G.k)
	for i := 0; i < G.k; i++ {
		G.P0[i] = 1 / float64(G.k)
	}
	// G.z_matrix = mat.NewSymDense(G.P_size, tool_)
	// normal_1, _ := distmv.NewNormal(G.u1, G.z_matrix, nil)
	// normal_2, _ := distmv.NewNormal(G.u2, G.z_matrix, nil)

}

func (G *Gmm) calculate_pro(id int, k int) float64 {
	normal, _ := distmv.NewNormal(G.U0[k], (G.z0[k]), nil)
	answer := normal.Prob(G.X[id*G.P_size : (id+1)*G.P_size])
	return answer
}

func (G *Gmm) calculate_gamaij(i int, j int) float64 {
	a := G.P0[j] * G.calculate_pro(i, j)
	b := float64(0)
	for num := 0; num < G.k; num++ {
		b += G.P0[num] * G.calculate_pro(i, num)
	}
	return a / b
}

func (G *Gmm) updata_P() []float64 {
	p_temp := make([]float64, G.k)
	for k_ := 0; k_ < G.k; k_++ {
		for i := 0; i < G.N_size; i++ {
			p_temp[k_] += G.calculate_gamaij(i, k_)
		}
		p_temp[k_] = p_temp[k_] / float64(G.N_size)
	}
	return p_temp
}

func (G *Gmm) updata_u() [][]float64 {
	temp_u := make([][]float64, G.k)
	for j := 0; j < G.k; j++ {
		b := float64(0)
		for i := 0; i < G.N_size; i++ {
			b += G.calculate_gamaij(i, j)
		}
		temp_u[j] = make([]float64, G.P_size)
		for k := 0; k < G.P_size; k++ {
			for i := 0; i < G.N_size; i++ {
				temp_u[j][k] += G.X[i*G.P_size+k] * G.calculate_gamaij(i, j)
			}
			temp_u[j][k] /= b
		}

	}
	return temp_u
}

func (G *Gmm) calculate_vecter_minus(x_id int, u_id int) []float64 {
	answer := make([]float64, G.P_size)
	for i := 0; i < G.P_size; i++ {
		answer[i] = G.X[x_id*G.P_size+i] - G.U0[u_id][i]
	}
	return answer
}

func (G *Gmm) make_a_diag_matirx(value float64, size int) mat.Dense {
	new_arr := make([]float64, size*size)
	for i := 0; i < size; i++ {
		new_arr[i*size+i] = value
	}
	new_matrix := mat.NewDense(size, size, new_arr)
	return *new_matrix

}

func (G *Gmm) updata_z() []*mat.Dense {
	temp_z0 := make([]*mat.Dense, G.k)
	for j := 0; j < G.k; j++ {
		matrix_now := mat.NewDense(G.P_size, G.P_size, G.U0[0])
		matrix_now.Zero()
		for i := 0; i < G.N_size; i++ {
			x_temp := mat.NewDense(G.P_size, 1, G.calculate_vecter_minus(i, j))
			var tool_matrix mat.Dense
			tool_matrix.Mul(x_temp, x_temp.T())
			new_tool := G.make_a_diag_matirx(G.calculate_gamaij(i, j), G.P_size)
			tool_matrix.Mul(&tool_matrix, &new_tool)
			matrix_now.Add(matrix_now, &tool_matrix)
		}
		temp_value := float64(0)
		for i := 0; i < G.N_size; i++ {
			temp_value += G.calculate_gamaij(i, j)
		}
		new_new_matrix := G.make_a_diag_matirx(1/temp_value, G.P_size)
		matrix_now.Mul(matrix_now, &new_new_matrix)
		temp_z0[j] = matrix_now
	}
	return temp_z0
}

func (G *Gmm) turn_matrix_to_sysmatrix(matrix mat.Dense) *mat.SymDense {
	size, _ := matrix.Dims()
	new_slice := make([]float64, size*size)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			new_slice[i*size+j] = matrix.At(i, j)
		}
	}
	answer := mat.NewSymDense(size, new_slice)
	return answer

}

func (G *Gmm) Fit() ([]float64, [][]float64, []*mat.SymDense) {
	G.init()
	for i := 0; i < G.Times; i++ {
		p_temp := G.updata_P()
		u_temp := G.updata_u()
		z_temp := G.updata_z()
		G.P0 = p_temp
		G.U0 = u_temp
		for i := 0; i < G.k; i++ {
			G.z0[i] = G.turn_matrix_to_sysmatrix(*z_temp[i])
		}
	}
	fmt.Println("权重：", G.P0)
	fmt.Println("均值：", G.U0)
	fmt.Println("协方差矩阵：", G.z0)
	return G.P0, G.U0, G.z0

}

func main() {
	data := read_excel.D_ex("./sample_10.xlsx", "Sheet1")
	fmt.Println(data)
	u := [][]float64{{4}, {6}}
	z := [][]float64{{1}, {1}}
	gm := Gmm{X: data["X"], Y: data["Y"], U0: u, Z0: z, Times: 1000}
	gm.Fit()

}

/* 一个例子
func main() {
	data := read_excel.D_ex("./test_data/sample_10.xlsx", "Sheet1")
	u := [][]float64{{4}, {6}}
	z := [][]float64{{1}, {1}}
	gm := GMM.Gmm{X: data["X"], Y: data["Y"], U0: u, Z0: z, Times: 1000}
	gm.Fit()

}
*/
