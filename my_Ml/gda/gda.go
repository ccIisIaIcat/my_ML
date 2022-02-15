package gda

import (
	"fmt"

	"gonum.org/v1/gonum/mat"

	"gonum.org/v1/gonum/stat/distmv"
)

type Gda struct {
	//基本参数
	X       []float64
	Y       []float64
	N_size  int
	P_size  int
	d_set_1 []int
	d_set_2 []int
	//需要计算的参数
	fai float64
	u1  []float64
	u2  []float64
	z   [][]float64
	//用于后续预测的类型
	z_matrix mat.Symmetric
}

func (G *Gda) init_() {
	G.N_size = len(G.Y)
	G.P_size = len(G.X) / len(G.Y)
	G.u1 = make([]float64, G.P_size)
	G.u2 = make([]float64, G.P_size)
	G.z = make([][]float64, G.P_size)
	for i := 0; i < G.P_size; i++ {
		temp := make([]float64, G.P_size)
		G.z[i] = append(G.z[i], temp...)
	}
	G.fai = 0
	//找到两类分类的点
	for i := 0; i < G.N_size; i++ {
		if G.Y[i] == float64(1) {
			G.d_set_1 = append(G.d_set_1, i)
		} else {
			G.d_set_2 = append(G.d_set_2, i)
		}
	}

}

//确定参数fai
func (G *Gda) find_fai() {
	for i := 0; i < G.N_size; i++ {
		G.fai += G.Y[i]
	}
	G.fai = G.fai / float64(G.N_size)

}

//确定参数u1
func (G *Gda) find_u1() {
	for i := 0; i < G.P_size; i++ {
		for j := 0; j < G.N_size; j++ {
			G.u1[i] += G.Y[j] * G.X[j*G.P_size+i]
		}
		G.u1[i] = G.u1[i] / float64(len(G.d_set_1))
	}

}

//确定参数u2
func (G *Gda) find_u2() {
	for i := 0; i < G.P_size; i++ {
		for j := 0; j < G.N_size; j++ {
			G.u2[i] += (1 - G.Y[j]) * G.X[j*G.P_size+i]
		}
		G.u2[i] = G.u2[i] / float64(len(G.d_set_2))
	}
}

//确定参数z
func (G *Gda) find_z() {
	u_v := make([]float64, G.P_size)
	for i := 0; i < G.P_size; i++ {
		for j := 0; j < G.N_size; j++ {
			u_v[i] += G.X[G.P_size*j+i]
		}
		u_v[i] = u_v[i] / float64(G.N_size)
	}
	x_u := make([]float64, G.P_size*G.N_size)
	for i := 0; i < G.N_size; i++ {
		for j := 0; j < G.P_size; j++ {
			x_u[i*G.P_size+j] = G.X[i*G.P_size+j] - u_v[j]
		}
	}
	for i := 0; i < G.N_size; i++ {
		for j := 0; j < G.P_size; j++ {
			for k := 0; k < G.P_size; k++ {
				G.z[j][k] += x_u[i*G.P_size+j] * x_u[i*G.P_size+k] / float64(G.N_size)
			}
		}
	}
}

//训练参数
func (G *Gda) Fit() {
	G.init_()
	G.find_fai()
	G.find_u1()
	G.find_u2()
	G.find_z()
	fmt.Println("fai:", G.fai)
	fmt.Println("u1:", G.u1)
	fmt.Println("u2:", G.u2)
	fmt.Println("z:", G.z)
}

//预测在样本下，两类事件分别发生的概率
func (G *Gda) Pridict(x_sample []float64) (float64, float64) {
	tool_ := make([]float64, G.P_size*G.P_size)
	for i := 0; i < G.P_size; i++ {
		for j := 0; j < G.P_size; j++ {
			tool_[i*G.P_size+j] = G.z[i][j]
		}
	}
	G.z_matrix = mat.NewSymDense(G.P_size, tool_)
	normal_1, _ := distmv.NewNormal(G.u1, G.z_matrix, nil)
	normal_2, _ := distmv.NewNormal(G.u2, G.z_matrix, nil)
	//计算属于“1”类的概率
	p_1 := normal_1.Prob(x_sample)
	fmt.Println("属于一类的概率为：", p_1)
	p_2 := normal_2.Prob(x_sample)
	fmt.Println("属于零类的概率为：", p_2)
	return p_1, p_2

}

/*一个案例

func main() {

	fmt.Println("HELLO")
	test_sample := read_excel.D_ex("./test_data/sample_4.xlsx", "Sheet1")
	g_d := gda.Gda{X: test_sample["X"], Y: test_sample["Y"]}
	g_d.Fit()
	aaa := []float64{0, 0, 0}
	g_d.Pridict(aaa)

}


*/
