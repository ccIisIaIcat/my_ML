package perception

//最下方附有一个使用案例

import (
	"fmt"

	"math/rand"

	"time"
)

type Per struct {
	X          []float64
	Y          []float64
	N_size     int
	P_size     int
	B          []float64
	d          []int
	d_length   int
	gradient   []float64
	K          float64
	Loss_value float64
	N_T        int
}

func (P *Per) Information_() map[string]interface{} {
	answer := make(map[string]interface{})
	answer["样本容量"] = P.N_size
	answer["样本维度"] = P.P_size
	answer["训练结果"] = P.B
	answer["错分样本个数"] = P.d_length
	answer["损失函数值"] = P.Loss_value
	answer["错分少半后最大迭代次数（N_T）"] = P.N_T
	fmt.Println("样本容量:", answer["样本容量"])
	fmt.Println("样本维度:", answer["样本维度"])
	fmt.Println("训练结果:", answer["训练结果"])
	fmt.Println("错分样本个数:", answer["错分样本个数"])
	fmt.Println("损失函数值:", answer["损失函数值"])
	fmt.Println("错分少半后最大迭代次数（N_T）:", answer["错分少半后最大迭代次数（N_T）"])
	return answer
}

//信息初始化
func (P *Per) init_() {
	P.N_size = len(P.Y)
	P.P_size = len(P.X) / len(P.Y)
	P.d = make([]int, P.N_size)
	P.gradient = make([]float64, P.P_size)
	P.d_length = 0
}

//找到分类错误的点
func (P *Per) find_d_set() {
	temp := float64(0)
	P.d_length = 0
	for i := 0; i < P.N_size; i++ {
		temp = float64(0)
		for j := 0; j < P.P_size; j++ {
			temp += P.X[i*P.P_size+j] * P.B[j]
		}
		if temp*P.Y[i] < 0 {
			P.d[P.d_length] = i
			P.d_length++
		}
	}
}

//计算出分类错误的点向下的梯度
func (P *Per) find_gradient() {
	var n_now int
	temp := float64(0)
	for i := 0; i < P.P_size; i++ {
		temp = 0
		for j := 0; j < P.d_length; j++ {
			n_now = P.d[j]
			temp += P.X[n_now*P.P_size+i] * P.Y[n_now]
		}
		P.gradient[i] = -temp
	}
}

// 计算当前损失函数的值
func (P *Per) lost_value() {

	temp := make([]float64, P.P_size)
	var row_now int
	temp_num := float64(0)
	for i := 0; i < P.P_size; i++ {
		temp_num = 0
		for j := 0; j < P.d_length; j++ {
			row_now = P.d[j]
			temp_num += (P.Y[row_now]) * (P.X[row_now*P.P_size+i])
		}
		temp[i] = float64(temp_num)
	}
	answer := float64(0)
	for i := 0; i < P.P_size; i++ {
		answer += P.B[i] * temp[i]
	}
	P.Loss_value = -answer
}

//随机选择一个方向更新(version1:不论在何种情况下都进行更新)
func (P *Per) upgrade1() {
	rand.Seed(time.Now().Unix())
	random_num := rand.Intn(P.P_size)
	P.B[random_num] = P.B[random_num] - P.K*P.gradient[random_num]
	P.find_d_set()
	P.find_gradient()
	P.lost_value()
	fmt.Println(P.Loss_value)
}

//随机选择一个方向更新(version2:只有在损失函数减少去情况下进行更新)
func (P *Per) upgrade2() int {
	rand.Seed(time.Now().Unix())
	random_num := rand.Intn(P.P_size)
	P.B[random_num] = P.B[random_num] - P.K*P.gradient[random_num]
	past_loss := P.Loss_value
	P.lost_value()
	if P.Loss_value < past_loss {
		P.find_d_set()
		P.find_gradient()
		//fmt.Println(P.Loss_value)
		return -1
	} else {
		P.B[random_num] = P.B[random_num] + P.K*P.gradient[random_num]
		P.lost_value()
		return 1
	}
}

func (P *Per) Iteration() []float64 {
	P.init_()
	P.find_d_set()
	P.find_gradient()
	P.lost_value()
	judge := 0

	for {
		// time.Sleep(time.Millisecond * 300)
		if P.d_length > P.N_size/2 {
			P.upgrade1()
		} else {
			P.upgrade2()
			judge++
			if P.d_length == 0 {
				break
			} else if judge > P.N_T {
				break
			}

		}
	}
	return P.B
}

/*
package main


import (
	"fmt"

	"my_ML/read_excel"

	"my_ML/perception"
)

func main() {
	fmt.Println("HELLO")
	test_answer := read_excel.D_ex("./test_data/sample_2.xlsx", "Sheet1")
	fmt.Println(test_answer["Y"])
	fmt.Println(test_answer["X"])
	b_in := []float64{1, 1, 1}

	pc := perception.Per{X: test_answer["X"], Y: test_answer["Y"], B: b_in, K: 0.1, N_T: 1000000}
	anananan := pc.Iteration()
	fmt.Println(anananan)
	pc.Information_()

}

*/
