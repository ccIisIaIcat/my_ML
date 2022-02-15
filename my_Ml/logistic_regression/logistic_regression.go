package logistic_regression

import (
	"fmt"
	"math/rand"
	"time"

	"math"
)

type Lo_r struct {
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

func (L *Lo_r) Information_() map[string]interface{} {
	answer := make(map[string]interface{})
	answer["样本容量"] = L.N_size
	answer["样本维度"] = L.P_size
	answer["训练结果"] = L.B
	answer["错分样本个数"] = L.d_length
	answer["损失函数值"] = L.Loss_value
	answer["错分少半后最大迭代次数（N_T）"] = L.N_T
	fmt.Println("样本容量:", answer["样本容量"])
	fmt.Println("样本维度:", answer["样本维度"])
	fmt.Println("训练结果:", answer["训练结果"])
	fmt.Println("错分样本个数:", answer["错分样本个数"])
	fmt.Println("损失函数值:", answer["损失函数值"])
	fmt.Println("错分少半后最大迭代次数（N_T）:", answer["错分少半后最大迭代次数（N_T）"])
	return answer
}

func (L *Lo_r) find_d_set() {
	temp := float64(0)
	L.d_length = 0
	P_ := float64(0)
	for i := 0; i < L.N_size; i++ {
		temp = float64(0)
		for j := 0; j < L.P_size; j++ {
			temp += L.X[i*L.P_size+j] * L.B[j]
		}
		P_ = 1 / (1 + math.Exp(-temp))
		if (P_-0.5)*(L.Y[i]-0.5) < 0 {
			L.d[L.d_length] = i
			L.d_length++
		}
	}
}

//信息初始化
func (L *Lo_r) init_() {
	L.N_size = len(L.Y)
	L.P_size = len(L.X) / len(L.Y)
	L.d = make([]int, L.N_size)
	L.gradient = make([]float64, L.P_size)
	L.d_length = 0
}

//计算出分类错误的点向下的梯度
func (L *Lo_r) find_gradient() {
	temp := float64(0)
	a := float64(0)
	for i := 0; i < L.N_size; i++ {
		a = 0
		temp = 0
		for j := 0; j < L.P_size; j++ {
			temp += L.B[j] * L.X[i*L.P_size+j]
		}
		a = L.Y[i] - 1/(1+math.Exp(-temp))
		for k := 0; k < L.P_size; k++ {
			L.gradient[k] += a * L.X[i*L.P_size+k]
		}
	}
	for k := 0; k < L.P_size; k++ {
		L.gradient[k] = L.gradient[k] / float64(L.N_size)
	}
}

// 计算当前损失函数的值
func (L *Lo_r) lost_value() {
	temp := float64(0)
	a := float64(0)
	for i := 0; i < L.N_size; i++ {
		temp = 0
		for j := 0; j < L.P_size; j++ {
			temp += L.B[j] * L.X[i*L.P_size+j]
		}
		a += (L.Y[i]-1)*temp - math.Log(1+math.Exp(-temp))
	}
	L.Loss_value = -a
	fmt.Println(L.Loss_value)

}

//随机选择一个方向更新
func (L *Lo_r) upgrade1() {
	rand.Seed(time.Now().Unix())
	random_num := rand.Intn(L.P_size)
	L.B[random_num] = L.B[random_num] + L.K*L.gradient[random_num]
	L.find_d_set()
	L.find_gradient()
	L.lost_value()
}

func (L *Lo_r) Iteration() []float64 {
	L.init_()
	L.find_d_set()
	L.find_gradient()
	L.lost_value()
	judge := 0

	for {
		L.upgrade1()
		judge++
		if L.d_length == 0 || judge >= L.N_T {
			fmt.Println(">>>>>>>>>>>>>>>>>>>>>>>>迭代最终结果>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
			L.Information_()
			break
		}
	}

	return L.B
}
