package k_means

import (
	"fmt"
)

type K_means struct {
	X            []float64 //必写
	Y            []float64 //必写(可以传入一个全为零的向量)
	K            int       //必写
	kernal_point []int
	N_size       int
	P_size       int
	Distance_set [][]float64
	judge_set    []int
	K_means_set  [][]float64
}

func (K *K_means) init() {

	K.N_size = len(K.Y)
	K.P_size = len(K.X) / len(K.Y)
	K.Distance_set = make([][]float64, 0)
	K.K_means_set = make([][]float64, 0)
	for i := 0; i < K.K; i++ {
		temp_arr := make([]float64, K.P_size)
		for j := 0; j < K.P_size; j++ {
			temp_arr[j] = K.X[i*K.P_size+j+j]
		}
		K.K_means_set = append(K.K_means_set, temp_arr)
	}
	K.judge_set = make([]int, K.N_size)

}

func (K *K_means) make_Distance_set() {
	K.Distance_set = make([][]float64, 0)
	for i := 0; i < K.N_size; i++ {
		temp_arr := make([]float64, K.K)
		for j := 0; j < K.K; j++ {
			for k := 0; k < K.P_size; k++ {
				temp_arr[j] += (K.X[i*K.P_size+k] - K.K_means_set[j][k]) * (K.X[i*K.P_size+k] - K.K_means_set[j][k])
			}
		}
		K.Distance_set = append(K.Distance_set, temp_arr)
	}
}

func (K *K_means) make_judge() {
	for i := 0; i < K.N_size; i++ {
		min_d := K.Distance_set[i][0]
		min_id := 0
		for j := 0; j < K.K; j++ {
			if K.Distance_set[i][j] < min_d {
				min_d = K.Distance_set[i][j]
				min_id = j
			}
		}
		K.judge_set[i] = min_id
	}
	fmt.Println(K.judge_set)

}

func (K *K_means) make_k_means_set() {
	temp_count := make([]int, K.K)
	K.K_means_set = make([][]float64, 0)
	for i := 0; i < K.K; i++ {
		temp_arr := make([]float64, K.P_size)
		K.K_means_set = append(K.K_means_set, temp_arr)
	}
	for i := 0; i < K.N_size; i++ {
		for j := 0; j < K.P_size; j++ {
			K.K_means_set[K.judge_set[i]][j] += K.X[i*K.P_size+j]
		}
		temp_count[K.judge_set[i]] += 1
	}
	for i := 0; i < K.K; i++ {
		for j := 0; j < K.P_size; j++ {
			if float64(temp_count[i]) != 0 {
				K.K_means_set[i][j] = K.K_means_set[i][j] / float64(temp_count[i])
			}
		}
	}
}

func (K *K_means) Ira(times int) {
	K.init()
	for i := 0; i < times; i++ {
		K.make_Distance_set()
		K.make_judge()
		K.make_k_means_set()
	}

}

/* 一个例子
func main() {
	fmt.Println("hello world")
	sample_data := read_excel.D_ex("./test_data/sample_8.xlsx", "Sheet1")
	k_m := k_means.K_means{X: sample_data["X"], Y: sample_data["Y"], K: 10}
	k_m.Ira(15)

}
*/
