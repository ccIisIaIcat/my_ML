package knn

type Knn struct {
	X            []float64 //必写
	Y            []float64 //必写
	K            int       //必写
	Sample_point []float64 //必写
	N_size       int
	P_size       int
	Distance_set []float64
	Knn_set      []float64
	Knn_set_id   []int
}

func (K *Knn) init() {

	K.N_size = len(K.Y)
	K.P_size = len(K.X) / len(K.Y)
	K.Distance_set = make([]float64, K.N_size)
	K.Knn_set = make([]float64, K.K)
	K.Knn_set_id = make([]int, K.K)

}

//计算目标向量和样本中每个向量的欧氏距离，存入切片K.Distance_set
func (K *Knn) make_d_set() {
	for i := 0; i < K.N_size; i++ {
		temp_distance := float64(0)
		for j := 0; j < K.P_size; j++ {
			temp_distance += (K.X[K.P_size*i+j] - K.Sample_point[j]) * (K.X[K.P_size*i+j] - K.Sample_point[j])
		}
		K.Distance_set[i] = temp_distance
	}
}

//找出其中最近的k个点，对应id存入K.Knn_set_id
func (K *Knn) make_knn_set() {
	max_in_knn_set := K.Distance_set[0]
	max_id := 0
	for i := 0; i < K.N_size; i++ {
		if i < K.K {
			K.Knn_set[i] = K.Distance_set[i]
			K.Knn_set_id[i] = i
			if K.Distance_set[i] > max_in_knn_set {
				max_in_knn_set = K.Distance_set[i]
				max_id = i
			}
		} else {
			if K.Distance_set[i] < max_in_knn_set {
				K.Knn_set[max_id] = K.Distance_set[i]
				K.Knn_set_id[max_id] = i
				max_in_knn_set = K.Distance_set[i]
				for j := 0; j < K.K; j++ {
					if K.Knn_set[j] > max_in_knn_set {
						max_in_knn_set = K.Knn_set[j]
						max_id = j
					}
				}
			}
		}
	}

}

//找出K.Knn_set_id中出现频率最高的类型，作为输出
func (K *Knn) Judge() float64 {
	K.init()
	K.make_d_set()
	K.make_knn_set()
	judge_map := make(map[float64]int, 0)
	for i := 0; i < K.K; i++ {
		if judge_map[K.Y[K.Knn_set_id[i]]] == 0 {
			judge_map[K.Y[K.Knn_set_id[i]]] = 1
		} else {
			judge_map[K.Y[K.Knn_set_id[i]]] += 1
		}
	}
	max_num := 0
	final_type := float64(0)
	for k, v := range judge_map {
		if v >= max_num {
			max_num = v
			final_type = k
		}
	}
	return final_type
}

/* 一个例子

func main() {

	sample_data := read_excel.D_ex("./sample_7.xlsx", "Sheet1")
	knn := knn.Knn{X: sample_data["X"], Y: sample_data["Y"], K: 3, Sample_point: []float64{8, 8, 8, 8}}
	fmt.Println(knn.Judge())
}
*/
