package naive_bayes

type N_b struct {
	X      []string
	Y      []string
	N_size int
	P_size int
	X_dic  []map[string]int
	Y_dic  map[string]int
}

//结构体的初始化
func (N *N_b) init_() {
	N.N_size = len(N.Y)
	N.P_size = len(N.X) / len(N.Y)
	N.X_dic = make([]map[string]int, N.P_size)
	N.Y_dic = make(map[string]int, 0)
}

//统计y，Xi的种类和对应个数
func (N *N_b) make_xy_dic() {
	for i := 0; i < N.P_size; i++ {
		temp_dic := make(map[string]int, 0)
		for j := 0; j < N.N_size; j++ {
			if temp_dic[N.X[j*N.P_size+i]] == 0 {
				temp_dic[N.X[j*N.P_size+i]] = 1
			} else {
				temp_dic[N.X[j*N.P_size+i]] += 1
			}
		}
		N.X_dic[i] = temp_dic
	}
	for i := 0; i < N.N_size; i++ {
		if N.Y_dic[N.Y[i]] == 0 {
			N.Y_dic[N.Y[i]] = 1
		} else {
			N.Y_dic[N.Y[i]] += 1
		}
	}
}

//判断该样本下y出现的概率
func (N *N_b) Predict(X_sample []string, Y_sample string) float64 {
	N.init_()
	N.make_xy_dic()
	if N.Y_dic[Y_sample] == 0 {
		return float64(0)
	}
	p_y_sample := float64(N.Y_dic[Y_sample]) / float64(N.N_size)
	p_xy_sample := make([]float64, N.P_size)
	for i := 0; i < N.P_size; i++ {
		temp_x := X_sample[i]
		for j := 0; j < N.N_size; j++ {
			if N.Y[j] == Y_sample && N.X[j*N.P_size+i] == temp_x {
				p_xy_sample[i] += 1
			}
		}
		p_xy_sample[i] = p_xy_sample[i] / float64(N.Y_dic[Y_sample])
	}
	numerator := p_y_sample
	for i := 0; i < N.P_size; i++ {
		numerator = numerator * p_xy_sample[i]
	}
	denominator := float64(0)
	for k, v := range N.Y_dic {
		p_xy_sample := make([]float64, N.P_size)
		temp_this := float64(1)
		y_this := k
		for i := 0; i < N.P_size; i++ {
			temp_x := X_sample[i]
			for j := 0; j < N.N_size; j++ {
				if N.Y[j] == y_this && N.X[j*N.P_size+i] == temp_x {
					p_xy_sample[i] += 1
				}
			}
			p_xy_sample[i] = p_xy_sample[i] / float64(N.Y_dic[y_this])
		}
		temp_this = float64(v) / float64(N.N_size)
		for i := 0; i < N.P_size; i++ {
			temp_this = temp_this * p_xy_sample[i]
		}
		denominator += temp_this
	}
	return numerator / denominator
}

//挑选可能性最大的y
func (N *N_b) Judge(X_sample []string) string {
	p_max := float64(0)
	answer := "该样本无法判断"
	for k := range N.Y_dic {
		if p_max < N.Predict(X_sample, k) {
			p_max = N.Predict(X_sample, k)
			answer = k
		}
	}
	return answer
}

/*一个实例
func main() {
	fmt.Println("HELLO")
	test_sample := read_excel.D_ex_s("./test_data/sample_3.xlsx", "Sheet1")
	n_b := naive_bayes.N_b{X: test_sample["X"], Y: test_sample["Y"]}
	aa := []string{"1", "1", "1"}
	fmt.Println(n_b.Predict(aa, "0"))
	fmt.Println(n_b.Judge(aa))
}

*/
