package smo

//X,Y,C必须输入
type Smo struct {
	X      []float64 //必写
	Y      []float64 //必写
	N_size int
	P_size int
	A_set  []float64 //必写
	C      float64   //必写
	//当前迭代下的相关参数
	a_now  int
	b_now  int
	u_now  float64
	ga_now float64
	gb_now float64
	ha_now float64
	hb_now float64
}

func (S *Smo) init_() {
	S.N_size = len(S.Y)
	S.P_size = len(S.X) / len(S.Y)
	S.A_set = make([]float64, S.N_size) //若有更好的非零初值，可在init之后自行修改
	S.a_now = 0
	S.b_now = 1

}

//用于修改当前迭代的变量序号
func (S *Smo) chang_ab(a int, b int) {
	S.a_now = a
	S.b_now = b

}

//用于计算两个X向量的内积
func (S *Smo) k(a int, b int) float64 {
	answer := float64(0)
	for i := 0; i < S.P_size; i++ {
		answer += S.X[a*S.P_size+i] * S.X[b*S.P_size+i]
	}
	return answer

}

//计算一系列当前迭代下的相关参数
func (S *Smo) new_parameter() {
	S.u_now = S.A_set[S.a_now]*S.Y[S.a_now] + S.A_set[S.b_now]*S.Y[S.b_now]
	S.ga_now = 0
	S.gb_now = 0
	for i := 0; i < S.N_size; i++ {
		S.ga_now += S.A_set[i] * S.Y[i] * S.k(i, S.a_now)
		S.gb_now += S.A_set[i] * S.Y[i] * S.k(i, S.b_now)
	}
	S.ha_now = S.ga_now - S.A_set[S.a_now]*S.Y[S.a_now]*S.k(S.a_now, S.a_now) - S.A_set[S.b_now]*S.Y[S.b_now]*S.k(S.b_now, S.a_now)
	S.hb_now = S.gb_now - S.A_set[S.a_now]*S.Y[S.a_now]*S.k(S.a_now, S.b_now) - S.A_set[S.b_now]*S.Y[S.b_now]*S.k(S.b_now, S.b_now)
}

//更新当前a，b对应的A_set参数
func (S *Smo) updata_ab() {
	L := max(S.A_set[S.b_now]-S.A_set[S.a_now], 0)
	H := min(S.C, S.C-S.A_set[S.a_now]+S.A_set[S.b_now])
	L2 := max(S.A_set[S.a_now]+S.A_set[S.b_now]-S.C, 0)
	H2 := min(S.A_set[S.a_now]+S.A_set[S.b_now], S.C)
	denominator := S.k(S.a_now, S.a_now) + S.k(S.b_now, S.b_now) - 1*S.k(S.a_now, S.b_now)
	S.A_set[S.b_now] = S.Y[S.b_now] * (S.Y[S.b_now] - S.Y[S.a_now] + S.u_now*(S.k(S.a_now, S.a_now)-S.k(S.b_now, S.b_now)) + S.ha_now - S.hb_now) / denominator
	S.A_set[S.a_now] = S.Y[S.a_now] * (S.Y[S.a_now] - S.Y[S.b_now] + S.u_now*(S.k(S.b_now, S.b_now)-S.k(S.a_now, S.a_now)) + S.ha_now - S.hb_now) / denominator
	if S.A_set[S.b_now] > H {
		S.A_set[S.b_now] = H
	}
	if S.A_set[S.b_now] < L {
		S.A_set[S.b_now] = L
	}
	if S.A_set[S.a_now] > H2 {
		S.A_set[S.a_now] = H2
	}
	if S.A_set[S.a_now] < L2 {
		S.A_set[S.a_now] = L2
	}

}

func max(a float64, b float64) float64 {
	if a >= b {
		return a
	} else {
		return b
	}
}

func min(a float64, b float64) float64 {
	if a >= b {
		return b
	} else {
		return a
	}
}

func (S *Smo) Get_A_set(MAX_iter int) []float64 {
	S.init_()
	for j := 0; j < MAX_iter; j++ {
		for i := 0; i < S.N_size-1; i++ {
			S.chang_ab(i, i+1)
			S.new_parameter()
			S.updata_ab()

		}
	}
	return S.A_set
}
