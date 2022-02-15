package entropy

import (
	"math"
)

//计算float64切片的熵
func En_slice(data []float64) float64 {
	p_map := make(map[float64]float64, 0)
	for i := 0; i < len(data); i++ {
		if p_map[data[i]] == 0 {
			p_map[data[i]] = 1
		} else {
			p_map[data[i]] += 1
		}
	}
	for k, v := range p_map {
		p_map[k] = v / float64(len(data))
	}
	entropy := float64(0)
	for _, v := range p_map {
		entropy -= v * math.Log(v)
	}
	return entropy

}

//用于计算string切片的熵
func En_slice_string(data []string) float64 {
	p_map := make(map[string]float64, 0)
	for i := 0; i < len(data); i++ {
		if p_map[data[i]] == 0 {
			p_map[data[i]] = 1
		} else {
			p_map[data[i]] += 1
		}
	}
	for k, v := range p_map {
		p_map[k] = v / float64(len(data))
	}
	entropy := float64(0)
	for _, v := range p_map {
		entropy -= v * math.Log(v)
	}
	return entropy

}

//计算给定特征序列下熵减最大的元素(float 64 切片)
func Max_entropy_loss_feature_float(Y []float64, X []float64) (float64, float64) {
	original_entorpy := En_slice(Y)
	feature_map := make(map[float64][]int, 0)
	for i := 0; i < len(X); i++ {
		if feature_map[X[i]] == nil {
			feature_map[X[i]] = []int{i}
		} else {
			feature_map[X[i]] = append(feature_map[X[i]], i)
		}
	}
	feature_y_map := make(map[float64]([]float64), 0)
	for k, v := range feature_map {
		temp_arr := make([]float64, 0)
		for i := 0; i < len(v); i++ {
			temp_arr = append(temp_arr, Y[v[i]])
		}
		feature_y_map[k] = temp_arr
	}
	max_loss := float64(0)
	max_feature := float64(0)
	for k, v := range feature_y_map {

		if max_loss < original_entorpy-En_slice(v) {
			max_loss = original_entorpy - En_slice(v)
			max_feature = k
		}
	}
	return max_feature, max_loss

}

//计算给定特征序列下熵减最大的元素(string 切片)
func Max_entropy_loss_feature_string(Y []string, X []string) (string, float64) {
	original_entorpy := En_slice_string(Y)
	feature_map := make(map[string][]int, 0)
	for i := 0; i < len(X); i++ {
		if feature_map[X[i]] == nil {
			feature_map[X[i]] = []int{i}
		} else {
			feature_map[X[i]] = append(feature_map[X[i]], i)
		}
	}
	feature_y_map := make(map[string]([]string), 0)
	for k, v := range feature_map {
		temp_arr := make([]string, 0)
		for i := 0; i < len(v); i++ {
			temp_arr = append(temp_arr, Y[v[i]])
		}
		feature_y_map[k] = temp_arr
	}
	max_loss := float64(0)
	max_feature := ""
	for k, v := range feature_y_map {
		if max_loss < original_entorpy-En_slice_string(v) {
			max_loss = original_entorpy - En_slice_string(v)
			max_feature = k
		}
	}
	return max_feature, max_loss

}

//计算给定特征序列下熵减比率最大的元素(string 切片)
func Max_c45_feature_string(Y []string, X []string) (string, float64) {
	original_entorpy := En_slice_string(Y)
	feature_map := make(map[string][]int, 0)
	for i := 0; i < len(X); i++ {
		if feature_map[X[i]] == nil {
			feature_map[X[i]] = []int{i}
		} else {
			feature_map[X[i]] = append(feature_map[X[i]], i)
		}
	}
	feature_y_map := make(map[string]([]string), 0)
	for k, v := range feature_map {
		temp_arr := make([]string, 0)
		for i := 0; i < len(v); i++ {
			temp_arr = append(temp_arr, Y[v[i]])
		}
		feature_y_map[k] = temp_arr
	}
	max_loss := float64(0)
	max_feature := ""
	for k, v := range feature_y_map {
		if max_loss < (original_entorpy-En_slice_string(v))/En_slice_string(X) {
			max_loss = (original_entorpy - En_slice_string(v)) / En_slice_string(X)
			max_feature = k
		}
	}
	return max_feature, max_loss

}

//计算当前长度下熵的最大值
func Max_limit(data []float64) float64 {
	return math.Log(float64(len(data)))
}
