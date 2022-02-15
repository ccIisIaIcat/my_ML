package math_tool

import (
	"fmt"
	"strconv"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

//计算多元正太函数在某一分位点上num_o个西格玛的累计分布值
func Calculate_mul_prob_value(u []float64, sigma []float64, point []float64, K int, num_o float64) float64 {

	sigma_m := mat.NewSymDense(len(u), sigma)
	new_n, ok := distmv.NewNormal(u, sigma_m, nil)
	if !ok {
		fmt.Println("错误")
	}
	do := make([]float64, 0)
	for i := 0; i < len(u); i++ {
		do = append(do, sigma_m.At(i, i)*num_o/float64(K))
	}
	dv, new_mash := make_mash(point, do, K)
	point_set := All_array_list_2(K, len(u))
	answer := float64(0)
	for i := 0; i < len(point_set); i++ {
		new_point := make([]float64, len(u))
		for j := 0; j < len(u); j++ {
			new_point[j] = new_mash[j][point_set[i][j]]
		}
		f := new_n.Prob(new_point)
		answer += f * dv
	}

	return answer

}

//生成含n个重复元素的k元全排列
func All_array_list_2(n int, k int) [][]int {
	return delete_the_same(all_sort(n, k))
}

//生成含n个重复元素的n元全排列
func All_array_list(n int) [][]int {
	return delete_the_same(all_sort(n, n))
}

func make_mash(point []float64, dl []float64, each_dimension int) (float64, [][]float64) {
	dv := float64(1)
	for i := 0; i < len(dl); i++ {
		dv = dv * dl[i]
	}
	new_mash := make([][]float64, len(dl))
	for i := 0; i < len(dl); i++ {
		for j := 0; j < each_dimension; j++ {
			tool_mash := make([]float64, len(new_mash[i]))
			copy(tool_mash, new_mash[i])
			new_mash[i] = append(tool_mash, point[i]-dl[i]*float64(j))
		}
	}
	return dv, new_mash

}

func all_sort(n int, l int) [][]int {
	answer := make([][]int, 0)
	if l == 1 {
		for i := 0; i < n; i++ {
			answer = append(answer, []int{i})
		}
		return answer
	} else {
		tool_mat := all_sort(n, l-1)
		tool_mat = delete_the_same(tool_mat)
		for i := 0; i < len(tool_mat); i++ {
			for j := 0; j < n; j++ {
				for k := 0; k < len(tool_mat[0])+1; k++ {
					tool := make([]int, len(tool_mat[i]))
					copy(tool, tool_mat[i])
					temp_arr := insert_in_position(tool, j, k)
					answer = append(answer, temp_arr)

				}
			}
			answer = delete_the_same(answer)

		}
		return answer
	}

}

func insert_in_position(nums []int, point int, position int) []int {

	answer := append(nums[:position], append([]int{point}, nums[position:]...)...)
	return answer
}

func delete_the_same(data [][]int) [][]int {
	tool_map := make(map[string]int)
	for i := 0; i < len(data); i++ {
		tool_map[turn_slice_to_string(data[i])] += 1
	}
	tool_slice := make([][]int, 0)
	for k := range tool_map {
		a := turn_string_to_slice(k)
		tool_slice = append(tool_slice, a)
	}
	return tool_slice
}
func turn_slice_to_string(sample []int) string {
	tool_str := ""
	for i := 0; i < len(sample); i++ {
		tool_str += strconv.Itoa(sample[i])
		tool_str += "A"
	}
	return tool_str
}

func turn_string_to_slice(str string) []int {
	tool_slice := make([]int, 0)
	l_p := 0
	r_p := 0
	for i := 0; i < len(str); i++ {
		r_p = i
		if string(str[i]) == "A" {
			a, _ := strconv.Atoi(string(str[l_p:r_p]))
			tool_slice = append(tool_slice, a)
			l_p = i + 1
		}
	}
	return tool_slice

}

func main() {
	uu := []float64{0, 0}
	mm := []float64{1, 0, 0, 1}
	point := []float64{0, 0}
	answer := Calculate_mul_prob_value(uu, mm, point, 250, 10)
	fmt.Println(answer)
	//fmt.Println(len(All_array_list(4)))
	// aaa := "66A"
	// fmt.Println(turn_string_to_slice(aaa))
}
