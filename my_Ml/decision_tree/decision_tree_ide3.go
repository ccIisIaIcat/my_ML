package decision_tree

import (
	"fmt"
	"my_ML/entropy"
)

type TreeNode struct {
	Judge map[int]string
	P_map map[string]float64
	Left  *TreeNode
	Right *TreeNode
}

type D_tree_id3 struct {
	X             []string //必写
	Y             []string //必写
	L             int      //必写,决策树最大深度
	feature_matix [][]string
	N_size        int
	P_size        int
	root_node     *TreeNode
}

//id3树的初始化
func (D *D_tree_id3) init() {

	D.N_size = len(D.Y)
	D.P_size = len(D.X) / len(D.Y)
	D.feature_matix = make([][]string, D.P_size)
	D.root_node = new(TreeNode)
	for i := 0; i < D.P_size; i++ {
		temp := make([]string, 0)
		for j := 0; j < D.N_size; j++ {
			temp = append(temp, D.X[j*D.P_size+i])
		}
		D.feature_matix[i] = temp
	}

}

//给定标准和数据矩阵找到当前最优维度和最优节点
func (D *D_tree_id3) find_best_demension(target []string, feature [][]string) (int, string) {
	entropy_loss_list := make([]float64, D.P_size)
	best_feature_list := make([]string, D.P_size)
	for i := 0; i < D.P_size; i++ {
		best_feature_list[i], entropy_loss_list[i] = entropy.Max_entropy_loss_feature_string(target, feature[i])
	}
	max_loss := float64(0)
	max_feature := ""
	max_id := 0
	for i := 0; i < D.P_size; i++ {
		if entropy_loss_list[i] > max_loss {
			max_loss = entropy_loss_list[i]
			max_feature = best_feature_list[i]
			max_id = i
		}
	}
	return max_id, max_feature

}

//按照一个决策标准将数据集一分为二
func (D *D_tree_id3) divide_data(target []string, feature [][]string, divider map[int]string) ([]string, [][]string, []string, [][]string) {
	di_id := 0
	di_feature := ""
	for k, v := range divider {
		di_id = k
		di_feature = v
	}
	Y1 := make([]string, 0)
	Y2 := make([]string, 0)
	f1 := make([][]string, D.P_size)
	f2 := make([][]string, D.P_size)
	for i := 0; i < len(target); i++ {
		if feature[di_id][i] == di_feature {
			Y1 = append(Y1, target[i])
			for j := 0; j < D.P_size; j++ {
				f1[j] = append(f1[j], feature[j][i])
			}
		} else {
			Y2 = append(Y2, target[i])
			for j := 0; j < D.P_size; j++ {
				f2[j] = append(f2[j], feature[j][i])
			}
		}
	}
	return Y1, f1, Y2, f2

}

//树的中序遍历
func (D *D_tree_id3) search(node *TreeNode) {
	fmt.Println(node.Judge)
	if node.Left != nil {
		D.search(node.Left)
	}
	if node.Right != nil {
		D.search(node.Right)
	}
}

//用递归方法建立一个决策树
func (D *D_tree_id3) build_a_tree(node_now *TreeNode, target []string, feature [][]string, deep int) {
	a, b := D.find_best_demension(target, feature)
	divider := map[int]string{a: b}
	node_now.Judge = divider
	deep++
	if deep <= D.L && b != "" {
		y1, f1, y2, f2 := D.divide_data(target, feature, divider)
		node_now.Left = new(TreeNode)
		D.build_a_tree(node_now.Left, y1, f1, deep)
		node_now.Right = new(TreeNode)
		D.build_a_tree(node_now.Right, y2, f2, deep)
	} else {
		delete(divider, a)
		tool_map := make(map[string]int, 0)
		max_num := 0
		max_feature := ""
		for k := 0; k < len(target); k++ {
			if tool_map[target[k]] == 0 {
				tool_map[target[k]] = 1
				if tool_map[target[k]] > max_num {
					max_num = tool_map[target[k]]
					max_feature = target[k]
				}
			} else {
				tool_map[target[k]] += 1
				if tool_map[target[k]] > max_num {
					max_num = tool_map[target[k]]
					max_feature = target[k]
				}
			}
		}
		divider[-1] = max_feature
		node_now.P_map = make(map[string]float64, 0)
		for k, v := range tool_map {
			node_now.P_map[k] = float64(v) / float64(len(target))
		}

	}

}

func (D *D_tree_id3) judge_sample_point(node_this *TreeNode, sample_x []string) (string, map[string]float64) {
	temp_id := 0
	temp_feature := ""
	for k, v := range node_this.Judge {
		temp_id = k
		temp_feature = v
	}
	if temp_id == -1 {
		return temp_feature, node_this.P_map
	} else if sample_x[temp_id] == temp_feature {
		a, b := D.judge_sample_point(node_this.Left, sample_x)
		return a, b
	} else {
		a, b := D.judge_sample_point(node_this.Right, sample_x)
		return a, b
	}
}

func (D *D_tree_id3) Fit() {
	D.init()
	D.build_a_tree(D.root_node, D.Y, D.feature_matix, 0)

}

func (D *D_tree_id3) Search() {
	D.search(D.root_node)
}

func (D *D_tree_id3) Make_a_judge(sample_x []string) (string, map[string]float64) {
	result, P_map := D.judge_sample_point(D.root_node, sample_x)
	return result, P_map

}
