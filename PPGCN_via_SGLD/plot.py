import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import pickle
import torch


def result_query(lr,std,C):
    pkl_file = open('pickle/'+ 'lr='+str(lr)+', std='+str(std)+', C='+str(C),'rb')
    saved_result = pickle.load(pkl_file)
    return saved_result




def plot_multi_line(lr1,std1,C1, lr2,std2,C2, lr3,std3,C3, lr4,std4,C4,type):
	result1 = result_query(lr1, std1, C1)
	result2 = result_query(lr2, std2, C2)
	result3 = result_query(lr3, std3, C3)
	result4 = result_query(lr4, std4, C4)	#plt.rcParams['font.size'] = 13

	plt.figure()
	plt.plot(result1[type],label='lr='+str(lr1)+', std='+str(std1)+', C='+str(C1)+', $\epsilon$='+str(round(torch.tensor(result1['eps']).mean().item(),4)))
	plt.plot(result2[type],label='lr='+str(lr2)+', std='+str(std2)+', C='+str(C2)+', $\epsilon$='+str(round(torch.tensor(result2['eps']).mean().item(),4)))
	plt.plot(result3[type],label='lr='+str(lr3)+', std='+str(std3)+', C='+str(C3)+', $\epsilon$='+str(round(torch.tensor(result3['eps']).mean().item(),4)))
	plt.plot(result4[type],label='lr='+str(lr4)+', std='+str(std4)+', C='+str(C4)+', $\epsilon$='+str(round(torch.tensor(result4['eps']).mean().item(),4)))
	plt.ylabel('validation accuracy')
	plt.xlabel('epoch')
	plt.legend()
	plt.show()

#plot_multi_line(0.01,0.1,0.3, 0.01,0.05,0.3, 0.01,0.01,0.3, 0.01,0.001,0.3,'val_acc')

def query_test_acc(lr1,std1,C1):
	result1 = result_query(lr1, std1, C1)
	print(result1['test_acc'])


query_test_acc(0.01,0.01,0.3)


#lr_list=[0.1,0.01,0.05,0.001]
#std = [0.1,0.01,0.05,0.001]

def plot_scatter(hyper_param_list,best_param):
	plt.figure()

	for hyper_param in hyper_param_list:
		if (hyper_param[0], hyper_param[1], hyper_param[2]) != best_param:
			result = result_query(hyper_param[0], hyper_param[1], hyper_param[2])
			name = 'lr='+str(hyper_param[0])+', std='+str(hyper_param[1])+', C='+str(hyper_param[2])

			plt.scatter(round(torch.tensor(result['eps']).mean().item(),4), result['test_acc'],marker='o',label=name)
		else:
			result = result_query(hyper_param[0], hyper_param[1], hyper_param[2])
			name = 'lr='+str(hyper_param[0])+', std='+str(hyper_param[1])+', C='+str(hyper_param[2])
			plt.scatter(round(torch.tensor(result['eps']).mean().item(),4), result['test_acc'], s=130 ,marker='*',label=name)

		#plt.text(round(torch.tensor(result['eps']).mean().item(),4), result['test_acc'],name)
	plt.xlabel('privacy budget $\epsilon$')
	plt.ylabel('test accuracy')
	plt.legend()
	plt.show()

hyper_param_list=[(0.01,0.1,0.3), (0.01,0.01,0.3), (0.01, 0.05, 0.3), (0.01,0.001,0.3), (0.1,0.01,0.3),(0.05,0.01,0.3),(0.001,0.01,0.3)]
best_param = (0.01,0.01,0.3)

plot_scatter(hyper_param_list,best_param)
