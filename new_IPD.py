import os
import numpy as np
from random import sample
import random
import time
import math
from sklearn.neighbors import BallTree
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import patches
from sklearn.utils import shuffle
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import silhouette_score as sil
#import  sklearn.metrics.calinski_harabasz_score as CH
from sklearn import preprocessing
import sys
argumentList = sys.argv[1:]
#os.chdir("/home/jayasree/jayasree/2021/write_Up/DBSCAN_Pattern Recognition/PR/shape/data/3/geojson/data/")
dataset=np.genfromtxt("/home/jayasree/jayasree/2020/Work/codes/CNAK_meets_DBSCAN/datasets/Aggregation.csv", delimiter="\t", usecols=range(2))
#TEST=np.genfromtxt("aqua_test_data.csv", delimiter=",", usecols=range(2))
print("dataset:",dataset.shape)

EPSILON=float( argumentList[0] )
MINPTS=int( argumentList[1] )
THRESHOLD=float( argumentList[2] )
#os.chdir("/home/jayasree/jayasree/2021/write_Up/DBSCAN_Pattern Recognition/PR/shape/data/1/")
PATH="eps="+str(EPSILON)+"_Minpts="+str(MINPTS)+"_threshold="+str(THRESHOLD)
print("PATH:",PATH)
if not os.path.exists(PATH):
	os.mkdir(PATH)
os.chdir(PATH)
start=time.time()

print("EPSILON=", EPSILON)
print("MINPTS:", MINPTS)


def _random_sampling(gamma):
	global unprocessed_points, points_available
	random_points=[]
	
	i=0
	while i < (gamma):
		point=sample(unprocessed_points,1)
		
		if unprocessed_points[int(point[0][0]),1]!=-1:
			random_points.append(int(point[0][0]))
			unprocessed_points[int(point[0][0]),1]=-1
			points_available-=1
			i+=1

	return random_points

def _create_ball_tree(samples):
	global dataset
	map_records={}
	data=[]
	for i in range(len(samples)):
		data.append(dataset[samples[i]])
		map_records[i]=samples[i]
	tree = BallTree(data, leaf_size=2)

	return tree, map_records

def _range_query(p, tree, data_records_index):
	global visited, neighbor_map
	visited[p]=True
	neighbor_p=tree.query_radius(dataset[p:p+1], r=EPSILON)
	for item in neighbor_p[0]:
		neighbor_map[p].add(data_records_index[item])
		neighbor_map[data_records_index[item]].add(p)
		
	return 

def _range_query_inc(p, tree, data_records_index):
	global  neighbor_map
	visited[p]=True
	neighbor_p=tree.query_radius(dataset[p:p+1], r=EPSILON)

	for item in neighbor_p[0]:
		neighbor_map[p].add(data_records_index[item])
		neighbor_map[data_records_index[item]].add(p)
		
	return 

def _estimate_eta():
	global s_prototype
	size=int(np.ceil(len(s_prototype)*0.3))
	points=sample(s_prototype,size)
	#print("points:",len(points))
	stat=[]
	for p in points:
		length=0
		for i in range(len(tree_list)):
			neighbor_p=tree_list[i].query_radius(dataset[p:p+1], r=EPSILON)
			length+=len(neighbor_p[0])
		
		stat.append(length)
	#print("eta now:", int(np.mean(stat)))
	return int(np.mean(stat))

def _dbscan(s_prototype, minpts):
	global visited

	cluster_id=0
	for i in range(len(s_prototype)):
		p=s_prototype[i]
		if visited[p]==False:
			
			if _expand_cluster(p, cluster_id, minpts):
				cluster_id+=1
	print("dbscan completed")
	return cluster_id

def _expand_cluster(p, cluster_id, minpts):
	global neighbor_map, type_list, clusters, cores, borders, noises
	

	_range_query(p, tree_list[0], map_records_list[0])
	visited[p]=True
	if len(neighbor_map[p])<minpts:
		type_list[p]="noise"
		labels[p]=-1
		if not (-1 in clusters.keys()):
			clusters[-1]=set()
		clusters[-1].add(p)
		noises.add(p)
		return False
	else:
		
		labels[p]=cluster_id
		type_list[p]="core"	
		cores.add(p)

		if not (cluster_id in clusters.keys()):
			clusters[cluster_id]=set()
		clusters[cluster_id].add(p)

		seeds=[]
		for j in neighbor_map[p]:
			seeds.append(j)
			visited[j]=True

		for q in seeds:
			_range_query(q, tree_list[0], map_records_list[0])
			
			clusters[cluster_id].add(q)
			labels[q]=cluster_id
			if len(neighbor_map[q])>=minpts:
				type_list[q]="core"
				cores.add(q)
				for item in neighbor_map[q]:
					if not (item in seeds):
						if (visited[item]==False) or (type_list[item]=="noise"):
							if (visited[item]==False):
								seeds.append(item)
								visited[item]=True
							else:
								type_list[item]="border"
								clusters[cluster_id].add(item)
								if (-1 in clusters.keys()) and (item in clusters[-1]):
									clusters[-1].remove(item)
								labels[item]=cluster_id
								borders.add(item)
								if item in noises:
									noises.remove(item)
								
			else:
				type_list[q]="border"
				borders.add(q)
				clusters[cluster_id].add(q)
				labels[q]=cluster_id
								
		return True
def compute_min_dist(J,selected_items):
	min_dist=12345
	for item in selected_items:
		tmp=np.linalg.norm(dataset[item]-dataset[J])
		if tmp<min_dist:
			min_dist=tmp
	return min_dist
def _limit_representatives(base, item):
	x=0
	index=np.zeros(len(base))
	selected_items=[]
	selected_items.append(item)
	Distances=np.zeros([len(base)+1, len(base)+1])
	
	for i in range(len(base)):
		for j in range(i+1,len(base)):
			Distances[i,j]=np.linalg.norm(dataset[base[i]]-dataset[base[j]])
			Distances[j,i]=Distances[i,j]
	for i in range(len(base)):
		Distances[i,len(base)]=np.linalg.norm(dataset[item]-dataset[base[i]])
		Distances[len(base),i]=Distances[i,len(base)]

	k=np.argmax(Distances[len(base), :])
	selected_items.append(base[k])
	index[k]=-1
	while  x<50:
		flag=True
		gain_index=[]
		gain=[]
		for i in range(len(base)):
			I=base[i]
			gain_i=0
			if index[i]!=-1:
				for j in range(len(base)):
					if index[j]!=-1 and j!=i:
						J=base[j]
						
						D_j=compute_min_dist(J,selected_items)

						d_ij= Distances[i,j]
						C_ji=max(D_j-d_ij, 0)
						gain_i+=C_ji
					
			
				gain.append(gain_i)
				gain_index.append(i)
		
		k=np.argmax(gain)
		selected_items.append(base[gain_index[k]])
		x+=1
	return selected_items

def _identify_representatives():
	global neighbor_map,  cores, labels, representatives, eta, clusters
	representatives={}
	max_density={}
	max_density_point_index={}
	for x in (cores):
		if not (labels[x]  in max_density.keys()):
			max_density[labels[x]]=0 
			max_density_point_index[labels[x]]=-1
		if len(neighbor_map[x]) > max_density[labels[x]]:	
			max_density[labels[x]]=len(neighbor_map[x])
			max_density_point_index[labels[x]]=x

	print("Cores:",len(cores))
	for key in clusters.keys():
		if key!=-1 and len(clusters[key])>0:
			for item in clusters[key]:
				if type_list[item]=="core":
					if labels[item] not in representatives.keys():
						representatives[labels[item]]=set() 
					if np.true_divide(len(neighbor_map[item]),max_density[labels[item]])<THRESHOLD:
						representatives[labels[item]].add(item)

					if len(neighbor_map[item])==max_density[labels[item]]:
						representatives[labels[item]].add(item)

	
	for key in representatives.keys():
		
		if len(representatives[key])>50:

			repr=sample(list(representatives[key]),50)
			'''
			print("before:",key,"---->",len(representatives[key]))
			repr=_limit_representatives(list(representatives[key]), max_density_point_index[key])
			'''
			representatives[key]=set()
			for item in repr:
				representatives[key].add(item)
			#print("after:",key,"---->",len(representatives[key]))
			
	return 

def _identify_representatives_old():
	global neighbor_map,  cores, labels, representatives, eta
	representatives={}
	max_density={}
	for x in (cores):
		if not (labels[x]  in max_density.keys()):
			max_density[labels[x]]=0 
		if len(neighbor_map[x]) > max_density[labels[x]]:	
			max_density[labels[x]]=len(neighbor_map[x])
		
	print("Cores:",len(cores))
	#print("current minpts:",eta)
	for x in (cores):
			
		if labels[x] not in representatives.keys():
			representatives[labels[x]]=set()
		#print("threshold:",np.true_divide(len(neighbor_map[x]),max_density[labels[x]]))
		if np.true_divide(len(neighbor_map[x]),max_density[labels[x]])<THRESHOLD:
			representatives[labels[x]].add(x)
				
				
		if len(neighbor_map[x])==max_density[labels[x]]:
			representatives[labels[x]].add(x)
			
	#print("_identify_representatives():",representatives.keys())
	return 

def compute_test_size(s_prototype):
	global clusters
	K=0
	#n=50000
	
	for j in clusters.keys():
		if len(clusters[j])>0:
			K+=1
	
	if K==0:
		K=1
	print("K:",K)
	
	#K=50
	#a=np.power(np.true_divide(1,np.sqrt(K)),np.true_divide(1,K))
	frac=np.true_divide(1,(2*len(dataset))) +np.sqrt(np.true_divide(K,len(dataset)))
	#frac_test=np.true_divide(1,(2*n)) +np.sqrt(np.true_divide(K,n))
	#print("test frac:",frac)
	#print("test frac:",frac_test)
	#print("test size:",int(np.ceil(frac_test*n)))
	print("test size:",int(np.ceil(frac*len(dataset))))
	#return int(np.ceil(frac_test*n)), K

	return int(np.ceil(frac*len(dataset))), K


def _label_test_samples(test):
	global representatives
	#total_repr=0
	test_labels=[]
	for i in range(len(test)):
		min_dist=np.Inf
		min_index=-1
		#print("representatives.keys():",representatives.keys())
		total_repr=0
		#print("length:",len(representatives.keys()))
		for key in representatives.keys():
			if key!=-1:
				
				total_repr+=len(representatives[key])
				for rep in representatives[key]:
					dist=np.linalg.norm(dataset[test[i]]-dataset[rep])
					#print("dist=",dist)
					if dist<min_dist:
						min_index=rep
						min_dist=dist
		#print("labels[min_index]:",min_index)
		test_labels.append(labels[min_index])
		labels[test[i]]=labels[min_index]
	print("total_repr:",total_repr)

	return test_labels#, total_repr


def _inc_dbscan(s_inc, eta, next_id):
	global visited
	
	for p in (s_inc):
		if visited[p]==False:
			flag, C= _inc_expand_cluster(p, next_id, eta)
			if flag:
				if C==next_id:
					next_id+=1

	return next_id

def _inc_expand_cluster(p, next_id, minpts):
	
	global type_list, cores, borders, noises, labels, visited, inserted, clusters
	
	for i in range(len(tree_list)):
		_range_query_inc(p, tree_list[i], map_records_list[i])
		
	_range_query(p, tree_inc, map_records_inc)
	visited[p]=True

	if len(neighbor_map[p]) < minpts:
		
		if  _is_noise(p, minpts): 					#p is a noise
			
			noises.add(p)
			type_list[p]="noise"
			labels[p]=-1
			if not (-1 in clusters.keys()):
				clusters[-1]=set()
			clusters[-1].add(p)
			
			return False, -1

		#else: 
			#print("it could be a border")	
			

	
	else:
		
		clusters_to_be_merged=set()
		
		type_list[p]="core"
		cores.add(p)		#updated cores --> +1 addition, -1 deletion
		
		labels[p]=next_id
		
		if not (next_id in clusters.keys()):
			clusters[next_id]=set()	
		clusters[next_id].add(p)
		
		seed=[]
		
		for item in neighbor_map[p] :
			if visited[item]==False :
				if inserted[item]==False:
					seed.append(item)
					inserted[item]=True
			else:
				if type_list[item]=="noise":
					labels[item]=next_id
					clusters[next_id].add(item)
					if (-1 in clusters.keys()) and (item in clusters[-1]):
						clusters[-1].remove(item)
					if (-1 in clusters.keys()) and len(clusters[-1])<=0:
						del clusters[-1]
					if len(neighbor_map[item])>=minpts: #noise to core transition
						type_list[item]="core"
						cores.add(item)
					else:
						type_list[item]="border"
						borders.add(item)
					noises.remove(item)
				
				clusters_to_be_merged.add(labels[item])
		
		for q in seed:
			
			for i in range(len(tree_list)):
				_range_query_inc(q, tree_list[i], map_records_list[i])
		
			_range_query(q, tree_inc, map_records_inc)


			if len(neighbor_map[q])>=minpts:
				
				cores.add(q)
				
				type_list[q]="core"
				labels[q]=next_id
				for item in neighbor_map[q] :
					if (visited[item]==False) :
						if inserted[item]==False:
							seed.append(item)
							inserted[item]=True
						
					else:
						"""it is about processed items"""
						if type_list[item]=="noise":
							labels[item]=next_id
							clusters[next_id].add(item)
							if (-1 in clusters.keys()) and (item in clusters[-1]):
								clusters[-1].remove(item)
							if (-1 in clusters.keys()) and len(clusters[-1])<=0:
								del clusters[-1]
							if len(neighbor_map[item])>=minpts: #noise to core transition
								type_list[item]="core"
								cores.add(item)
							else:
								type_list[item]="border"    #noise to border transition
								borders.add(item)
							noises.remove(item)

						if type_list[item]=="border":
							if len(neighbor_map[item])>=minpts: #border to core transition
								type_list[item]="core"
								cores.add(item)
								borders.remove(item)
							labels[item]=next_id
							
						clusters_to_be_merged.add(labels[item])


			else:
				borders.add(q)
				type_list[q]="border"
				labels[q]=next_id
				

			
			
			clusters[next_id].add(q)
		clusters_to_be_merged.add(next_id)

		if len(clusters_to_be_merged)>1:
			next_id=_merge(clusters_to_be_merged, next_id)
			
						
		return True, next_id
	

def _merge(clusters_to_be_merged,cluster_id):
	global clusters, labels
	#print("clusters_to_be_merged",clusters_to_be_merged)
	#cluster_id=min(clusters_to_be_merged)
	tmp=np.sort(np.asarray(list(clusters_to_be_merged)))
	i=0
	cluster_id=-1
	while cluster_id==-1:
		cluster_id=tmp[i]
		i+=1
	#print("merge cluster into",cluster_id)
	for k in clusters_to_be_merged:
		if not (k==cluster_id):
		   if (k in clusters.keys()) and (len(clusters[k])>0):
			for item in clusters[k]:
				if not (cluster_id in clusters.keys()):
					clusters[cluster_id]=set()
				clusters[cluster_id].add(item)
				labels[item]=cluster_id
			
	
	clusters_to_be_merged.remove(cluster_id)
	
	for k in clusters_to_be_merged:
		if (k in clusters.keys()):
			del clusters[k]
		
	return cluster_id

def _verify_core_states(minpts):
	global cores, borders, noises, neighbor_map, type_list, clusters,labels
	remove=[]
	for item in cores:
		if len(neighbor_map[item])<minpts:
			remove.append(item)
	
	for item in remove:
		cores.remove(item)
		
		
		flag=True
		for q in neighbor_map[item]:
			if len(neighbor_map[q])>=minpts:
				flag=False
				type_list[item]="border"			#core-->border transition
				borders.add(item)
				labels[item]=labels[q] #newly added
				break
		if flag:
			if (labels[item] in clusters.keys()) :
				if item in clusters[labels[item]]:
					clusters[labels[item]].remove(item)

				if len(clusters[labels[item]])<=0:
					del clusters[labels[item]]
			noises.add(item)
			type_list[item]="noise"					#core--->noise transition
			labels[item]=-1
			if not (-1 in clusters.keys()):
				clusters[-1]=set()
			clusters[-1].add(item)
			
		
		_recursive_phenomena_checking(item, minpts) 
	
	return 

def _recursive_phenomena_checking(p, minpts):
	global borders, noises, neighbor_map, type_list, clusters
	
	for item in  neighbor_map[p]:
		if type_list[item]=="border":
			flag=False
			for x in neighbor_map[item]:
				if len(neighbor_map[x])>=minpts:
					flag=True
					break
			if flag==False:                                                 #cascading effect
				type_list[item]="noise"					#border-->noise transition
				borders.remove(item)
				noises.add(item)
				if (labels[item] in clusters.keys()) :

					if item in clusters[labels[item]]:
						clusters[labels[item]].remove(item)
					if len(clusters[labels[item]])<=0:
						del clusters[labels[item]]
				labels[item]=-1
				if -1 not in clusters.keys():
					clusters[-1]=set()
				clusters[-1].add(item)
				
	return

def _is_noise(p, minpts):
	global neighbor_map, labels, type_list

	for item in neighbor_map[p]:
		if len(neighbor_map[item])>=minpts:
			#print("is noise:",item,"labels[item]:",labels[item])
			return False, #labels[item]

	return True#, -1

def _compute_stability(test_labels, new_test_labels):
    mismatched_decision=0
    #print("test_labels:",len(test_labels))
    for i in range(len(test_labels)):
        for j in range(i+1, len(test_labels)):
            if test_labels[i]==test_labels[j]:
		if new_test_labels[i]!=new_test_labels[j]:
                	mismatched_decision+=1
             
            if test_labels[i]!=test_labels[j]:
		if new_test_labels[i]==new_test_labels[j]:
                	mismatched_decision+=1   
    
    fact = math.factorial
    fact(2)
    
    a=fact(len(test_labels))
    b=fact(len(test_labels)-2)*fact(2)
    denom=a/b
    
    return np.true_divide(mismatched_decision, denom)


def _label_test_points(itr):
	global representatives, labels
	test_labels=np.zeros(len(TEST))
	for i in range(len(TEST)):
		
			min_dist=np.Inf
			nearest_core=-1
			for key in representatives.keys() :
				for rep in representatives[key]:
					dist=np.linalg.norm(TEST[i]-dataset[rep])
					if dist<min_dist:
						nearest_core=rep
						min_dist=dist
	
			test_labels[i]=labels[nearest_core]
	np.savetxt("aqua_PREDICTED_test_labels_"+str(itr)+".csv",test_labels, delimiter=",")
	return	test_labels



def _label_unprocessed_points():
	global representatives, unprocessed_points, labels
	
	for i in range(len(unprocessed_points)):
		if unprocessed_points[i,1]!=-1:
			min_dist=np.Inf
			nearest_core=-1
			for key in representatives.keys() :
				for rep in representatives[key]:
					dist=np.linalg.norm(dataset[unprocessed_points[i,0]]-dataset[rep])
					if dist<min_dist:
						nearest_core=rep
						min_dist=dist
		
			labels[unprocessed_points[i,0]]=labels[nearest_core]
	return	



def _noise_estimate(x_h):
	global representatives, noises, labels
	L=list(representatives.keys())
	distribution=[]
	c=0
	for x in noises:
		repr_class, dist=_get_nearest_representative(x)
		#hausdorff_dist=_get_nearest_inter_cluster_distance(repr_class)
		#print("inter cluster separation:",hausdorff_dist)
		#print("ratio:",np.true_divide(dist, hausdorff_dist))
		#print("*"*50)
		'''
		if dist <=np.true_divide(hausdorff_dist,2):
			labels[x]=repr_class
		'''
		if round(np.true_divide(dist, 1),2)<=x_h:
			labels[x]=repr_class
			c+=1

		
	#print("#noise deleted:",c)
	return
def _confidence_interval():
	global borders
	samples=[]
	for x in borders:
		repr_class, dist=_get_nearest_representative(x)
		#hausdorff_dist=_get_nearest_inter_cluster_distance(repr_class)
		#samples.append(np.true_divide(dist,hausdorff_dist))
		samples.append(dist)
	#99% confidence interval

	mean=np.mean(samples)
	sigma=np.std(samples)
	#print("sigma:", sigma)
	x_h=mean+3.29*np.true_divide(sigma,np.sqrt(len(samples))) 
	x_l=mean-3.29*sigma #np.true_divide(sigma,np.sqrt(len(samples)))

	#print("confidence interval:",[x_l, round(x_h, 2)])

	return x_l, round(x_h, 2)
def scatter_plot(data_index, itr):
	plt.figure()
	colors=["aqua","chartreuse","darkgreen","gold","springgreen","saddlebrown","khaki","teal","magenta","maroon","tan",
        "olive","purple","steelblue", "rosybrown", "lightcoral", "sandybrown", "bisque", "navajowhite", "orange", "wheat",
         "forestgreen", "springgreen", "darkslategrey", "cadetblue", "mediumorchid", "crimson", "palegreen", "powderblue", "navy", "plum"]
	d=len(colors)
	color_map={}
	k=0
	#print("clusters:",clusters.keys())
	for key in clusters.keys():
			if key!=-1:
				color_map[key]=k
				k+=1

	print("right clusters:",k)
	'''
	c=0
	for i in (data_index):
			if labels[i]==-1:
				plt.scatter(dataset[i,0], dataset[i,1], c="black",marker="x")
			elif (labels[i]==-2):
				c+=1
			else:
				#print("colors[color_map[labels[i]]]:",labels[i])
				plt.scatter(dataset[i,0], dataset[i,1], c=colors[color_map[labels[i]]], marker="o")
		
	for key in representatives:
		for i in representatives[key]:
			plt.scatter(dataset[i,0], dataset[i,1], c="red",marker="x")
	'''
	#print("-2 count=", c)
	#plt.savefig("final_"+str(itr)+".png")
	
	#plt.show()
	return k
def S_prototype_scatter_plot(s_prototype, title):
	plt.figure()
	colors=["aqua","chartreuse","darkgreen","gold","springgreen","saddlebrown","khaki","teal","magenta","maroon","tan",
        "olive","purple","steelblue", "rosybrown", "lightcoral", "sandybrown", "bisque", "navajowhite", "orange", "wheat",
         "forestgreen", "springgreen", "darkslategrey", "cadetblue", "mediumorchid", "crimson", "palegreen", "powderblue", "navy", "plum"]
	d=len(colors)
	color_map={}
	color_tray={}
	k=0
	#print("clusters:",clusters.keys())
	for key in clusters.keys():
		#red=random.uniform(0, 1)
		#green=random.uniform(0, 1)
		#blue=random.uniform(0, 1)
		color_map[key]=k
		#color_tray[key]=(red,green,blue)
		#print("red:",red,"green:",green,"blue:",blue)
		k+=1
	
	c=0
	
	for i in (s_prototype):

			if labels[i]==-1:
				plt.scatter(dataset[i,0], dataset[i,1], c="black",marker="x")
			elif (labels[i]==-2):
				c+=1
			else:
				try:
					plt.scatter(dataset[i,0], dataset[i,1], c=colors[color_map[labels[i]]], marker="o")
				except:
					s=0
				
	
	for key in representatives:
		for i in representatives[key]:
			plt.scatter(dataset[i,0], dataset[i,1], c="red",marker="x")
			#_p=patches.Circle((dataset[i,0],dataset[i,1]), radius=EPSILON, fill=False, facecolor=None, edgecolor=color_tray[labels[i]])
			#plt.gca().add_patch(_p)

	print("-2 count=",c)
	#plt.title(title)
	plt.savefig(title+".png")
	#plt.show()
	return


def _save_partition(itr):
	results=[]
	for i in range(len(dataset)):	
		tmp=[]
		tmp.append(labels[i])
		if type_list[i]=="core":
			tmp.append(1)	
		elif type_list[i]=="border":
			tmp.append(2)
		elif type_list[i]=="noise":
			tmp.append(-1)
		else:
			tmp.append(0)
		results.append(tmp)
	#ni=quality_checking(results)
	file1="RISCAN_partition_"+str(itr)+".csv"
	np.savetxt(file1, np.asarray(results), delimiter=",")
	return 

def get_CH(s_prototype):
	global labels
	count_noise=0
	data=[]
	temp_labels=[]
	for item in s_prototype:
		if labels[item]>=0:
			data.append(dataset[item])
			temp_labels.append(labels[item])
		if labels[item]==-1:
			count_noise+=1

	return sil(data,temp_labels), count_noise


def _noise_estimate(x_h):
	global representatives, noises, labels
	L=list(representatives.keys())
	distribution=[]
	c=0
	for x in noises:
		repr_class, dist=_get_nearest_representative(x)
		#hausdorff_dist=_get_nearest_inter_cluster_distance(repr_class)
		#print("inter cluster separation:",hausdorff_dist)
		#print("ratio:",np.true_divide(dist, hausdorff_dist))
		#print("*"*50)
		'''
		if dist <=np.true_divide(hausdorff_dist,2):
			labels[x]=repr_class
		'''
		if round(np.true_divide(dist, 1),2)<=x_h:
			labels[x]=repr_class
			c+=1

		
	#print("#noise deleted:",c)
	return

def _get_nearest_representative(x):
	global representatives
	min_dist=np.Inf
	label=-1
	for key in representatives.keys():
		for p in representatives[key]:
			tmp=np.linalg.norm(dataset[p]-dataset[x])
			if tmp<min_dist:
				min_dist=tmp
				label=key

	return label, min_dist


def riscan(itr):
	start=time.time()
	global neighbor_map, type_list, labels, visited, inserted
	neighbor_map={}		#This will map a point to its epsilon neighbours 
	type_list={}		#It will capture the state of a point during the execution
	labels={} 		#It captures the clustering label of a point 
	visited={} 		#It depicts whether a point is processed 
	inserted={}

	global cores, added_cores, deleted_cores, borders, noises, eta
	cores=set()
	added_cores=set()
	deleted_cores=set()		
	borders=set()
	noises=set()
	
	global representatives, clusters
	representatives={}	#It captures  representatives for each cluster based on threshold
	clusters={}		

	global convergence_stat
	convergence_stat=[]
	
	"""parameter initialization"""
	
	
	beta=50#int(np.ceil(len(dataset)*0.02))
	gamma=50#int(np.ceil(len(dataset)*0.05))
	#print("gamma:",gamma)


	"""List initialization"""
	global unprocessed_points, points_available, s_prototype
	unprocessed_points=np.zeros((len(dataset),2), dtype=np.int)
	points_available=len(dataset)
	for i in range(len(dataset)):
		unprocessed_points[i,0]=i
		labels[i]=-2
		neighbor_map[i]=set() 
		type_list[i]="unknown" 
		visited[i]=False 
		inserted[i]=False 
	s_prototype=_random_sampling(gamma)
	#print(s_prototype)
	
	"""indexing the sample to get epsilon neighbors"""
	global tree_list, map_records_list
	tree_list=[]
	map_records_list=[]

	tree, map_records=_create_ball_tree(s_prototype)
	tree_list.append(tree)
	map_records_list.append(map_records)
	#print("check uery radius",tree_list[0].query_radius(dataset[2:3], r=1))
	eta=2
	eta_tmp=_estimate_eta()
	if (eta_tmp>eta) and (eta_tmp<MINPTS) :
		eta=eta_tmp
	
	"""generate cluster structure for the intial prototype"""
	next_id=_dbscan(s_prototype, eta)
	"""For validating cluster structure in prototype test dataset is prepared"""
	alpha, K=compute_test_size(s_prototype)
	s_test=_random_sampling(alpha)
	

	"""Labelling test samples"""
	_identify_representatives()
	
	if next_id==0: # No cluster found by DBSCAN
		test_labels=np.asarray(s_test)
		test_labels.fill(-1)
		#representatives=[]
	
	else:
		test_labels=_label_test_samples(s_test)
	
	""" Incrementlly update cluster structure of the prototype"""
	delta=np.Inf
	iteration=1
	global tree_inc, map_records_inc 
	g=0 
	while((delta>0.0) and ( points_available>0)):	
		#print("within while")
		
		tmp_convergence=np.zeros(7,dtype=float)
		tmp_convergence[0]=g
		g+=1
		
		print("eta:",eta)
		#print("MinPts:",MINPTS)
		#start=time.time()
		if eta<MINPTS:
			eta=eta+1
			eta_tmp=_estimate_eta()
			if (eta_tmp>eta) and (eta_tmp<MINPTS) :
				eta=eta_tmp
			#elif (eta<MINPTS) and (eta_tmp>MINPTS):
			#	eta=eta+1 
			_verify_core_states(eta)
		#stop=time.time()
		#print("_verify_core_states:", stop-start)	
		if beta >  points_available :
			 beta=points_available
		#start=time.time()
		s_inc=_random_sampling(beta)	
		#stop=time.time()
		#print("sampling time:", stop-start)	
		#update prototype with  new samples
		#start=time.time()
		tree_inc, map_records_inc=_create_ball_tree(s_inc)
		next_id=_inc_dbscan(s_inc, eta, next_id)			#incrementally update prototype
		#print("labels after increment :",np.unique(labels.values()))
		#stop=time.time()
		#print("create ball tree+inc_dbscan time:", stop-start)
		#start=time.time()
		s_prototype.extend(s_inc)
		K=_identify_representatives()
		#stop=time.time()
		#print("identify representatives time:", stop-start)
		'''
		for k in representatives.keys():
			for item in representatives[k]:
				if labels[item]==-1:
		'''		
		#print("item's label:",labels[item])
		#start=time.time()
		new_test_labels=_label_test_samples(s_test)
		#stop=time.time()
		#print("_label_test_samples :", len(new_test_labels))
		#start=time.time()
		delta=_compute_stability(test_labels, new_test_labels)
		#stop=time.time()
		#print("compute stability time:", stop-start)
		tree_list.append(tree_inc)
		map_records_list.append(map_records_inc)
		iteration+=1
		#start=time.time()
		
		tmp_convergence[1]=len(test_labels)
		tmp_convergence[2]=delta
		tmp_convergence[3], tmp_convergence[4]=get_CH(s_prototype)
		tmp_convergence[5]=len(s_prototype)
		tmp_convergence[6]=K
		convergence_stat.append(tmp_convergence)
		
		alpha_new, K=compute_test_size(s_prototype)
		#stop=time.time()
		#print("compute_test_size time:", stop-start)
		if (alpha_new-alpha)>0:
			if ((alpha_new-alpha) < points_available):
				new_test=_random_sampling(alpha_new-alpha)
				
				alpha_new=alpha

				additional_test_labels=_label_test_samples(new_test)
			
				new_test_labels.extend(additional_test_labels)
				test_labels=new_test_labels
				s_test.extend(new_test)
			
		else:
			test_labels=new_test_labels
		#stop=time.time()
		#print("alpha_new-alpha time:", stop-start)
		S_prototype_scatter_plot(s_prototype, "g="+str(g))
		print("delta:",delta)
		'''
		if (delta==0.0):
			#print("if delta==0.0, minpts:",eta)
			if(eta<MINPTS):
				eta+=1 #MINPTS
				delta=1 
				_verify_core_states(eta)
		'''

	stop=time.time()
	print("total time:",stop-start)
	print("eta:",eta)
	'''
	np.savetxt("convergence_stat_n_with_prototype_on_traindata_"+str(itr)+".csv", convergence_stat, delimiter=",")
	
	print("after while points_available:",points_available)
	print("S_prototype:",len(s_prototype))
	#print("representative keys:",representatives.keys())
	print("eta:", eta)
	#print("MINPTS:",MINPTS)
	print("labels:",np.unique(labels.values()))
	S_prototype_scatter_plot(s_prototype, "before merge")
	'''
	'''
	""" Merge nearby  clusters till eta reaches MinPts"""
	if eta<MINPTS:
		_merge_when_no_core()
	S_prototype_scatter_plot(s_prototype, "after _merge_when_no_core()")
	THRESHOLD=1
	_identify_representatives()	
	_label_test_samples(s_test)
	'''
	#print("border:",len(borders))
	#%_label_test_points(itr)
	#_label_unprocessed_points()
	#S_prototype_scatter_plot(s_prototype,"with all+++ the points after merge")
	#scatter_plot(range(len(dataset)))
	'''
	if len(borders)>0:
		x_l,x_h=_confidence_interval()
		_noise_estimate(x_h)
	'''
	#stop=time.time()
	#print("time:",stop-start)
	#_save_partition(itr)
	S_prototype_scatter_plot(s_prototype, "after _merge_when_no_core()")
        #K=scatter_plot(range(len(dataset)), itr),
	return  stop-start #delta, K


for itr in range(1):
	riscan(itr)
