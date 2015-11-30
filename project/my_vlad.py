from sklearn.utils.extmath import squared_norm
import os, sys
import image_norm_test as myData;
import numpy as np
import time

class my_vlad(object):

    def __init__(self , centroids):
        self.unitVec = []
        for item in centroids:
    #        print(item)
            item_norm = np.linalg.norm(item)
    #        print(item/item_norm)
            self.unitVec.append(item/item_norm)
        
    def get_vlad(self,datas) :
        result = []
        ### Assign data to nearest centriods
        mapping = self.NN(datas,self.unitVec)
        #print("mapping: ",mapping)
        
        for item in mapping:
            summ = np.sum(np.array(item),0)
            if(summ.all() != 0):
                ### used intra_normalization ###   
                ans = self.normalization(summ)
            else :
                ans = np.zeros(len(self.unitVec[0]))
            result.append(ans.tolist())
#        print(result)
        result = self.normalization(result)
#        print(result.tolist())
        #return sum( cosine_similarity(data,centroids))
        return result
    
    def normalization(self,summ):
        
        return (summ/np.linalg.norm(summ))
    
    def NN(self,datas, centroids):
        start = time.time()
        ####### find which centroids the x is closet to, and put x into the centroids location #####
        group = [[] for n in range(len(centroids))]
#        group = [[np.zeros(len(centroids[0]))] for n in range(len(centroids))]

        for x in datas:
            min_dist = sys.maxsize
            min_diff = list()
            index = 0
            for i in range(len(centroids)):
                diff = x - centroids[i]
                dist = np.dot(diff, diff)
                if dist < min_dist :
                    min_dist = dist
                    min_diff = diff
                # end if
            # end for
            group[index].append(min_diff)
        # end for
        print("End of NN: ",(time.time()-start))
        return group

    