from typing import List,Tuple

class Tensor():
    
    def tensor(self,l:List[int],size:Tuple[int]):
        check=1
        for j in size:
            check=(j*check)
            
        if len(l)!=check or len(l)==0:
            raise ValueError('Size Is Not Correct')

        s_len=len(size)
        l_len=len(l)
        index=[[0 for _ in range(l_len)] for _ in range(s_len)]
        tot=1
        list_total=[]
        size_revers=size[::-1]
        for idx,i in enumerate(size_revers):
            li=[]
            tot=i*tot
            for j in range(tot):
                st=int((l_len/tot)*j)
                en=int((l_len/tot)*(j+1))
                li=[j for i in index[idx][st:en]]
                index[idx][st:en]=li                                    
        index=index[::-1]
        index.append(l)
        zip_total=list(zip(*index))
        #print i and j and element of list (i,j,element) which size=(i,j)
        print(zip_total)


lst=[2,4,2,1,6,7,8,6]
obj=Tensor()
obj.tensor(l=lst,size=(2,4))













