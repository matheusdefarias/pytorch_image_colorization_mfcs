# **************************************
# ***** Richard Zhang / 2016.08.06 *****
# **************************************
import numpy as np
import warnings
import os
import sklearn.neighbors as nn
import caffe
from skimage import color

# ************************
# ***** CAFFE LAYERS *****
# ************************
class BGR2LabLayer(caffe.Layer):
    ''' Layer converts BGR to Lab
    INPUTS    
        bottom[0].data  Nx3xXxY     
    OUTPUTS
        top[0].data     Nx3xXxY     
    '''
    def setup(self,bottom, top):
        warnings.filterwarnings("ignore")

        if(len(bottom)!=1):
            raise Exception("Layer should a single input")
        if(bottom[0].data.shape[1]!=3):
            raise Exception("Input should be 3-channel BGR image")

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,3,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = color.rgb2lab(bottom[0].data[:,::-1,:,:].astype('uint8').transpose((2,3,0,1))).transpose((2,3,0,1))

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            # bottom[i].diff[...] = np.zeros_like(bottom[i].data)


'''
>>> l
array([[0, 1, 2],
       [3, 4, 5]])


>>> ab
array([[[ 6,  7,  8],
        [ 9, 10, 11]],

       [[12, 13, 14],
        [15, 16, 17]]])
'''

class NNEncLayer(caffe.Layer):
    ''' Layer which encodes ab map into Q colors
    INPUTS    
        bottom[0]   Nx2xXxY     
    OUTPUTS
        top[0].data     NxQ     
    '''
    def setup(self,bottom, top):
        warnings.filterwarnings("ignore")

        if len(bottom) == 0:
            raise Exception("Layer should have inputs")
        self.NN = 10
        self.sigma = 5.
        self.ENC_DIR = './resources/'
        self.nnenc = NNEncode(self.NN,self.sigma,km_filepath=os.path.join(self.ENC_DIR,'pts_in_hull.npy'))

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]
        self.Q = self.nnenc.K # 313

    def reshape(self, bottom, top):
        top[0].reshape(self.N,self.Q,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = self.nnenc.encode_points_mtx_nd(bottom[0].data[...],axis=1)

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class PriorBoostLayer(caffe.Layer):
    ''' Layer boosts ab values based on their rarity
    INPUTS    
        bottom[0]       NxQxXxY     
    OUTPUTS
        top[0].data     Nx1xXxY
    '''
    def setup(self,bottom, top):
        if len(bottom) == 0:
            raise Exception("Layer should have inputs")

        self.ENC_DIR = './resources/'
        self.gamma = .5
        self.alpha = 1.
        self.pc = PriorFactor(self.alpha,gamma=self.gamma,priorFile=os.path.join(self.ENC_DIR,'prior_probs.npy'))

        self.N = bottom[0].data.shape[0]
        self.Q = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = self.pc.forward(bottom[0].data[...],axis=1)

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class NonGrayMaskLayer(caffe.Layer):
    ''' Layer outputs a mask based on if the image is grayscale or not
    INPUTS    
        bottom[0]       Nx2xXxY     ab values
    OUTPUTS
        top[0].data     Nx1xXxY     1 if image is NOT grayscale
                                    0 if image is grayscale
    '''
    def setup(self,bottom, top):
        if len(bottom) == 0:
            raise Exception("Layer should have inputs")

        self.thresh = 5 # threshold on ab value
        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
 
    def forward(self, bottom, top):
        # if an image has any (a,b) value which exceeds threshold, output 1
        top[0].data[...] = (np.sum(np.sum(np.sum(np.abs(bottom[0].data) > self.thresh,axis=1),axis=1),axis=1) > 0)[:,na(),na(),na()]

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class ClassRebalanceMultLayer(caffe.Layer):
    ''' INPUTS
        bottom[0]   NxMxXxY     feature map
        bottom[1]   Nx1xXxY     boost coefficients
    OUTPUTS
        top[0]      NxMxXxY     on forward, gets copied from bottom[0]
    FUNCTIONALITY
        On forward pass, top[0] passes bottom[0]
        On backward pass, bottom[0] gets boosted by bottom[1]
        through pointwise multiplication (with singleton expansion) '''
    def setup(self, bottom, top):
        # check input pair
        if len(bottom)==0:
            raise Exception("Specify inputs")

    def reshape(self, bottom, top):
        i = 0
        if(bottom[i].data.ndim==1):
            top[i].reshape(bottom[i].data.shape[0])
        elif(bottom[i].data.ndim==2):
            top[i].reshape(bottom[i].data.shape[0], bottom[i].data.shape[1])
        elif(bottom[i].data.ndim==4):
            top[i].reshape(bottom[i].data.shape[0], bottom[i].data.shape[1], bottom[i].data.shape[2], bottom[i].data.shape[3])

    def forward(self, bottom, top):
        # output equation to negative of inputs
        top[0].data[...] = bottom[0].data[...]
        # top[0].data[...] = bottom[0].data[...]*bottom[1].data[...] # this was bad, would mess up the gradients going up

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[0].diff[...] = top[0].diff[...]*bottom[1].data[...]
            # print 'Back-propagating class rebalance, %i'%i

class LossMeterLayer(caffe.Layer):
    ''' Layer acts as a "meter" to track loss values '''
    def setup(self,bottom,top):
        if(len(bottom)==0):
            raise Exception("Layer needs inputs")

        self.param_str_split = self.param_str.split(' ')
        self.LOSS_DIR = self.param_str_split[0]
        self.P = int(self.param_str_split[1])
        self.H = int(self.param_str_split[2])
        if(len(self.param_str_split)==4):
            self.prefix = self.param_str_split[3]
        else:
            self.prefix = ''

        self.cnt = 0 # loss track counter
        # self.P = 1 # interval to print losses
        self.h = 0 # index into history
        self.L = len(bottom)
        self.losses = np.zeros((self.L,self.H))

        self.ITER_PATH = os.path.join(self.LOSS_DIR,'iter.npy')
        self.LOG_PATH = os.path.join(self.LOSS_DIR,'loss_log')

        if(not os.path.exists(self.LOSS_DIR)):
            os.mkdir(self.LOSS_DIR)
            
        if(os.path.exists(self.ITER_PATH)):
            self.iter = np.load(self.ITER_PATH)
        else:
            self.iter = 0 # iteration counter
        print ('Initial iteration: %i'%(self.iter+1))

    def reshape(self,bottom,top):
        0;

    def forward(self,bottom,top):
        for ll in range(self.L):
            self.losses[ll,self.h] = bottom[ll].data[...]

        if(np.mod(self.cnt,self.P)==self.P-1): # print
            if(self.cnt >= self.H-1):
                tmp_str = 'NumAvg %i, Loss '%(self.H)
                for ll in range(self.L):
                    tmp_str += '%.3f, '%np.mean(self.losses[ll,:])
            else:
                tmp_str = 'NumAvg %i, Loss '%(self.h)
                for ll in range(self.L):
                    tmp_str += '%.3f, '%np.mean(self.losses[ll,:self.cnt+1])
            print_str = '%s: Iter %i, %s'%(self.prefix,self.iter+1,tmp_str)
            print(print_str)

            self.f = open(self.LOG_PATH,'a')
            self.f.write(print_str)
            self.f.write('\n')
            self.f.close()
            np.save(self.ITER_PATH,self.iter)

        self.h = np.mod(self.h+1,self.H) # roll through history
        self.cnt = self.cnt+1
        self.iter = self.iter+1

    def backward(self,top,propagate_down,bottom):
        for ll in range(self.L):
            continue

# ***************************
# ***** SUPPORT CLASSES *****
# ***************************
class PriorFactor():
    ''' Class handles prior factor '''
    def __init__(self,alpha,gamma=0,verbose=True,priorFile=''):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior**alpha
        #   gamma           integer     percentage to mix in uniform prior with empirical prior
        #   priorFile       file        file which contains prior probabilities across classes

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = np.load(priorFile) # realiza a leitura do arquivo de probabilidade para cada bin(313) de cor.

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs) # Cria uma matriz de zeros com a mesma dimensão da matriz do arquivo prior_probs.npy (313,)
        self.uni_probs[self.prior_probs!=0] = 1. # preenche a matriz com 1 caso os valores em prior_probs sejam diferentes de 0.
        self.uni_probs = self.uni_probs/np.sum(self.uni_probs) # calcula a probabilidade para cada bin de cor  que nesse caso será a mesma para cada bin. Distribuição uniforme.

        # convex combination of empirical prior and uniform distribution
        # O prior_mix sera utilizado para gerar o prior_factor, entao há uma ponderação entre as duas distribuições de probabilidade.       
        self.prior_mix = (1-self.gamma)*self.prior_probs + self.gamma*self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha # geração do prior_factor
        self.prior_factor = self.prior_factor/np.sum(self.prior_probs*self.prior_factor) # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs*self.prior_factor # Aplica o fator na matriz de probabilidade de cada bin de cor lida do arquivo prior_probs.npy
        self.implied_prior = self.implied_prior/np.sum(self.implied_prior) # re-normalize

        if(self.verbose):
            self.print_correction_stats()

    def print_correction_stats(self):
        print('Prior factor correction:')
        print('  (alpha,gamma) = (%.2f, %.2f)'%(self.alpha,self.gamma))
        print('  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)'%(np.min(self.prior_factor),np.max(self.prior_factor),np.mean(self.prior_factor),np.median(self.prior_factor),np.sum(self.prior_factor*self.prior_probs)))

    def forward(self,data_ab_quant,axis=1):
        data_ab_maxind = np.argmax(data_ab_quant,axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if(axis==0):
            return corr_factor[na(),:]
        elif(axis==1):
            return corr_factor[:,na(),:] # Retorna os fatores referentes aos bins que apresentaram a maior probabilidade para cada pixel no par (a,b) da imagem.
        elif(axis==2):
            return corr_factor[:,:,na(),:]
        elif(axis==3):
            return corr_factor[:,:,:,na()]

class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        if(check_value(cc,-1)):
            self.cc = np.load(km_filepath) # Carregou o arquivo com os bins de cor (313,2)
        else:
            self.cc = cc
        self.K = self.cc.shape[0] # 313
        self.NN = int(NN) # 10
        self.sigma = sigma # 0.5
        self.nbrs = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc) # Instancia do K-NN
        self.alreadyUsed = False

    def encode_points_mtx_nd(self,pts_nd,axis=1,returnSparse=False,sameBlock=True):
        pts_flt = flatten_nd_array(pts_nd,axis=axis) # deveria ser axis = 0 | converte uma matriz 3d em 2d

        '''
        >>> ab
        array([[[ 6,  7,  8],
                [ 9, 10, 11]],

               [[12, 13, 14],
                [15, 16, 17]]])

        >>> pts_flt = flatten_nd_array(ab,axis=0)
        >>> pts_flt
        array([[ 6, 12],
               [ 7, 13],
               [ 8, 14],
               [ 9, 15],
               [10, 16],
               [11, 17]])

        '''

        P = pts_flt.shape[0]
        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0 # already pre-allocated
        else:
            self.alreadyUsed = True

            '''
            >>> pts_enc_flt = np.zeros((P,k))
            >>> pts_enc_flt
            array([[0., 0., 0., ..., 0., 0., 0.],
                   [0., 0., 0., ..., 0., 0., 0.],
                   [0., 0., 0., ..., 0., 0., 0.],
                   [0., 0., 0., ..., 0., 0., 0.],
                   [0., 0., 0., ..., 0., 0., 0.],
                   [0., 0., 0., ..., 0., 0., 0.]])
            '''

            self.pts_enc_flt = np.zeros((P,self.K))

            '''
            >>> p_inds = np.arange(0,P,dtype='int')[:,na()]
            >>> p_inds
            array([[0],
                   [1],
                   [2],
                   [3],
                   [4],
                   [5]])

            '''
            self.p_inds = np.arange(0,P,dtype='int')[:,na()]

        P = pts_flt.shape[0]

        (dists,inds) = self.nbrs.kneighbors(pts_flt) 
        '''
        >>> (dists,inds) = nbrs.kneighbors(pts_flt)
        >>> (dists, inds)
        (array([[ 4.47213595,  6.32455532,  8.94427191, 10.        , 12.64911064,
        13.41640786, 14.14213562, 16.1245155 , 16.1245155 , 17.88854382],
       [ 4.24264069,  7.61577311,  7.61577311,  9.89949494, 13.34166406,
        13.34166406, 14.76482306, 14.76482306, 17.2626765 , 17.2626765 ],
       [ 4.47213595,  6.32455532,  8.94427191, 10.        , 12.64911064,
        13.41640786, 14.14213562, 16.1245155 , 16.1245155 , 17.88854382],
       [ 5.09901951,  5.09901951, 10.29563014, 10.29563014, 12.08304597,
        12.08304597, 15.03329638, 15.03329638, 17.49285568, 17.49285568],
       [ 4.        ,  6.        , 10.77032961, 10.77032961, 11.66190379,
        11.66190379, 14.        , 16.        , 17.20465053, 17.20465053],
       [ 3.16227766,  7.07106781,  9.48683298, 11.40175425, 11.40175425,
        13.03840481, 13.03840481, 15.8113883 , 17.02938637, 17.02938637]]), array([[140, 121, 141, 122, 139, 120, 158, 159, 103, 104],
       [140, 141, 121, 122, 139, 158, 120, 159, 103, 142],
       [140, 141, 121, 122, 158, 159, 139, 120, 142, 123],
       [140, 141, 122, 121, 159, 158, 142, 139, 120, 123],
       [141, 140, 122, 159, 158, 121, 142, 139, 123, 160],
       [141, 140, 159, 122, 158, 142, 121, 160, 123, 139]]))
        >>> cc[140]
        array([10, 10])

        '''
        wts = np.exp(-dists**2/(2*self.sigma**2)) # Converte as distancias para probabilidade. Quanto maior a distancia, menor a probabilidade
        wts = wts/np.sum(wts,axis=1)[:,na()] # Algum tipo de normalizacao
        '''
        >>> l
        array([[0, 1, 2],
               [3, 4, 5]])
        >>> teste = np.exp(-l**2/(2*5**2))
        >>> teste
        array([[1.        , 0.98019867, 0.92311635],
               [0.83527021, 0.72614904, 0.60653066]])
        >>> teste = teste/np.sum(teste,axis=1)[:,na()]
        >>> teste
        array([[0.34443386, 0.33761361, 0.31795253],
               [0.38528114, 0.33494733, 0.27977153]])

        '''

        '''
        array([[0.4308033 , 0.28877608, 0.12975546, 0.08697769, 0.02619718,
        0.01756049, 0.01177115, 0.0035454 , 0.0035454 , 0.00106785],
       [0.44921494, 0.20184528, 0.20184528, 0.09069493, 0.01831099,
        0.01831099, 0.00822766, 0.00822766, 0.00166114, 0.00166114],
       [0.4308033 , 0.28877608, 0.12975546, 0.08697769, 0.02619718,
        0.01756049, 0.01177115, 0.0035454 , 0.0035454 , 0.00106785],
       [0.38033574, 0.38033574, 0.07678846, 0.07678846, 0.03450328,
        0.03450328, 0.00696609, 0.00696609, 0.00140643, 0.00140643],
       [0.46181362, 0.30956293, 0.06249968, 0.06249968, 0.04189479,
        0.04189479, 0.01261847, 0.00380061, 0.00170772, 0.00170772],
       [0.51818539, 0.2328357 , 0.10461983, 0.04700872, 0.04700872,
        0.02112238, 0.02112238, 0.00426453, 0.00191618, 0.00191618]])

        '''

        self.pts_enc_flt[self.p_inds,inds] = wts # Coloca as probabilidades dos 10 bins, cada um no seu respectivo bin(coluna) de acordo com o indice retornado pelo K-NN
        '''
        >>> pts_enc_flt[p_inds,inds] = wts
        >>> pts_enc_flt
        array([[0., 0., 0., ..., 0., 0., 0.],
               [0., 0., 0., ..., 0., 0., 0.],
               [0., 0., 0., ..., 0., 0., 0.],
               [0., 0., 0., ..., 0., 0., 0.],
               [0., 0., 0., ..., 0., 0., 0.],
               [0., 0., 0., ..., 0., 0., 0.]])
        >>> pts_enc_flt[0,140]
        0.4308032952771784
        >>> pts_enc_flt[0,0]
        0.0
        >>> pts_enc_flt[0,140]
        0.4308032952771784
        >>> pts_enc_flt[0,121]
        0.2887760847225033
        >>> pts_enc_flt[0,120]
        0.01756049179664067
        >>> pts_enc_flt[0,1]
        0.0

        '''
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt,pts_nd,axis=axis)

        return pts_enc_nd # Retorna a distribuição de probabilidades dos bins para o par (a,b) da imagem.

    def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt,self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
        return pts_dec_nd

    def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
        pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
        pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
        if(returnEncode):
            return (pts_dec_nd,pts_1hot_nd)
        else:
            return pts_dec_nd

# *****************************
# ***** Utility functions *****
# *****************************
def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if(np.array(inds).size==1):
        if(inds==val):
            return True
    return False

def na(): # shorthand for new axis
    return np.newaxis

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out
