#coding=utf-8
import sys
sys.path.append('.')

reload(sys)
sys.setdefaultencoding('utf-8')

import os
import json
from time import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import six.moves.cPickle as pickle
import gensim
import copy
import codecs
from openpyxl import Workbook

import numpy as np
import theano
import theano.tensor as T
#from theano.sandbox.cuda.dnn import dnn_conv
import random

import jieba
from Seq2Seq1 import make_chat_corpus


from lib import activations
from lib import updates
from lib import inits
#from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng,t_rng,t_rng_cpu
from lib.theano_utils import floatX, sharedX
#from lib.np_utils import np_softmax

sys.path.append('.')
sys.path.append('./lib/coco_caption/')
sys.path.append('./lib/coco_caption/pycxevalcap/')

from pycxevalcap.eval import COCOEvalCap

from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider


my_bleu=Bleu(4)
my_meteor=Meteor()
my_rouge=Rouge()
my_cider=Cider()
#################################################### make result dir
desc = 'weibo_model_v3'
select_epochs=  15

model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc

#################################################### load Word2Vec model
model = gensim.models.Word2Vec.load("./models/Word2Vec/word2vec_gensim")

word_vectors = model.wv

dict = word_vectors.vocab
dict_index2word = word_vectors.index2word

sorted_vecs = []
for tmp_w in dict_index2word:
    tmp_vec = word_vectors[tmp_w]
    sorted_vecs.append(tmp_vec)

sorted_vecs = np.asarray(sorted_vecs, dtype='float32')

dict_index2word.append(u'EOF')
sorted_vecs = np.concatenate((sorted_vecs, 7 * np.ones((1, sorted_vecs.shape[1]), dtype='float32')), axis=0)


n_word_dict=sorted_vecs.shape[0]
n_word_dim =sorted_vecs.shape[1]  # # of dim of word representation

########## params

nbatch = 1      # # of examples in batch
max_T=30          # # sentense length
max_T_M=20        # # sentense length of M
n_LSTM=500       # # of LSTM_hidden_units    #1000 1500
dimAttention= 100
N_M=5

# adam optim  params
l2 = 1e-5         # l2 weight decay
b1=0.95
b2=0.999
learning_rate=0.0001 # init:0.001

######### init settings
relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
orfn=inits.Orthogonal(scale=1)
gifn = inits.Normal(scale=0.01)

gain_ifn = inits.Normal(loc=1., scale=0.01)
bias_ifn = inits.Constant(c=0.)

startword_ifn = inits.Constant(c=-7.)

################################### load saved model :  modified

file_name_saved='Test_test_corpus_900.pkl'

word_start = startword_ifn((1, 1, n_word_dim), 'word_start')

shared_Word_vecs = sharedX(sorted_vecs)  # T._shared(sorted_vecs, borrow=True)

[LSTM_hidden0, W_LSTM_hidden_enc, W_LSTM_in_enc, b_LSTM_enc,
                    LSTM_hidden0_rev, W_LSTM_hidden_enc_rev, W_LSTM_in_enc_rev, b_LSTM_enc_rev,
 U_attention_gen , W_attention_gen, b_attention_gen, v_attention_gen,
                      W_init_h0, b_init_h0, W_init_c0, b_init_c0,
                      W1_M, b1_M, W2_M, WM_mu_zt, bM_mu_zt, WM_sigma_zt, bM_sigma_zt, W3_M, b3_M , Wp_M_mu, bp_M_mu, Wp_M_sigma, bp_M_sigma,
                      W1_M0, b1_M0, W2_M0, WM_mu_zt0, bM_mu_zt0, WM_sigma_zt0, bM_sigma_zt0, W3_M0, b3_M0 , Wp_M_mu0, bp_M_mu0, Wp_M_sigma0, bp_M_sigma0,
                     W_LSTM_hidden_gen, W_LSTM_in_gen, b_LSTM_gen,W_word_gen, b_word_gen,W_softmax_gen, b_softmax_gen,
                    W_bow1, b_bow1, W_bow2,  b_bow2,  W_softmax_bow,  b_softmax_bow,
                    W_bow1t, b_bow1t, W_bow2t,  b_bow2t,  W_softmax_bowt,  b_softmax_bowt
 ] = \
    [sharedX(p) for p in joblib.load('models/%s/%d_total_params.jl' % (desc, select_epochs))]

enc_params =   [LSTM_hidden0, W_LSTM_hidden_enc, W_LSTM_in_enc, b_LSTM_enc,
                    LSTM_hidden0_rev, W_LSTM_hidden_enc_rev, W_LSTM_in_enc_rev, b_LSTM_enc_rev]

gen_params =   [W_LSTM_in_gen,W_LSTM_hidden_gen,b_LSTM_gen,
                  W_attention_gen, b_attention_gen, v_attention_gen,
                  W_word_gen,b_word_gen,W_softmax_gen,b_softmax_gen,
                W3_M, b3_M , Wp_M_mu, bp_M_mu, Wp_M_sigma, bp_M_sigma]



######################################
def encoder_network(Qs_words, Qs_masks, LSTM_hidden0, W_LSTM_hidden_enc, W_LSTM_in_enc, b_LSTM_enc,
                    LSTM_hidden0_rev, W_LSTM_hidden_enc_rev, W_LSTM_in_enc_rev, b_LSTM_enc_rev):

    LSTM_h0 = (T.extra_ops.repeat(LSTM_hidden0, repeats=Qs_words.shape[1], axis=0)).astype(theano.config.floatX)

    LSTM_h0_rev = (T.extra_ops.repeat(LSTM_hidden0_rev, repeats=Qs_words.shape[1], axis=0)).astype(theano.config.floatX)

    cell0 = T.zeros((Qs_words.shape[1], n_LSTM), dtype=theano.config.floatX)

    ##################################################################

    def recurrence_enc(word_t,t_mask,h_t_prior,c_t_prior,W_LSTM_hidden_enc,W_LSTM_in_enc,b_LSTM_enc): #x_temp :  batch_size * dim_features



        lstm_t = T.dot(h_t_prior, W_LSTM_hidden_enc) + T.dot(word_t, W_LSTM_in_enc) + b_LSTM_enc
        i_t_enc = T.nnet.sigmoid(lstm_t[:, 0*n_LSTM:1*n_LSTM])
        f_t_enc = T.nnet.sigmoid(lstm_t[:, 1*n_LSTM:2*n_LSTM])

        cell_t_enc = f_t_enc * c_t_prior + i_t_enc * T.tanh(lstm_t[:, 2*n_LSTM:3*n_LSTM])
        cell_t_enc = t_mask.dimshuffle([0, 'x']) * cell_t_enc + (1. - t_mask.dimshuffle([0, 'x'])) * c_t_prior


        o_t_enc = T.nnet.sigmoid(lstm_t[:, 3*n_LSTM:4*n_LSTM])
        h_t = o_t_enc * T.tanh(cell_t_enc)
        h_t = t_mask.dimshuffle([0, 'x']) * h_t + (1. - t_mask.dimshuffle([0, 'x'])) * h_t_prior

        #y_t=sigmoid(T.dot(h_t, W_dis) + b_dis)

        return h_t.astype(theano.config.floatX) ,cell_t_enc.astype(theano.config.floatX)


    (h_list , _), _ = theano.scan(recurrence_enc,sequences=[Qs_words,Qs_masks],
                                                    outputs_info=[LSTM_h0,cell0],
                                                    non_sequences=[W_LSTM_hidden_enc,W_LSTM_in_enc,b_LSTM_enc],
                                                    n_steps=Qs_words.shape[0],
                                                    strict=True)

    (h_list_rev , _ ), _ = theano.scan(recurrence_enc,sequences=[Qs_words[::-1,:,:],Qs_masks[::-1,:]],
                                                    outputs_info=[LSTM_h0_rev,cell0],
                                                    non_sequences=[W_LSTM_hidden_enc_rev,W_LSTM_in_enc_rev,b_LSTM_enc_rev],
                                                    n_steps=Qs_words.shape[0],
                                                    strict=True)

    h_t_lang = T.concatenate([h_list, h_list_rev[::-1,:,:]], axis=2)  # was -1

    gen_init0_lang=T.concatenate([h_list[-1], h_list_rev[-1]], axis=1)

    return h_t_lang, gen_init0_lang # T * batch * (2*n_LSTM)



######################################

def generate_next(h_t_prior,word_t_prior,z_t_prior,c_t_prior, Qs_masks,  h_enc, hid_align,

                  W_LSTM_in_gen,W_LSTM_hidden_gen,b_LSTM_gen,
                  W_attention_gen, b_attention_gen, v_attention_gen,
                  W_word_gen,b_word_gen,W_softmax_gen,b_softmax_gen,
                  W3_M, b3_M , Wp_M_mu, bp_M_mu, Wp_M_sigma, bp_M_sigma

                  ): #x_temp :  batch_size * dim_features


        ################################################ calculate input
    word_t_prior2 = T.concatenate([word_t_prior, z_t_prior], axis=1)

    lstm_t = T.dot(h_t_prior, W_LSTM_hidden_gen) + T.dot(word_t_prior2, W_LSTM_in_gen) + b_LSTM_gen
    i_t_enc = T.nnet.sigmoid(lstm_t[:, 0*n_LSTM:1*n_LSTM])
    f_t_enc = T.nnet.sigmoid(lstm_t[:, 1*n_LSTM:2*n_LSTM])


    cell_t_enc = f_t_enc * c_t_prior + i_t_enc * T.tanh(lstm_t[:, 2*n_LSTM:3*n_LSTM])
    #cell_t_enc = t_mask.dimshuffle([0, 'x']) * cell_t_enc + (1. - t_mask.dimshuffle([0, 'x'])) * c_t_prior

    o_t_enc = T.nnet.sigmoid(lstm_t[:, 3*n_LSTM:4*n_LSTM])

    h_list = o_t_enc * T.tanh(cell_t_enc)
    #h_t = t_mask.dimshuffle([0, 'x']) * h_t + (1. - t_mask.dimshuffle([0, 'x'])) * h_t_prior

    #################################VAE   VAE calculate p(Zt)
    h_prior_0=lrelu(T.dot(h_list, W3_M) + b3_M)   #T_dec  x batch_size x dim_atten
    u_0t=T.dot(h_prior_0, Wp_M_mu) + bp_M_mu
    log_sigma_0t=T.dot(h_prior_0,Wp_M_sigma) + bp_M_sigma

    eps = t_rng.normal(size=(u_0t.shape[0] , u_0t.shape[1]), avg=0.0, std=1.0, dtype=theano.config.floatX)
    Zt = u_0t + T.exp(log_sigma_0t) * eps  #T_dec  x batch_size x dim_atten

    #################################

    #hid_align = T.dot(h_enc, U_attention_gen)  # T_enc*Batch* dimAtten

    h_t_info = T.concatenate([Zt, word_t_prior], axis=1)

    hdec_align = T.dot(h_t_info, W_attention_gen)  # *Batch* dimAtten

    all_align = T.tanh(hid_align + hdec_align.dimshuffle(['x', 0, 1]) + b_attention_gen.dimshuffle(['x','x', 0]))
    # T_enc  x batch_size x dimAttention

    e = all_align * v_attention_gen.dimshuffle(['x','x',0])
    e = e.sum(axis=2) * Qs_masks # T_enc  x batch_size

    # normalize
    alpha = T.nnet.softmax(e.T) #  #  (batch_size) * T_enc


    # conv_feature representation at time T
    attention_enc = alpha.dimshuffle([1, 0, 'x']) * h_enc # T_enc x batch_size x h_dim
    attention_enc = attention_enc.sum(axis=0) # T_dec x T_enc x batch_size x h_dim --> T_dec  x batch_size x h_dim

    prepare_word=T.concatenate([attention_enc,h_list, Zt], axis=1)

    word_t=lrelu(T.dot(prepare_word, W_word_gen) + b_word_gen)  #T * batch * middle_dim
    word_soft=T.dot(word_t, W_softmax_gen)+b_softmax_gen
    word_soft_K=T.nnet.softmax(word_soft)


    return word_soft_K.astype(theano.config.floatX) , h_list.astype(theano.config.floatX) ,cell_t_enc.astype(theano.config.floatX),Zt.astype(theano.config.floatX)

'''
##################################################### # batch *M * T
Qns_word_list = T.tensor3('Qns_word_list', dtype='int32')     # batch *M * T
Qns_mask = T.tensor3('Qns_mask', dtype='float32')             # batch *M * T
Ans_word_list = T.tensor3('Ans_word_list', dtype='int32')     # batch *M * T
Ans_mask = T.tensor3('Ans_mask', dtype='float32')             # batch *M * T

Qns_word_list.tag.test_value = np.random.randint(1000,size=(nbatch,5,max_T)).astype(np.int32)
Qns_mask.tag.test_value = np.random.randint(1,size=(nbatch,5,max_T)).astype(np.float32)
Ans_word_list.tag.test_value = np.random.randint(1000,size=(nbatch,5,max_T)).astype(np.int32)
Ans_mask.tag.test_value = np.random.randint(1,size=(nbatch,5,max_T)).astype(np.float32)

####################################################  encode QM

Qns_word_list_flat = T.flatten(Qns_word_list,ndim=1) #
Qns_word_vecs = shared_Word_vecs[Qns_word_list_flat].reshape([Qns_word_list.shape[0]* Qns_word_list.shape[1], Qns_word_list.shape[2], n_word_dim]) # (batch* M) * T* n_dim
#Qns_word_vecs_in=Qns_word_vecs.reshape([Qns_word_list.shape[0], Qns_word_list.shape[1], Qns_word_list.shape[2], n_word_dim])
Qns_word_vecs_in= Qns_word_vecs.dimshuffle([1, 0, 2])

Qns_mask_in= Qns_mask.reshape([Qns_mask.shape[0]* Qns_mask.shape[1], Qns_mask.shape[2]]) #(batch *M) * T

_, hQns_enc_end = encoder_network(Qns_word_vecs_in,Qns_mask_in.T,*enc_params)  # T *(batch *M) * (2*n_LSTM),      (batch *M) * (2*n_LSTM)

hQns_enc_end= hQns_enc_end.reshape([Qns_word_list.shape[0],Qns_word_list.shape[1],hQns_enc_end.shape[1]]) #batch *M * (2*n_LSTM)


####################################################  encode AM

Ans_word_list_flat = T.flatten(Ans_word_list,ndim=1) #
Ans_word_vecs = shared_Word_vecs[Ans_word_list_flat].reshape([Ans_word_list.shape[0]* Ans_word_list.shape[1], Ans_word_list.shape[2], n_word_dim]) # (batch* M) * T* n_dim
#Qns_word_vecs_in=Qns_word_vecs.reshape([Qns_word_list.shape[0], Qns_word_list.shape[1], Qns_word_list.shape[2], n_word_dim])
Ans_word_vecs_in= Ans_word_vecs.dimshuffle([1, 0, 2])

Ans_mask_in= Ans_mask.reshape([Ans_mask.shape[0]* Ans_mask.shape[1], Ans_mask.shape[2]]) #(batch *M) * T

_, hAns_enc_end = encoder_network(Ans_word_vecs_in,Ans_mask_in.T,*enc_params)  # T *(batch *M) * n_LSTM,      (batch *M) * (2*n_LSTM)

hAns_enc_end= hAns_enc_end.reshape([Ans_word_list.shape[0],Ans_word_list.shape[1],hAns_enc_end.shape[1]]) #batch *M * (2*n_LSTM)

#Total_M=T.concatenate([hQns_enc_end, hAns_enc_end], axis=1)  #batch * 2M * (2*n_LSTM)
#Total_M = Total_M.sum(axis=1)  #batch  * (2*n_LSTM)

Total_M0= T.concatenate([hQns_enc_end, hAns_enc_end], axis=1)  #batch * 2M * (2*n_LSTM)
Total_M = Total_M0.sum(axis=1)  #batch  * (2*n_LSTM)
'''

####################################################  encode  decode
Qs_word_list = T.matrix('Qs_word_list', dtype='int32')  # batch * T
Qs_mask = T.matrix('Qs_mask', dtype='float32')  # batch * T
#As_word_list = T.matrix('As_word_list', dtype='int32')  # batch * T
#As_mask = T.matrix('As_mask', dtype='int32')  # batch * T

# provide Theano with a default test-value
Qs_word_list.tag.test_value = np.random.randint(1000,size=(nbatch,max_T)).astype(np.int32)
#As_word_list.tag.test_value = np.random.randint(1000,size=(nbatch,max_T)).astype(np.int32)

Qs_mask.tag.test_value = np.random.randint(1,size=(nbatch,max_T)).astype(np.float32)
#As_mask.tag.test_value = np.random.randint(1,size=(nbatch,max_T)).astype(np.int32)

####################################################
Qs_word_list_flat = T.flatten(Qs_word_list.T,ndim=1) #
Qs_word_vecs = shared_Word_vecs[Qs_word_list_flat].reshape([Qs_word_list.shape[1], Qs_word_list.shape[0], n_word_dim]) # T * batch * n_dim

#As_word_list_flat = T.flatten(As_word_list.T,outdim=1) #words x #samples
#As_word_vecs = shared_Word_vecs[As_word_list_flat].reshape([As_word_list.shape[1], As_word_list.shape[0], n_word_dim]) # T * batch * n_dim


h_t_lang, gen_init0_lang = encoder_network(Qs_word_vecs,Qs_mask.T,*enc_params)  # batch * n_LSTM


#calculate p(Zt)
h_prior_00=lrelu(T.dot(gen_init0_lang, W3_M0) + b3_M0)   #batch_size x dim_atten
u_0t0=T.dot(h_prior_00, Wp_M_mu0) + bp_M_mu0
log_sigma_0t0=T.dot(h_prior_00,Wp_M_sigma0) + bp_M_sigma0



scale_Z=T.scalar('scale_Z', dtype='float32')
eps = t_rng.normal(size=(u_0t0.shape[0] , u_0t0.shape[1]), avg=0.0, std=1.0, dtype=theano.config.floatX)
#eps = t_rng.binomial(size=(u_0t.shape[0],u_0t.shape[1]), p=0.5, dtype=theano.config.floatX)
Zt = (u_0t0 + T.exp(log_sigma_0t0) * eps* scale_Z).astype(theano.config.floatX)


LSTM_h0=T.tanh(T.dot(Zt, W_init_h0)+b_init_h0)
cell0=T.tanh(T.dot(Zt, W_init_c0)+b_init_c0)
################################


word0= (T.extra_ops.repeat(word_start, repeats=Qs_word_list.shape[0], axis=1)).astype(theano.config.floatX)


#Total_M_h_enc= T.concatenate([Total_m0.dimshuffle([1, 0, 2]),h_t_lang], axis=0)
#Qs_mask_in= T.concatenate([T.ones((Total_m0.shape[1],Total_m0.shape[0]),dtype=theano.config.floatX), Qs_mask.T], axis=0)   # Qs_mask: batch * T

hid_align = T.dot(h_t_lang, U_attention_gen)  # T_enc*Batch* dimAtten
#return h_t_lang, hid_align ,LSTM_h0, cell0, word0 # T * batch * (2*n_LSTM)



h_t_prior = T.matrix()
c_t_prior = T.matrix()
z_t_prior = T.matrix()
word_t_prior = T.matrix()

h_enc = T.tensor3()
hid_align_in = T.tensor3()

word_soft_K, h_t_next, c_t_next ,z_t_next= generate_next(h_t_prior, word_t_prior,z_t_prior, c_t_prior ,
                                                Qs_mask.T, h_enc, hid_align_in,
                                                *gen_params) #T *batch * n_word_dict



print 'COMPILING'
t = time()
#_gen_init_M = theano.function([Qns_word_list,Ans_word_list,Qns_mask,Ans_mask], [Total_M, Total_M0])
_gen_init = theano.function([Qs_word_list,Qs_mask,scale_Z], [h_t_lang, hid_align , LSTM_h0, cell0, word0, Qs_mask,Zt])
_gen_next = theano.function([h_t_prior, word_t_prior, z_t_prior,c_t_prior , Qs_mask, h_enc, hid_align_in],[word_soft_K, h_t_next, c_t_next,z_t_next])

print '%.2f seconds to compile theano functions'%(time()-t)


######################################################
def generate_captions_perX(Qs_word_list,Qs_mask,Bsize, strategy,scale_Z): # batch * T

    sample = []
    sample_score = []

    hyp_all_h_list, hyp_hid_align_list, hyp_h_list, hyp_c_list , hyp_word_list_embed, Qs_mask,hyp_z_list=_gen_init(Qs_word_list,Qs_mask,scale_Z)  #h_enc, word0, cell0 ,  batch * X

    hyp_word_list_embed=hyp_word_list_embed.squeeze(axis=0)

    hyp_word_list=[]

    hyp_scores=np.zeros((1,)).astype(theano.config.floatX)

    dead_k=0
    live_k=1

    for ii in range(max_T):

        #hyp_scores #B*Bsize

        word_soft_list,h_next_list,c_next_list,z_next_list=_gen_next(hyp_h_list,hyp_word_list_embed,hyp_z_list,hyp_c_list,Qs_mask,hyp_all_h_list,hyp_hid_align_list)

        voc_size=word_soft_list.shape[1]

        if strategy==1:  # Max

            cand_scores = hyp_scores[:,None] - np.log(word_soft_list)
            cand_scores_flat=np.reshape(cand_scores,(cand_scores.shape[0]*cand_scores.shape[1],))

            ranks_flat=np.argsort(cand_scores_flat)[:(Bsize-dead_k)]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size

        elif strategy==2: # Softmax
            cand_scores = hyp_scores[:,None] + np.log(word_soft_list)
            cand_scores_flat=np.reshape(cand_scores,(cand_scores.shape[0]*cand_scores.shape[1],))
            cand_scores_flat=np_softmax(cand_scores_flat.astype('float64'))

            ranks_flat = np.random.multinomial(7*(Bsize-dead_k), cand_scores_flat, size=1)
            ranks_flat = np.argsort(-ranks_flat).squeeze()[:(Bsize-dead_k)]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size

        elif strategy==3: #

            jiaquan=0.1
            cand_scores = hyp_scores[:,None]*jiaquan - np.log(word_soft_list)
            cand_scores_flat=np.reshape(cand_scores,(cand_scores.shape[0]*cand_scores.shape[1],))

            ranks_flat=np.argsort(cand_scores_flat)[:(Bsize-dead_k)]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size

        elif strategy==5: #  random select

            select_range= int(1.5*(Bsize-dead_k))
            jiaquan=0.01
            cand_scores = hyp_scores[:,None]*jiaquan - np.log(word_soft_list)
            cand_scores_flat=np.reshape(cand_scores,(cand_scores.shape[0]*cand_scores.shape[1],))

            #ranks_flat=np.argsort(cand_scores_flat)[:(Bsize-dead_k)]

            ranks_select=np.argsort(cand_scores_flat)[:select_range]


            # 从a~d中取出不重复的三个字母
            ranks_flat = np.array(random.sample(ranks_select, Bsize-dead_k))

            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size

        elif strategy==4: # first_fix

            if ii==1:
                hyp_scores = hyp_scores * 0

            cand_scores = hyp_scores[:,None] - np.log(word_soft_list)
            cand_scores_flat=np.reshape(cand_scores,(cand_scores.shape[0]*cand_scores.shape[1],))

            ranks_flat=np.argsort(cand_scores_flat)[:(Bsize-dead_k)]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size

        new_hyp_h_list=[]
        new_hyp_c_list=[]
        new_hyp_z_list=[]
        #new_hyp_alpha_list=[]
        new_hyp_word_list_embed=[]

        new_Qs_mask_list=[]
        new_hyp_all_h_list_list=[]
        new_hyp_hid_align_list_list=[]


        new_hyp_scores=[]
        new_hyp_word_list=[]

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_h=copy.copy(h_next_list[ti,:])
            new_hyp_c=copy.copy(c_next_list[ti,:])
            new_hyp_z=copy.copy(z_next_list[ti,:])
            #new_hyp_alpha=copy.copy(alpha_next_list[ti,:])
            new_Qs_mask=copy.copy(Qs_mask[ti,:])
            new_hyp_all_h_list=copy.copy(hyp_all_h_list[:,ti,:])
            new_hyp_hid_align_list=copy.copy(hyp_hid_align_list[:,ti,:])

            new_hyp_word_embed=sorted_vecs[wi]

            hyp_score=cand_scores[ti,wi]


            new_hyp_h_list.append(new_hyp_h)
            new_hyp_c_list.append(new_hyp_c)
            new_hyp_z_list.append(new_hyp_z)
            #new_hyp_alpha_list.append(new_hyp_alpha)
            new_Qs_mask_list.append(new_Qs_mask)
            new_hyp_all_h_list_list.append(new_hyp_all_h_list)
            new_hyp_hid_align_list_list.append(new_hyp_hid_align_list)

            new_hyp_word_list_embed.append(new_hyp_word_embed)
            new_hyp_scores.append(hyp_score)

            if len(hyp_word_list)==0:
                temp_hyp_word_list=[wi]
            else:
                temp_hyp_word_list=copy.copy(hyp_word_list[ti])
                temp_hyp_word_list.append(wi)

            new_hyp_word_list.append(temp_hyp_word_list)

        new_live_k=0

        hyp_h_list=[]
        hyp_c_list=[]
        hyp_z_list=[]
        #hyp_alpha_list=[]
        Qs_mask=[]
        hyp_all_h_list=[]
        hyp_hid_align_list=[]

        hyp_word_list_embed=[]

        hyp_scores=[]
        hyp_word_list=[]


        for idx in range(len(new_hyp_word_list)):
            if new_hyp_word_list[idx][-1]==voc_size-1:
                sample.append(new_hyp_word_list[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k=new_live_k+1
                hyp_h_list.append(new_hyp_h_list[idx])
                hyp_c_list.append(new_hyp_c_list[idx])
                hyp_z_list.append(new_hyp_z_list[idx])
                #hyp_alpha_list.append(new_hyp_alpha_list[idx])
                Qs_mask.append(new_Qs_mask_list[idx])
                hyp_all_h_list.append(new_hyp_all_h_list_list[idx])
                hyp_hid_align_list.append(new_hyp_hid_align_list_list[idx])

                hyp_word_list_embed.append(new_hyp_word_list_embed[idx])

                hyp_scores.append(new_hyp_scores[idx])
                hyp_word_list.append(new_hyp_word_list[idx])

        live_k=new_live_k

        if new_live_k < 1:
            break
        if dead_k >= Bsize:
            break

        hyp_scores=np.array(hyp_scores).astype(theano.config.floatX)

        hyp_h_list=np.array(hyp_h_list).astype(theano.config.floatX)
        hyp_c_list=np.array(hyp_c_list).astype(theano.config.floatX)
        hyp_z_list=np.array(hyp_z_list).astype(theano.config.floatX)
       # hyp_alpha_list=np.array(hyp_alpha_list).astype(theano.config.floatX)
        Qs_mask = np.array(Qs_mask).astype(theano.config.floatX)
        hyp_all_h_list = np.array(hyp_all_h_list).astype(theano.config.floatX)
        hyp_hid_align_list = np.array(hyp_hid_align_list).astype(theano.config.floatX)

        if hyp_all_h_list.ndim==3:
            hyp_all_h_list= np.transpose(hyp_all_h_list, (1, 0, 2))
            hyp_hid_align_list= np.transpose(hyp_hid_align_list, (1, 0, 2))
        else:
            hyp_all_h_list=  np.expand_dims(hyp_all_h_list, axis=1)
            hyp_hid_align_list= np.expand_dims(hyp_hid_align_list, axis=1)


        hyp_word_list_embed=np.array(hyp_word_list_embed).astype(theano.config.floatX)
        if hyp_word_list_embed.ndim==1:
            hyp_word_list_embed= np.expand_dims(hyp_word_list_embed, axis=0)

    if live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_word_list[idx])
            sample_score.append(hyp_scores[idx])


    return sample, sample_score

######################################################
def Init_Sentences_from_list(word_list,dict):

   Qs=[]
   for line in word_list:


        seg_list0 = jieba.cut(line)

        QQ = [w for w in seg_list0]


        temp_res_Q = [dict[w].index for w in QQ if w in dict]
        if len(temp_res_Q)<=max_T:
            Qs.append(temp_res_Q)
        else:
            Qs.append(temp_res_Q[0:max_T])

   return Qs #B* n_words


def Init_Sentences_from_listoflist(word_list,dict):

   Qss=[]
   for temp_list in word_list:
       Qs=[]
       for line in temp_list:

            seg_list0 = jieba.cut(line)

            QQ = [w for w in seg_list0]


            temp_res_Q = [dict[w].index for w in QQ if w in dict]
            if len(temp_res_Q)<=max_T_M:
                Qs.append(temp_res_Q)
            else:
                Qs.append(temp_res_Q[0:max_T_M])

       Qss.append(Qs)

   return Qss   #B*N_M* n_words

def prepare_files_Q(Qs_batch,word_end_inx):

    word_end_inx=word_end_inx-1

    Qs_lens = [len(tl) for tl in Qs_batch]
    #As_lens = [len(tl) for tl in As_batch]

    max_Qs = max(Qs_lens)
    #max_As = max(As_lens)+1

    batch_Q_word_list = []
    #batch_Q_word_list_reverse = []
    batch_Q_mask_list = []


    #batch_A_word_list = []
    #batch_A_mask_list = []

    for tll in range(len(Qs_batch)):

        temp_s=Qs_batch[tll]
        temp_len = len(temp_s)
        word_list = np.concatenate((np.asarray(temp_s,dtype='int32'), word_end_inx*np.ones(max_Qs-temp_len,dtype='int32')))
        #word_list_reverse = np.concatenate((np.asarray(temp_s,dtype='int32')[::-1], word_end_inx*np.ones(max_Qs-temp_len,dtype='int32')))
        mask_list = np.concatenate((np.ones(temp_len,dtype='int32'), np.zeros(max_Qs-temp_len,dtype='int32')))

        batch_Q_word_list.append(word_list)
        #batch_Q_word_list_reverse.append(word_list_reverse)
        batch_Q_mask_list.append(mask_list)




    return np.asarray(batch_Q_word_list,dtype='int32'),np.asarray(batch_Q_mask_list,dtype='float32')


################################################################################   testing

begin = time()
print "Loading data --------"
test_file_name='./test20.txt'
test_file= codecs.open(test_file_name)
test_words_list=[]
while True:
    line = test_file.readline()
    if line:
        sline = line.strip().decode('gbk').encode('utf8')
        test_words_list.append(sline)
    else:
        break

Q_list = Init_Sentences_from_list(test_words_list,dict)   ### index list
end = time()
print "Total loading time: %d seconds" % (end - begin)
print "--------"


Beam=1
strategy=1

for Beam in [1]:
    n = len(test_words_list)

    total_captions=[]
    total_eval_captions=[]

    for kk in range(n):
        print kk
        Q_in = [Q_list[kk]]

        W1, W2 = prepare_files_Q(Q_in,n_word_dict)

        scale_captions=[]
        for temp_scale in [1]:

            test_gen_captions=[]
            for t in [1,2,3,4,5,6,7,8,9,10]:
                temp_gen_captions, scores = generate_captions_perX(W1,W2, Beam,strategy,temp_scale)
                test_gen_captions.extend(temp_gen_captions)


            ############################################ for print json
            temp_dict = {}
            temp_dict['Q']=test_words_list[kk]

            temp_list = []
            for ii in range(len(test_gen_captions)):

                temp_words = test_gen_captions[ii]
                temp_str = ''
                for jj in range(len(temp_words)):
                    temp_word = dict_index2word[temp_words[jj].astype(np.int32)]
                    if temp_word != u'EOF':
                        temp_str = temp_str + temp_word.encode('utf8')  #.encode('utf-8')
                    else:
                        break
                temp_list.append(temp_str)

            #print temp_list
            temp_dict['Scale_Z']=str(temp_scale)
            temp_dict['QAs']=temp_list

            scale_captions.append(temp_dict)

        total_captions.append(scale_captions)

    end = time()
    print "test time: %d seconds" % (end - begin)
    print "--------"

    # writing txt result
    file = './samples/%s/test_linespaceZ_%s_epoach_%d_strategy_%d_Beam_%d.txt'%(desc,desc,select_epochs,strategy,Beam)

    fp = open(file,'w')
    fp.write(json.dumps(total_captions, ensure_ascii=False))
    fp.close()
    # WRINTING xml RESult
    file_xml = './samples/%s/test_linespaceZ_%s_epoach_%d_strategy_%d_Beam_%d.xlsx'%(desc,desc,select_epochs,strategy,Beam)
    wb= Workbook()
    sheet=wb.active
    ii=1
    jj=0
    for temp_caption in total_captions:
        jj+=1
        if jj<50:
            for temp_dict in temp_caption:
                str_Q=temp_dict['Q']
                ii0=ii
                for temp_a in temp_dict['QAs']:
                    ii+=1
                    sheet["A%d"%ii].value=str_Q
                    sheet["B%d"%ii].value=temp_a

                sheet.merge_cells('F%d:F%d'%(ii0+1,ii0+10))

    sheet["A1"].value='Post'
    sheet["B1"].value='Response'
    sheet["C1"].value='Good'
    sheet["D1"].value='Normal'
    sheet["E1"].value='Bad'
    sheet["F1"].value='Diverse'
    wb.save(file_xml)


    file = './eval/res_%s_beam_%d_%d.json'%(desc,Beam,select_epochs)
    evalFile = './eval/eval_%s_beam_%d_%d.json'%(desc,Beam,select_epochs)

    cocoEval = COCOEvalCap(my_bleu, my_meteor,my_rouge,my_cider)
    cocoEval.evaluate(As_test_in_Q,total_eval_captions)
    #json.dump(cocoEval.evalImgs, open(evalImgsFile, 'w'))
    json.dump(cocoEval.eval,  open(evalFile, 'w'))
    ##############################
    fp = open(file,'w')
    fp.write(json.dumps([item[0].encode('utf8') for item in total_eval_captions],ensure_ascii=False))
    fp.close()

    print 'writing caption files epoch %d  ok'%(select_epochs)




