#coding=utf-8
import sys
sys.path.append('.')
import os
os.environ['MKL_NUM_THREADS'] = '8'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'   #for debugging
import json
from time import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import six.moves.cPickle as pickle
import gensim
import random
import jieba

import numpy as np
import theano
import theano.tensor as T
#from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import updates
from lib import inits
#from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng,t_rng,t_rng_cpu
from lib.theano_utils import floatX, sharedX

from theano.printing import pydotprint

#################################################### make result dir
desc = 'weibo_model_v3'
model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)
print desc.upper()
#################################################### load Word2Vec model
model = gensim.models.Word2Vec.load("./word2vec_gensim")

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

nbatch = 20      # # of examples in batch
max_T=30          # # sentense length
max_T_M=20        # # sentense length of M
n_LSTM=500       # # of LSTM_hidden_units    #1000 1500
dimAttention= 100
N_M=2

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
#gifn_CPU = inits.Normal_CPU(scale=0.01)

gain_ifn = inits.Normal(loc=1., scale=0.01)
bias_ifn = inits.Constant(c=0.)

startword_ifn = inits.Constant(c=-7.)

#bias_ifn_CPU = inits.Constant_CPU(c=0.)
###################################
First=True
if First:
    select_epochs=0

    word_start = startword_ifn((1, 1, n_word_dim), 'word_start')

    shared_Word_vecs = sharedX(sorted_vecs)#T._shared(sorted_vecs, borrow=True)      #   sharedX(sorted_vecs)              # force on CPU

    LSTM_hidden0 = gifn((1,n_LSTM), 'LSTM_hidden0')
    LSTM_hidden0_rev = gifn((1, n_LSTM), 'LSTM_hidden0_rev')

    ########## encoder params :
    W_LSTM_hidden_enc = orfn((n_LSTM,4*n_LSTM), 'W_LSTM_hidden_enc')
    W_LSTM_in_enc     = gifn((n_word_dim,4*n_LSTM), 'W_LSTM_in_enc')
    b_LSTM_enc        = bias_ifn((4*n_LSTM), 'b_LSTM_enc')

    W_LSTM_hidden_enc_rev = orfn((n_LSTM,4*n_LSTM), 'W_LSTM_hidden_enc_rev')
    W_LSTM_in_enc_rev     = gifn((n_word_dim,4*n_LSTM), 'W_LSTM_in_enc_rev')
    b_LSTM_enc_rev        = bias_ifn((4*n_LSTM), 'b_LSTM_enc_rev')


    W_LSTM_hidden_gen = orfn((n_LSTM,4*n_LSTM), 'W_LSTM_hidden_gen')
    W_LSTM_in_gen     = gifn((n_word_dim+n_LSTM,4*n_LSTM), 'W_LSTM_in_gen')
    b_LSTM_gen        = bias_ifn((4*n_LSTM), 'b_LSTM_gen')


    W_init_h0=gifn((n_LSTM, n_LSTM),'W_init_h0')
    b_init_h0= bias_ifn((n_LSTM), 'b_init_h0')
    W_init_c0=gifn((n_LSTM, n_LSTM),'W_init_c0')
    b_init_c0= bias_ifn((n_LSTM), 'b_init_c0')

    ###used for VAE_sentence
    W1_M0= gifn((2*n_LSTM,n_LSTM), 'W1_M')
    b1_M0=bias_ifn((n_LSTM), 'b1_M')
    W2_M0=gifn((2*n_LSTM,n_LSTM), 'W2_M')
    WM_mu_zt0=gifn((n_LSTM,n_LSTM), 'WM_mu_zt')
    bM_mu_zt0=bias_ifn((n_LSTM), 'bM_mu_zt')
    WM_sigma_zt0=gifn((n_LSTM,n_LSTM), 'WM_sigma_zt')
    bM_sigma_zt0=bias_ifn((n_LSTM), 'bM_sigma_zt')

    W3_M0=gifn((2*n_LSTM,n_LSTM), 'W3_M')
    b3_M0=bias_ifn((n_LSTM), 'b3_M')
    Wp_M_mu0=gifn((n_LSTM,n_LSTM), 'Wp_M_mu')
    bp_M_mu0=bias_ifn((n_LSTM), 'bp_M_mu')
    Wp_M_sigma0=gifn((n_LSTM,n_LSTM), 'Wp_M_sigma')
    bp_M_sigma0=bias_ifn((n_LSTM), 'bp_M_sigma')

    ###used for VAE_word
    W1_M= gifn((n_LSTM,n_LSTM), 'W1_M')
    b1_M=bias_ifn((n_LSTM), 'b1_M')
    W2_M=gifn((n_word_dim,n_LSTM), 'W2_M')
    WM_mu_zt=gifn((n_LSTM,n_LSTM), 'WM_mu_zt')
    bM_mu_zt=bias_ifn((n_LSTM), 'bM_mu_zt')
    WM_sigma_zt=gifn((n_LSTM,n_LSTM), 'WM_sigma_zt')
    bM_sigma_zt=bias_ifn((n_LSTM), 'bM_sigma_zt')

    W3_M=gifn((n_LSTM,n_LSTM), 'W3_M')
    b3_M =bias_ifn((n_LSTM), 'b3_M')
    Wp_M_mu=gifn((n_LSTM,n_LSTM), 'Wp_M_mu')
    bp_M_mu=bias_ifn((n_LSTM), 'bp_M_mu')
    Wp_M_sigma=gifn((n_LSTM,n_LSTM), 'Wp_M_sigma')
    bp_M_sigma=bias_ifn((n_LSTM), 'bp_M_sigma')


    ########## attention params :
    U_attention_gen = gifn((2*n_LSTM, dimAttention),'U_attention_gen')
    W_attention_gen = gifn((n_LSTM+n_word_dim, dimAttention),'W_attention_gen')
    b_attention_gen = bias_ifn((dimAttention),'b_attention_gen')
    v_attention_gen = gifn((dimAttention),'v_attention_gen')


    W_word_gen      = gifn((4*n_LSTM,n_LSTM), 'W_word_gen')
    b_word_gen      = bias_ifn((n_LSTM), 'b_word_gen')

    W_softmax_gen   = gifn((n_LSTM,n_word_dict), 'W_softmax_gen')       # force on CPU: gifn_CPU
    b_softmax_gen   = bias_ifn((n_word_dict), 'b_softmax_gen')         # force on CPU: bias_ifn_CPU
    ########## Bow params :
    W_bow1=gifn((n_LSTM,n_LSTM), 'W_bow1')
    b_bow1=bias_ifn((n_LSTM), 'b_bow1')
    W_bow2=gifn((n_LSTM,n_LSTM), 'W_bow2')
    b_bow2=bias_ifn((n_LSTM), 'b_bow2')
    W_softmax_bow=gifn((n_LSTM,n_word_dict), 'W_softmax_bow')
    b_softmax_bow=bias_ifn((n_word_dict), 'b_softmax_bow')

    ########## Bow in T  params :
    W_bow1t=gifn((n_LSTM,n_LSTM), 'W_bow1t')
    b_bow1t=bias_ifn((n_LSTM), 'b_bow1t')
    W_bow2t=gifn((n_LSTM,n_LSTM), 'W_bow2t')
    b_bow2t=bias_ifn((n_LSTM), 'b_bow2t')
    W_softmax_bowt=gifn((n_LSTM,n_word_dict), 'W_softmax_bow')
    b_softmax_bowt=bias_ifn((n_word_dict), 'b_softmax_bow')
    ##########
    enc_params =   [LSTM_hidden0, W_LSTM_hidden_enc, W_LSTM_in_enc, b_LSTM_enc,
                    LSTM_hidden0_rev, W_LSTM_hidden_enc_rev, W_LSTM_in_enc_rev, b_LSTM_enc_rev]

    gen_params =   [U_attention_gen , W_attention_gen, b_attention_gen, v_attention_gen,
                      W_init_h0, b_init_h0, W_init_c0, b_init_c0,
                      W1_M, b1_M, W2_M, WM_mu_zt, bM_mu_zt, WM_sigma_zt, bM_sigma_zt, W3_M, b3_M , Wp_M_mu, bp_M_mu, Wp_M_sigma, bp_M_sigma,
                      W1_M0, b1_M0, W2_M0, WM_mu_zt0, bM_mu_zt0, WM_sigma_zt0, bM_sigma_zt0, W3_M0, b3_M0 , Wp_M_mu0, bp_M_mu0, Wp_M_sigma0, bp_M_sigma0,
                     W_LSTM_hidden_gen, W_LSTM_in_gen, b_LSTM_gen,W_word_gen, b_word_gen,W_softmax_gen, b_softmax_gen,
                    W_bow1, b_bow1, W_bow2,  b_bow2,  W_softmax_bow,  b_softmax_bow,
                    W_bow1t, b_bow1t, W_bow2t,  b_bow2t,  W_softmax_bowt,  b_softmax_bowt]

    total_params=[]
    #total_params.append(shared_Word_vecs)
    total_params.extend(enc_params)
    total_params.extend(gen_params)

else:

    total_params=[]

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

    h_t_lang = T.concatenate([h_list, h_list_rev[::-1,:,:]], axis=2)

    gen_init0_lang=T.concatenate([h_list[-1], h_list_rev[-1]], axis=1)

    return h_t_lang, gen_init0_lang

######################################

def generate_captions(As_words, As_masks, h_enc , gen_init0_lang, gen_init0_lang_Y ,Qs_masks , U_attention_gen , W_attention_gen, b_attention_gen, v_attention_gen,
                      W_init_h0, b_init_h0, W_init_c0, b_init_c0,
                      W1_M, b1_M, W2_M, WM_mu_zt, bM_mu_zt, WM_sigma_zt, bM_sigma_zt, W3_M, b3_M , Wp_M_mu, bp_M_mu, Wp_M_sigma, bp_M_sigma,
                      W1_M0, b1_M0, W2_M0, WM_mu_zt0, bM_mu_zt0, WM_sigma_zt0, bM_sigma_zt0, W3_M0, b3_M0 , Wp_M_mu0, bp_M_mu0, Wp_M_sigma0, bp_M_sigma0,
                     W_LSTM_hidden_gen, W_LSTM_in_gen, b_LSTM_gen,W_word_gen, b_word_gen,W_softmax_gen, b_softmax_gen,
                      W_bow1, b_bow1, W_bow2,  b_bow2,  W_softmax_bow,  b_softmax_bow,
                      W_bow1t, b_bow1t, W_bow2t,  b_bow2t,  W_softmax_bowt,  b_softmax_bowt):
    ###Discourse - level###
    ###calculate Q(zd|Y,X)  : X gen_init0_lang   Y gen_init0_lang_Y
    m_10 = lrelu(T.dot(gen_init0_lang, W1_M0) +T.dot(gen_init0_lang_Y, W2_M0) +b1_M0)  #  batch_size x 2*lstm    lstm

    u_zt0= T.dot(m_10, WM_mu_zt0) + bM_mu_zt0   # batch_size x lstm
    log_sigma_zt0= T.dot(m_10, WM_sigma_zt0) + bM_sigma_zt0

    #sample Q(Zd)
    eps0 = t_rng.normal(size=(u_zt0.shape[0] , u_zt0.shape[1]), avg=0.0, std=1.0, dtype=theano.config.floatX)
    Zt0 = u_zt0 + T.exp(log_sigma_zt0) * eps0  #batch_size x dim_atten

    ########################calculate BOWs loss

    t_bow1=lrelu(T.dot(Zt0, W_bow1) + b_bow1)  #batch * middle_dim  W_bow1, b_bow1, W_bow2,  b_bow2,  W_softmax_bow,  b_softmax_bow
    t_bow2=lrelu(T.dot(t_bow1, W_bow2) + b_bow2)
    word_soft_bow=T.dot(t_bow2, W_softmax_bow)+b_softmax_bow
    bow_K=T.nnet.softmax(word_soft_bow)

    #calculate p(Zd)
    h_prior_00=lrelu(T.dot(gen_init0_lang, W3_M0) + b3_M0)   #batch_size x dim_atten
    u_0t0=T.dot(h_prior_00, Wp_M_mu0) + bp_M_mu0
    log_sigma_0t0=T.dot(h_prior_00,Wp_M_sigma0) + bp_M_sigma0


    #calculate KL_d
    KL_t0= (log_sigma_0t0-log_sigma_zt0)+((T.exp(2*log_sigma_zt0)+(u_zt0-u_0t0)**2)/(2*T.exp(2*log_sigma_0t0)))-0.5
    KL_t0=T.sum(KL_t0)

    KL_t0= (KL_t0 / u_0t0.shape[0]).astype(theano.config.floatX)


    LSTM_h0=T.tanh(T.dot(Zt0, W_init_h0)+b_init_h0)
    cell0=T.tanh(T.dot(Zt0, W_init_c0)+b_init_c0)

    word0= (T.extra_ops.repeat(word_start, repeats=As_words.shape[1], axis=1)).astype(theano.config.floatX)


    this_real_words=T.concatenate([word0, As_words], axis=0)

    eps_list = t_rng.normal(size=(As_masks.shape[0],Zt0.shape[0],Zt0.shape[1]), avg=0.0, std=1.0, dtype=theano.config.floatX)


    def recurrence(word_t_prior,word_t,t_mask,eps,h_t_prior,c_t_prior,z_t_prior,W_LSTM_in_gen,W_LSTM_hidden_gen,b_LSTM_gen,
                   W1_M,W2_M,b1_M,WM_mu_zt,bM_mu_zt,WM_sigma_zt,bM_sigma_zt,W3_M,b3_M,Wp_M_mu,bp_M_mu,Wp_M_sigma,bp_M_sigma
                   ):


        ################################################ calculate input
        word_t_prior = T.concatenate([word_t_prior, z_t_prior], axis=1)

        lstm_t = T.dot(h_t_prior, W_LSTM_hidden_gen) + T.dot(word_t_prior, W_LSTM_in_gen)+ b_LSTM_gen
        i_t_enc = T.nnet.sigmoid(lstm_t[:, 0*n_LSTM:1*n_LSTM])
        f_t_enc = T.nnet.sigmoid(lstm_t[:, 1*n_LSTM:2*n_LSTM])


        cell_t_enc = f_t_enc * c_t_prior + i_t_enc * T.tanh(lstm_t[:, 2*n_LSTM:3*n_LSTM])
        cell_t_enc = t_mask.dimshuffle([0, 'x']) * cell_t_enc + (1. - t_mask.dimshuffle([0, 'x'])) * c_t_prior

        o_t_enc = T.nnet.sigmoid(lstm_t[:, 3*n_LSTM:4*n_LSTM])

        h_t = o_t_enc * T.tanh(cell_t_enc)
        h_t = t_mask.dimshuffle([0, 'x']) * h_t + (1. - t_mask.dimshuffle([0, 'x'])) * h_t_prior


        ###################################Word - level###

        m_1 = lrelu(T.dot(h_t, W1_M)+T.dot(word_t, W2_M) + b1_M)  #  using h_t    T_dec  x batch_size x dim_atten

        u_zt= T.dot(m_1, WM_mu_zt) + bM_mu_zt   #T_dec  x batch_size x dim_atten
        log_sigma_zt= T.dot(m_1, WM_sigma_zt) + bM_sigma_zt

        #sample Q(Zwt)

        z_w_t = u_zt + T.exp(log_sigma_zt) * eps  #T_dec  x batch_size x dim_atten


        #calculate p(Zwt)
        h_prior_0=lrelu(T.dot(h_t, W3_M) + b3_M)   #T_dec  x batch_size x dim_atten
        u_0t=T.dot(h_prior_0, Wp_M_mu) + bp_M_mu
        log_sigma_0t=T.dot(h_prior_0,Wp_M_sigma) + bp_M_sigma


        #calculate KL_t  using : mask_t[:, None]
        KL_t= (log_sigma_0t-log_sigma_zt)+((T.exp(2*log_sigma_zt)+(u_zt-u_0t)**2)/(2*T.exp(2*log_sigma_0t)))-0.5
        KL_t=T.sum(KL_t * t_mask.dimshuffle([0,'x']))

        KL_t= (KL_t / h_t.shape[0]).astype(theano.config.floatX)


        return h_t.astype(theano.config.floatX) ,cell_t_enc.astype(theano.config.floatX),z_w_t.astype(theano.config.floatX),KL_t.astype(theano.config.floatX)

    (h_list , _, Zt, KL_t_list ), _ = theano.scan(recurrence,sequences=[this_real_words[0:-1],As_words,As_masks,eps_list],
                                                    outputs_info=[LSTM_h0,cell0,Zt0,None],
                                                    non_sequences=[W_LSTM_in_gen,W_LSTM_hidden_gen,b_LSTM_gen,W1_M,W2_M,b1_M,WM_mu_zt,bM_mu_zt,WM_sigma_zt,bM_sigma_zt,W3_M,b3_M,Wp_M_mu,bp_M_mu,Wp_M_sigma,bp_M_sigma],
                                                    n_steps=As_masks.shape[0],
                                                    strict=True)



    hid_align = T.dot(h_enc, U_attention_gen)  # T_enc*Batch* dimAtten

    h_t_info = T.concatenate([Zt, this_real_words[0:-1]], axis=2)  # T_dec*Batch* (n_LSTM+dim word)

    hdec_align = T.dot(h_t_info, W_attention_gen)  # T_dec*Batch* dimAtten

    all_align = T.tanh(hid_align.dimshuffle([0,'x', 1, 2]) + hdec_align.dimshuffle(['x', 0, 1, 2]) + b_attention_gen.dimshuffle(['x','x', 'x', 0]))
    # T_enc x T_dec x batch_size x dimAttention

    e = all_align * v_attention_gen.dimshuffle(['x','x','x',0])
    e = e.sum(axis=3) * Qs_masks.dimshuffle([0,'x', 1]) # (T_enc_2M) x T_dec x batch_size
    e = e.dimshuffle([1, 2, 0]) # T_dec x batch_size x T_enc

    e2= T.reshape(e,[e.shape[0]*e.shape[1],e.shape[2]],ndim=2) # (T_dec x batch_size) x T_enc

    # normalize
    alpha = T.nnet.softmax(e2) #  #  (T_dec x batch_size) * T_enc

    alpha = T.reshape(alpha, [e.shape[0], e.shape[1] , e.shape[2]], ndim=3)  #  T_dec x batch_size * T_enc

    attention_enc = alpha.dimshuffle([0, 2, 1, 'x']) * h_enc.dimshuffle(['x', 0, 1, 2]) #  T_dec x T_enc x batch_size x h_dim
    attention_enc = attention_enc.sum(axis=1) # T_dec x T_enc x batch_size x h_dim --> T_dec  x batch_size x h_dim

    ################################  word
    prepare_word=T.concatenate([attention_enc,h_list,Zt], axis=2)

    word_t=lrelu(T.dot(prepare_word, W_word_gen) + b_word_gen)  #T * batch * middle_dim
    word_soft=T.dot(word_t, W_softmax_gen)+b_softmax_gen
    word_soft_K=T.nnet.softmax(T.reshape(word_soft,[word_soft.shape[0]*word_soft.shape[1], word_soft.shape[2]],ndim=2))

    ################################# Auxiliary-path

    t_bow1t=lrelu(T.dot(Zt, W_bow1t) + b_bow1t)  #batch * middle_dim  W_bow1, b_bow1, W_bow2,  b_bow2,  W_softmax_bow,  b_softmax_bow
    t_bow2t=lrelu(T.dot(t_bow1t, W_bow2t) + b_bow2t)
    word_soft_bowt=T.dot(t_bow2t, W_softmax_bowt)+b_softmax_bowt
    word_soft_K_Zt=T.nnet.softmax(T.reshape(word_soft_bowt,[word_soft_bowt.shape[0]*word_soft_bowt.shape[1], word_soft_bowt.shape[2]],ndim=2))

    return word_soft_K,(KL_t0).astype(theano.config.floatX),(T.sum(KL_t_list)).astype(theano.config.floatX),(bow_K).astype(theano.config.floatX),word_soft_K_Zt.astype(theano.config.floatX) ### (T *batch ) * n_word_dict

####################################################
KL_weight = T.scalar('KL_weight', dtype='float32')

KL_weight.tag.test_value = 1
#################################################### # batch * T
Qs_word_list = T.matrix('Qs_word_list', dtype='int32')  # batch * T
Qs_mask = T.matrix('Qs_mask', dtype='float32')  # batch * T
As_word_list = T.matrix('As_word_list', dtype='int32')  # batch * T
As_mask = T.matrix('As_mask', dtype='float32')  # batch * T

# provide Theano with a default test-value
Qs_word_list.tag.test_value = np.random.randint(1000,size=(nbatch,max_T)).astype(np.int32)
As_word_list.tag.test_value = np.random.randint(1000,size=(nbatch,max_T)).astype(np.int32)

Qs_mask.tag.test_value = np.random.randint(1,size=(nbatch,max_T)).astype(np.float32)
As_mask.tag.test_value = np.random.randint(1,size=(nbatch,max_T)).astype(np.float32)

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
Qns_word_vecs_in= Qns_word_vecs.dimshuffle([1, 0, 2])
Qns_mask_in= Qns_mask.reshape([Qns_mask.shape[0]* Qns_mask.shape[1], Qns_mask.shape[2]]) #(batch *M) * T
_, hQns_enc_end = encoder_network(Qns_word_vecs_in,Qns_mask_in.T,*enc_params)  # T *(batch *M) * (2*n_LSTM),      (batch *M) * (2*n_LSTM)
hQns_enc_end= hQns_enc_end.reshape([Qns_word_list.shape[0],Qns_word_list.shape[1],hQns_enc_end.shape[1]]) #batch *M * (2*n_LSTM)

####################################################  encode AM

Ans_word_list_flat = T.flatten(Ans_word_list,ndim=1) #
Ans_word_vecs = shared_Word_vecs[Ans_word_list_flat].reshape([Ans_word_list.shape[0]* Ans_word_list.shape[1], Ans_word_list.shape[2], n_word_dim]) # (batch* M) * T* n_dim
Ans_word_vecs_in= Ans_word_vecs.dimshuffle([1, 0, 2])
Ans_mask_in= Ans_mask.reshape([Ans_mask.shape[0]* Ans_mask.shape[1], Ans_mask.shape[2]]) #(batch *M) * T
_, hAns_enc_end = encoder_network(Ans_word_vecs_in,Ans_mask_in.T,*enc_params)  # T *(batch *M) * n_LSTM,      (batch *M) * (2*n_LSTM)
hAns_enc_end= hAns_enc_end.reshape([Ans_word_list.shape[0],Ans_word_list.shape[1],hAns_enc_end.shape[1]]) #batch *M * (2*n_LSTM)
Total_M0= T.concatenate([hQns_enc_end, hAns_enc_end], axis=1)  #batch * 2M * (2*n_LSTM)
Total_M = Total_M0.sum(axis=1)  #batch  * (2*n_LSTM)



#################################################### encode  decode
Qs_word_list_flat = T.flatten(Qs_word_list.T,ndim=1) #
Qs_word_vecs = shared_Word_vecs[Qs_word_list_flat].reshape([Qs_word_list.shape[1], Qs_word_list.shape[0], n_word_dim]) # T * batch * n_dim

As_word_list_flat = T.flatten(As_word_list.T,ndim=1) #words x #samples
As_word_vecs = shared_Word_vecs[As_word_list_flat].reshape([As_word_list.shape[1], As_word_list.shape[0], n_word_dim]) # T * batch * n_dim

h_enc,gen_init0_lang = encoder_network(Qs_word_vecs,Qs_mask.T, *enc_params)  # h_enc: T * batch * (2*n_LSTM)
####################################################  encode Y
h_enc_Y,gen_init0_lang_Y = encoder_network(As_word_vecs,As_mask.T, *enc_params)  # h_enc: T * batch * (2*n_LSTM)

Total_M_h_enc= T.concatenate([Total_M0.dimshuffle([1, 0, 2]),h_enc], axis=0)
Qs_mask_in= T.concatenate([T.ones((Total_M0.shape[1],Total_M0.shape[0]),dtype=theano.config.floatX), Qs_mask.T], axis=0)   # Qs_mask: batch * T

word_K_list,KL_cost0, KL_cost_t,bow_K,word_K_list_ZT = generate_captions(As_word_vecs,As_mask.T,Total_M_h_enc,gen_init0_lang,gen_init0_lang_Y,Qs_mask_in,*gen_params) #T *batch * n_word_dict


#################################################### bow   cost
large_matrix=T.ones((As_word_list.shape[1],As_word_list.shape[0]),dtype=theano.config.floatX)  # As_word_list: batch * T
T_bow_K=bow_K.dimshuffle(['x', 0, 1])*large_matrix.dimshuffle([ 0, 1,'x'])   #T * batch * n_word_dict

T_bow_K_flat = T.flatten(T_bow_K,ndim=1)

bow_cost1 = -T.log(T_bow_K_flat[T.arange(As_word_list_flat.shape[0])*n_word_dict+As_word_list_flat]+1e-7)
bow_cost_re = T.reshape(bow_cost1,[As_word_list.shape[1], As_word_list.shape[0]],ndim=2) #T *batch
cost1_bow=bow_cost_re*As_mask.T#T *batch
cost2_bow=cost1_bow.sum(axis=0)#/Mask_captions.sum(axis=0)
cost3_bow=cost2_bow.mean()
#################################################### bow T  cost
'''
T_bow_KT=bow_KT.dimshuffle(['x', 0, 1])*large_matrix.dimshuffle([ 0, 1,'x'])   #T * batch * n_word_dict

T_bow_K_flatT = T.flatten(T_bow_KT,ndim=1)

bow_cost1T = -T.log(T_bow_K_flatT[T.arange(As_word_list_flat.shape[0])*n_word_dict+As_word_list_flat]+1e-7)
bow_cost_reT = T.reshape(bow_cost1T,[As_word_list.shape[1], As_word_list.shape[0]],ndim=2) #T *batch
cost1_bowT=bow_cost_reT*As_mask.T#T *batch
cost2_bowT=cost1_bowT.sum(axis=0)#/Mask_captions.sum(axis=0)
cost3_bowT=cost2_bowT.mean()
'''

#################################################### encode  decode   cost
word_K_list_flat = T.flatten(word_K_list,ndim=1)
cost = -T.log(word_K_list_flat[T.arange(As_word_list_flat.shape[0])*n_word_dict+As_word_list_flat]+1e-7) #tensor.arange(x_flat.shape[0])   *  probs.shape[1]  +  x_flat
cost_re = T.reshape(cost,[As_word_list.shape[1], As_word_list.shape[0]],ndim=2) #T *batch

cost1=cost_re*As_mask.T#T *batch
cost2=cost1.sum(axis=0)#/Mask_captions.sum(axis=0)
cost3=cost2.mean()

######################################################
word_K_list_flat_ZT = T.flatten(word_K_list_ZT,ndim=1)
cost_ZT = -T.log(word_K_list_flat_ZT[T.arange(As_word_list_flat.shape[0])*n_word_dict+As_word_list_flat]+1e-7) #tensor.arange(x_flat.shape[0])   *  probs.shape[1]  +  x_flat
cost_re_ZT = T.reshape(cost_ZT,[As_word_list.shape[1], As_word_list.shape[0]],ndim=2) #T *batch

cost1_ZT=cost_re_ZT*As_mask.T#T *batch
cost2_ZT=cost1_ZT.sum(axis=0)#/Mask_captions.sum(axis=0)
cost3_ZT=cost2_ZT.mean()

cost4=cost3+ (KL_cost0+KL_cost_t)*KL_weight+cost3_bow*alpha+ beta*cost3_ZT

lrt = sharedX(learning_rate)
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2),clipnorm=10)
g_updates = g_updater(total_params, cost4)


print 'COMPILING'
t = time()


_train = theano.function([KL_weight,Qs_word_list,As_word_list,Qs_mask,As_mask,Qns_word_list,Ans_word_list,Qns_mask,Ans_mask], [cost4,cost3,KL_cost0,KL_cost_t,cost3_bow], updates=g_updates)#, profile=True)
print '%.2f seconds to compile theano functions'%(time()-t)
print 'finish printing'

#####################################
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

def prepare_files(Qs_batch,As_batch, QM, AM ,word_end_inx):

    word_end_inx=word_end_inx-1

    Qs_lens = [len(tl) for tl in Qs_batch]
    As_lens = [len(tl) for tl in As_batch]

    max_Qs = max(Qs_lens)
    max_As = max(As_lens)+1

    batch_Q_word_list = []
    #batch_Q_word_list_reverse = []
    batch_Q_mask_list = []


    batch_A_word_list = []
    batch_A_mask_list = []

    for tll in range(len(Qs_batch)):

        temp_s=Qs_batch[tll]
        temp_len = len(temp_s)
        word_list = np.concatenate((np.asarray(temp_s,dtype='int32'), word_end_inx*np.ones(max_Qs-temp_len,dtype='int32')))
        word_list_reverse = np.concatenate((np.asarray(temp_s,dtype='int32')[::-1], word_end_inx*np.ones(max_Qs-temp_len,dtype='int32')))
        mask_list = np.concatenate((np.ones(temp_len,dtype='int32'), np.zeros(max_Qs-temp_len,dtype='int32')))

        batch_Q_word_list.append(word_list)
        #batch_Q_word_list_reverse.append(word_list_reverse)
        batch_Q_mask_list.append(mask_list)

        temp_s=As_batch[tll]
        temp_len = len(temp_s)
        word_list = np.concatenate((np.asarray(temp_s,dtype='int32'), word_end_inx*np.ones(max_As-temp_len,dtype='int32')))

        mask_list = np.concatenate((np.ones(temp_len+1,dtype='int32'), np.zeros(max_As-temp_len-1,dtype='int32')))

        batch_A_word_list.append(word_list)
        batch_A_mask_list.append(mask_list)


    #########
    QM_lens = [[len(tl) for tl in temp_QM] for temp_QM in QM]
    AM_lens = [[len(tl) for tl in temp_AM] for temp_AM in AM]

    max_QMs = np.asarray(QM_lens).max()
    max_AMs = np.asarray(AM_lens).max()

    batch_QM_word_list = []
    batch_QM_mask_list = []


    batch_AM_word_list = []
    batch_AM_mask_list = []

    for tll0 in range(len(QM)):

        temp_QM=QM[tll0]
        temp_AM=AM[tll0]

        QM_word_list = []
        QM_mask_list = []


        AM_word_list = []
        AM_mask_list = []

        for tll in range(len(temp_QM)):

            temp_s=temp_QM[tll]
            temp_len = len(temp_s)
            #print tll0, max_QMs, temp_len, temp_s,QM_lens
            word_list = np.concatenate((np.asarray(temp_s,dtype='int32'), word_end_inx*np.ones(max_QMs-temp_len,dtype='int32')))

            mask_list = np.concatenate((np.ones(temp_len,dtype='int32'), np.zeros(max_QMs-temp_len,dtype='int32')))

            QM_word_list.append(word_list)
            QM_mask_list.append(mask_list)
            ######
            temp_s=temp_AM[tll]
            temp_len = len(temp_s)
            #print tll0,tll, max_AMs, temp_len, temp_s, AM_lens#, temp_AM
            word_list = np.concatenate((np.asarray(temp_s,dtype='int32'), word_end_inx*np.ones(max_AMs-temp_len,dtype='int32')))

            mask_list = np.concatenate((np.ones(temp_len,dtype='int32'), np.zeros(max_AMs-temp_len,dtype='int32')))

            AM_word_list.append(word_list)
            AM_mask_list.append(mask_list)

        batch_QM_word_list.append(QM_word_list)
        batch_QM_mask_list.append(QM_mask_list)
        batch_AM_word_list.append(AM_word_list)
        batch_AM_mask_list.append(AM_mask_list)

    return np.asarray(batch_Q_word_list,dtype='int32'),\
           np.asarray(batch_Q_mask_list,dtype='float32'),\
           np.asarray(batch_A_word_list, dtype='int32'),\
           np.asarray(batch_A_mask_list, dtype='float32'),\
           np.asarray(batch_QM_word_list,dtype='int32'),\
           np.asarray(batch_QM_mask_list,dtype='float32'),\
           np.asarray(batch_AM_word_list, dtype='int32'),\
           np.asarray(batch_AM_mask_list, dtype='float32')
################################################################################   training

import math


def weight1(x):
  #return 1 / (1 + math.exp(-(0.00005*(x-205000))))
  return 1 / (1 + math.exp(-(0.0002*(x-55000))))

def weight2(x):
  return 1 / (1 + math.exp(-(0.00005*(x-505000))))

def weight3(x):
  return 1 / (1 + math.exp(-(0.000015*(x-805000))))

def weight4(x):
  return 1 / (1 + math.exp(-(0.00005*(x-805000))))

def weight5(x):
  return 1 / (1 + math.exp(-(0.00005*(x-1105000))))

def weight6(x):
  return 1 / (1 + math.exp(-(0.00005*(x-105000))))
  #return 1 / (1 + math.exp(-(0.0002*(x-55000))))
def weight7(x):
  return 1 / (1 + math.exp(-(0.00005*(x-205000))))

def weight8(x):
  return 1 / (1 + math.exp(-(0.00005*(x-305000))))

def weight9(x):
  return 1 / (1 + math.exp(-(0.00002*(x-405000))))

niter=80

training_files=['./Total_chat_corpu.pkl'
                ]
loss_curve=[]
p_y_loss_curve=[]
KL_loss_curve=[]
KL_loss_curve0=[]
KL_loss_curvet=[]
BOWs_loss_curve=[]
Z_Squre_loss_curve=[]

num_updates=0
for epoch in range(niter):

    begin = time()

    for temp_g in range(len(training_files)):

        begin = time()
        print "Loading data --------"
        Q_list = pickle.load(open(training_files[temp_g], 'rb'))
        end = time()
        print "Total loading group %s : %d seconds" % (training_files[temp_g],end - begin)
        print "--------"

        #Q_list=Q_list[0:30]

        n = len(Q_list)
        batches = n / nbatch

        temp_index=np.random.permutation(n)#.astype(np.int32)

        #loss_curve=[]

        begin1 = time()
        begin3 = time()

        for kk in range(batches):

            start = kk * nbatch
            end = (kk + 1) * nbatch
            if end > n:
                end = n

            select_index=temp_index[int(start):int(end)]

            Q_in_Q_batch = [Q_list[i]['Q'] for i in select_index]  #list
            A_in_A_batch = [random.sample(Q_list[i]['As'],1)[0] for i in select_index]  #list

            Q_in=Init_Sentences_from_list(Q_in_Q_batch,dict)
            A_in=Init_Sentences_from_list(A_in_A_batch,dict)

            selected_M= random.sample(range(10), N_M)

            QM_in_Q_batch= [[Q_list[i]['Qs_K'][j] for j in  selected_M] for i in select_index]  #list of list   Batch * M
            AM_in_A_batch= [[Q_list[i]['As_K'][j] for j in  selected_M] for i in select_index]

            Q_M_in = Init_Sentences_from_listoflist(QM_in_Q_batch,dict)
            A_M_in = Init_Sentences_from_listoflist(AM_in_A_batch,dict)


            #W1, W2, W3, W4, W5 = prepare_files(Q_in, A_in,n_word_dict)
            batch_Q_word_list, batch_Q_mask_list,batch_A_word_list,batch_A_mask_list, \
            batch_QM_word_list,batch_QM_mask_list,batch_AM_word_list,batch_AM_mask_list = prepare_files(Q_in, A_in,Q_M_in, A_M_in, n_word_dict)
            # batch_Q_word_list   Batch * T
            # batch_QM_word_list   Batch *M *T


            #values= _print_value(batch_QM_word_list,batch_AM_word_list,batch_QM_mask_list,batch_AM_mask_list)
            #print ok
            temp_weight=weight9(num_updates)
            num_updates+=1

            #cost4,cost3,KL_cost0,KL_cost_t,cost3_bow,Z_squre_loss

            [MSE_cost, py_loss, KL_loss0,KL_losst,BOWs_loss]= _train(temp_weight,batch_Q_word_list,batch_A_word_list,batch_Q_mask_list,batch_A_mask_list,\
                              batch_QM_word_list,batch_AM_word_list,batch_QM_mask_list,batch_AM_mask_list)  #_train3_time = theano.function([word_K_list_test,As_word_list,As_mask], [cost3])

            #print "epoch time3: %.4f seconds" % (end3 - begin3)

            if (kk+1)%10==0:
                loss_curve.append(MSE_cost)
                p_y_loss_curve.append(py_loss)
                KL_loss_curve.append(KL_loss0+KL_losst)
                KL_loss_curve0.append(KL_loss0)
                KL_loss_curvet.append(KL_losst)
                BOWs_loss_curve.append(BOWs_loss)
                #Z_Squre_loss_curve.append(Z_Squre)
                end3 = time()
                print "time: %.4f seconds" % (end3 - begin3)
                print 'epoch: %.0f  batch: %.0f/%.0f groups: %d  cost:  %.2f ,  pycost: %.2f, KLcost: %.2f, KLcost0: %.2f, KLcostt: %.2f, BOWscost: %.2f,  %.4f h cost , %.4f h letf'%(epoch,kk,batches,temp_g,float(MSE_cost),float(py_loss),float(KL_loss0+KL_losst),float(KL_loss0),float(KL_losst),float(BOWs_loss),(end3 - begin1)/3600,(batches-kk)*((end3 - begin1)/3600)/kk)
                begin3 = time()

    #########plot
    joblib.dump([loss_curve], 'models/%s/%d_loss_curve.jl' % (desc, epoch))
    joblib.dump([p_y_loss_curve], 'models/%s/%d_p_y_loss_curve.jl' % (desc, epoch))
    joblib.dump([KL_loss_curve], 'models/%s/%d_KL_loss_curve.jl' % (desc, epoch))
    joblib.dump([KL_loss_curve0], 'models/%s/%d_KL_loss_curve0.jl' % (desc, epoch))
    joblib.dump([KL_loss_curvet], 'models/%s/%d_KL_loss_curvet.jl' % (desc, epoch))
    joblib.dump([BOWs_loss_curve], 'models/%s/%d_BOWs_loss_curve.jl' % (desc, epoch))
    joblib.dump([Z_Squre_loss_curve], 'models/%s/%d_Z_Squre_loss_curve.jl' % (desc, epoch))

    loss_curve0=np.asarray(loss_curve)
    loss_curve1=np.asarray(p_y_loss_curve)
    loss_curve2=np.asarray(KL_loss_curve)
    #loss_curve=np.reshape(loss_curve,(len(loss_curve)/3,3))
    l=loss_curve0.shape[0]
    x = np.arange(l)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(x, loss_curve0, 'k-', x, loss_curve1, 'r-', x, loss_curve2, 'b-')
    plt.subplot(212)
    plt.plot(x, loss_curve0, 'k-', x, loss_curve1, 'r-', x, loss_curve2*500, 'b-')
    plt.savefig('samples/%s/loss_curve_%s_%d.jpg'%(desc,desc,epoch))


    #########save
    #if (epoch+1)%5==0:
    joblib.dump([p.get_value() for p in total_params], 'models/%s/%d_total_params.jl' % (desc, epoch))

    end = time()
    print "epoch time: %d seconds" % (end - begin)
    print "--------"

joblib.dump([p.get_value() for p in total_params], 'models/%s/total_params.jl' % (desc))
