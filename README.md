## weibo-comments
A demo code for the weibo-comments creation. <br>
The modules are implemented based on the DeepLearning 0.1 documentation (http://deeplearning.net/tutorial/).<br>

## Requirements
The code is written in `Theano`. <br>
To use it you will also need: coco-caption envaluate package, `cPickle`,`scikit-learn`, `matplotlib`,`tqdm`,`jieba`,`gensim` and `PIL`.<br>
Before running the code make sure that you have set floatX to float32 in Theano settings.<br>
Dataset is not available for now because of the authority issue.<br>

## Usage
Make a subfolder named lib, and put activations.py, inits.py , rng.py, theano_utils.py and updates.py in it.<br>
Train: weibo_train_v3.py<br>
Envalue:weibo_v3_enval_with_beamsearch.py<br>

## Reference
If you found this code or our paper useful, please consider citing the following paper:<br>

@inproceedings{IJCAI20weibo,<br>
    author    = {Shijie Yang, Liang Li, Shuhui Wang, Weigang Zhang, Qingming Huang, Qi Tian},<br>
    title     = {A Structured Latent Variable Recurrent Network with Stochastic Attention for GeneratingWeibo Comments},<br>
    booktitle = {IJCAI},<br>
    year      = {2020}<br>
}<br>
