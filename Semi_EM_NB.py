# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:25:30 2022

@author: Tianyang Liu
"""

import numpy as np
from copy import deepcopy
from scipy.sparse import csr_matrix, vstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from scipy.linalg import get_blas_funcs



""" Naive Bayes classifier for multinomial models for semi-supervised learning. Use both labeled and unlabeled data to train NB classifier, update parameters using unlabeled data, and all data to evaluate performance of classifier. Optimize classifier using Expectation-Maximization algorithm. """

class Semi_EM_MultinomialNB(): 
  
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, max_iter=20, tol=0.0000001): 
        self.alpha = alpha 
        self.fit_prior = fit_prior 
        self.class_prior = class_prior 
        self.clf = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior, class_prior=self.class_prior) 
        self.log_lkh = -np.inf # log likelihood 
        self.max_iter = max_iter # max number of EM iterations 
        self.tol = tol # tolerance of log likelihood increment 

    def fit(self, X_l, y_l, X_u): 
        n_ul_docs = X_u.shape[0] # number of unlabeled samples 
        # initialization (n_docs = n_ul_docs) 
        clf = deepcopy(self.clf)# build new copy of classifier 
        clf.fit(X_l, y_l) # use labeled data only to initialize classifier parameters 
        prev_log_lkh = self.log_lkh # record log likelihood of previous EM iteration 
        lp_w_c = clf.feature_log_prob_ # log CP of word given class [n_classes, n_words] 
        b_w_d = (X_u > 0) # words in each document [n_docs, n_words] 
        lp_d_c = get_blas_funcs("gemm", [lp_w_c, b_w_d.T]) # log CP of doc given class [n_classes, n_docs] 
        lp_d_c = lp_d_c(alpha=1.0, a=lp_w_c, b=b_w_d.T) 
        lp_c = np.matrix(clf.class_log_prior_).T # log prob of classes [n_classes, 1] 
        lp_c = np.repeat(lp_c, n_ul_docs, axis=1) # repeat for each doc [n_classes, n_docs] 
        lp_dc = lp_d_c + lp_c # joint prob of doc and class [n_classes, n_docs] 
        p_c_d = clf.predict_proba(X_u) # weight of each class in each doc [n_docs, n_classes] 
        expectation = get_blas_funcs("gemm", [p_c_d, lp_dc]) # expectation of log likelihood over all unlabeled docs 
        expectation = expectation(alpha=1.0, a=p_c_d, b=lp_dc).trace() 
        self.clf = deepcopy(clf) 
        self.log_lkh = expectation 

        # Loop until log likelihood does not improve 
        iter_count = 0 # count EM iteration 
        while (self.log_lkh-prev_log_lkh>=self.tol and iter_count<self.max_iter): 
            # while (iter_count<self.max_iter): 
            iter_count += 1 

            # E-step: Estimate class membership of unlabeled documents 
            y_u = clf.predict(X_u) 
            # M-step: Re-estimate classifier parameters 
            X = vstack([X_l, X_u]) 
            y = np.concatenate((y_l, y_u), axis=0) 
            clf.fit(X, y) # check convergence: update log likelihood 
            p_c_d = clf.predict_proba(X_u) 
            

            lp_w_c = clf.feature_log_prob_ # log CP of word given class [n_classes, n_words] 
            b_w_d = (X_u > 0) # words in each document 
            lp_d_c = get_blas_funcs("gemm", [lp_w_c, b_w_d.transpose()]) # log CP of doc given class [n_classes, n_docs] 
            lp_d_c = lp_d_c(alpha=1.0, a=lp_w_c, b=b_w_d.transpose()) 
            lp_c = np.matrix(clf.class_log_prior_).T # log prob of classes [n_classes, 1] 
            lp_c = np.repeat(lp_c, n_ul_docs, axis=1) # repeat for each doc [n_classes, n_docs] 
            lp_dc = lp_d_c + lp_c # joint prob of doc and class [n_classes, n_docs] 
            expectation = get_blas_funcs("gemm", [p_c_d, lp_dc]) # expectation of log likelihood over all unlabeled docs 
            expectation = expectation(alpha=1.0, a=p_c_d, b=lp_dc).trace() 

            if (expectation-self.log_lkh >= self.tol): 
                prev_log_lkh = self.log_lkh 
                self.log_lkh = expectation 
                self.clf = deepcopy(clf) 
            # else:
            #     for i in range(X_u.shape[0]):
            #         if(y_u[i]>0):
            #             # print(p_c_d[i][1]-p_c_d[i][0]) 
            #     break 
        
        return self 

    def partial_fit(self, X_l, y_l, X_u): 
        pass 

    def predict(self, X): 
        return self.clf.predict(X) 

    def score(self, X, y): 
        pass
