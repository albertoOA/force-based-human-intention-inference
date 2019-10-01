#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import copy

# variables to save results of the evaluation
all_results = dict() # general dictionary

dtw_d_results = dict() 
dtw_i_results = dict()

dtw_d_results['accuracy'] = list()
dtw_d_results['precision'] = list()
dtw_d_results['f1_score'] = list()
dtw_d_results['recall'] = list()
dtw_d_results['rmse'] = list()
dtw_d_results['total_time'] = list()
dtw_d_results['time_one_sample'] = list()
dtw_d_results['confusion_matrix'] = list()

dtw_i_results['accuracy'] = list()
dtw_i_results['precision'] = list()
dtw_i_results['f1_score'] = list()
dtw_i_results['recall'] = list()
dtw_i_results['rmse'] = list()
dtw_i_results['total_time'] = list()
dtw_i_results['time_one_sample'] = list()
dtw_i_results['confusion_matrix'] = list()


results_d = dict()
results_i = dict()

results_d['fastDTW'] = copy.deepcopy(dtw_d_results)
results_i['fastDTW'] = copy.deepcopy(dtw_i_results)
