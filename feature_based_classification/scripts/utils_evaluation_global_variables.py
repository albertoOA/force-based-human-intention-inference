#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# variables to save results of the evaluation
all_results = dict() # general dictionary for gplvm

gplvm_op_results = dict() # results for the case in which the inference is optimized
gplvm_nop_results = dict() # results for the case in which the inference is not optimized
bgplvm_op_results = dict() # results for the case in which the inference is optimized
bgplvm_nop_results = dict() # results for the case in which the inference is not optimized
pca_results = dict() # general dictionary for pca results

gplvm_op_results['accuracy'] = list()
gplvm_op_results['f1_score'] = list()
gplvm_op_results['recall'] = list()
gplvm_op_results['rmse'] = list()
gplvm_op_results['total_time'] = list()
gplvm_op_results['time_one_sample'] = list()
gplvm_op_results['confusion_matrix'] = list()
gplvm_op_results['precision'] = list()

gplvm_nop_results['accuracy'] = list()
gplvm_nop_results['f1_score'] = list()
gplvm_nop_results['recall'] = list()
gplvm_nop_results['rmse'] = list()
gplvm_nop_results['total_time'] = list()
gplvm_nop_results['time_one_sample'] = list()
gplvm_nop_results['confusion_matrix'] = list()
gplvm_nop_results['precision'] = list()

bgplvm_op_results['accuracy'] = list()
bgplvm_op_results['f1_score'] = list()
bgplvm_op_results['recall'] = list()
bgplvm_op_results['rmse'] = list()
bgplvm_op_results['total_time'] = list()
bgplvm_op_results['time_one_sample'] = list()
bgplvm_op_results['confusion_matrix'] = list()
bgplvm_op_results['precision'] = list()

bgplvm_nop_results['accuracy'] = list()
bgplvm_nop_results['f1_score'] = list()
bgplvm_nop_results['recall'] = list()
bgplvm_nop_results['rmse'] = list()
bgplvm_nop_results['total_time'] = list()
bgplvm_nop_results['time_one_sample'] = list()
bgplvm_nop_results['confusion_matrix'] = list()
bgplvm_nop_results['precision'] = list()

pca_results['accuracy'] = list()
pca_results['f1_score'] = list()
pca_results['recall'] = list()
pca_results['rmse'] = list()
pca_results['total_time'] = list()
pca_results['time_one_sample'] = list()
pca_results['confusion_matrix'] = list()
pca_results['precision'] = list()

misclassified_samples_gplvm_op = dict()
misclassified_samples_gplvm_nop = dict()
misclassified_samples_bgplvm_op = dict()
misclassified_samples_bgplvm_nop = dict()
misclassified_samples_pca = dict()

misclassified_samples_gplvm_op_percentage = dict()
misclassified_samples_gplvm_nop_percentage = dict()
misclassified_samples_bgplvm_op_percentage = dict()
misclassified_samples_bgplvm_nop_percentage = dict()
misclassified_samples_pca_percentage = dict()

test_samples_count_dict_gplvm_op = dict()
test_samples_count_dict_gplvm_nop = dict()
test_samples_count_dict_bgplvm_op = dict()
test_samples_count_dict_bgplvm_nop = dict()
test_samples_count_dict_pca = dict()
