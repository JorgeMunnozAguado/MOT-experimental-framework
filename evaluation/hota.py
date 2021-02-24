
from motmetrics import math_util


################################################################
################################################################
# 
# HOTA metric for MOT
# (Higher Order Metric for Evaluating Multi-Object Tracking)
#
# This is the HOTA implementation over motmetrics
# repository.
# 
################################################################
################################################################



def hota_tp(df, num_matches):
    
    del df
    return num_matches

def hota_fn(df):
    pass

def hota_fp(df, num_false_positives):
    
    del df
    return num_false_positives

def AssA(df):
    pass


def hota(df, num_misses, num_switches, num_false_positives, num_objects):
    """Multiple object tracker accuracy."""
    del df  # unused
    return 1. - math_util.quiet_divide(
        num_misses + num_switches + num_false_positives,
        num_objects)