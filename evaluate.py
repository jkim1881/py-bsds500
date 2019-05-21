import os, argparse, sys

import tqdm
import numpy as np
from bsds.bsds_dataset import BSDSDataset
from bsds import evaluate_boundaries
from skimage.util import img_as_float
from skimage.color import rgb2grey
from skimage.io import imread

# parser = argparse.ArgumentParser(description='Test output')
# parser.add_argument('bsds_path', type=str,
#                     help='the root path of the BSDS-500 dataset')
# parser.add_argument('pred_path', type=str,
#                     help='the root path of the predictions')
# parser.add_argument('val_test', type=str,
#                     help='val or test')
# parser.add_argument('thresholds', type=str, default='5',
#                     help='the number of thresholds')
# parser.add_argument('suffix_ext', type=str,
#                     help='suffix and extension')
#
# args = parser.parse_args()

# bsds_path = args.bsds_path
# pred_path = args.pred_path
# val_test = args.val_test
# suffix_ext = args.suffix_ext
# thresholds = args.thresholds
# thresholds = thresholds.strip()

zero_as_edges = True
do_nms = False
do_thinning = True
# bsds_or_multicue = 'multicue-boundaries' #'multicue-boundaries' #'bsds'
# bsds_path = '/media/data_cifs/pytorch_projects/datasets'# '/media/data_cifs/pytorch_projects/datasets'
# pred_path = '/media/data_cifs/pytorch_projects/MB_model_out_001' #'/media/data_cifs/pytorch_projects/MB_model_out_001' # '/media/data_cifs/pytorch_projects/model_out_001'
# val_test = 'test'
# suffix_ext = '.jpg'
# thresholds = 100

bsds_or_multicue = 'bsds' #'multicue-boundaries' #'bsds'
bsds_path = '/media/data_cifs/pytorch_projects/datasets'
pred_path = '/media/data_cifs/cluster_projects/refactor_gammanet/bsds_for_jk/1'
val_test = 'test'
suffix_ext = '.tiff'
thresholds = 100


try:
    n_thresholds = int(thresholds)
    thresholds = n_thresholds
except ValueError:
    try:
        if thresholds.startswith('[') and thresholds.endswith(']'):
            thresholds = thresholds[1:-1]
            thresholds = np.array([float(t.strip()) for t in thresholds.split(',')])
        else:
            print('Bad threshold format; should be a python list of floats (`[a, b, c]`)')
            sys.exit()
    except ValueError:
        print('Bad threshold format; should be a python list of ints (`[a, b, c]`)')
        sys.exit()


ds = BSDSDataset(bsds_path, bsds_or_multicue)
if val_test == 'val':
    SAMPLE_NAMES = ds.val_sample_names
elif val_test == 'test':
    SAMPLE_NAMES = ds.test_sample_names
else:
    print('need to specify either val or test, not {}'.format(val_test))
    sys.exit()


def load_gt_boundaries(sample_name):
    return ds.boundaries(sample_name)


def load_pred(sample_name):
    sample_path = os.path.join(pred_path, '{}{}'.format(sample_name, suffix_ext))
    pred = rgb2grey(img_as_float(imread(sample_path)))
    bnds = ds.boundaries(sample_name)
    tgt_shape = bnds[0].shape
    pred = pred[:tgt_shape[0], :tgt_shape[1]]
    pred = np.pad(pred, [(0, tgt_shape[0]-pred.shape[0]), (0, tgt_shape[1]-pred.shape[1])], mode='constant')
    return pred


sample_results, threshold_results, overall_result = evaluate_boundaries.pr_evaluation(
    thresholds, SAMPLE_NAMES, load_gt_boundaries, load_pred,
    zero_as_edges=zero_as_edges, progress=tqdm.tqdm, nms=do_nms, thinning=do_thinning
)


print('Per image:')
for sample_index, res in enumerate(sample_results):
    print('{:<10d} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        sample_index + 1, res.threshold, res.recall, res.precision, res.f1))


print('')
print('Per threshold:')
print('Threshold')
str = ''
for _, res in enumerate(threshold_results):
    str += '{:<10.6f}'.format(res.threshold) + ','
print(str)
str = ''
print('Precision')
for _, res in enumerate(threshold_results):
    str += '{:<10.6f}'.format(res.precision) + ','
print(str)
str = ''
print('Recall')
for _, res in enumerate(threshold_results):
    str += '{:<10.6f}'.format(res.recall) + ','
print(str)
str = ''
print('F1')
for _, res in enumerate(threshold_results):
    str += '{:<10.6f}'.format(res.f1) + ','
print(str)

print('Summary:')
print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
    overall_result.threshold, overall_result.recall, overall_result.precision, overall_result.f1,
    overall_result.best_recall, overall_result.best_precision, overall_result.best_f1,
    overall_result.area_pr))
