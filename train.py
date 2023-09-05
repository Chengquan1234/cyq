import time

import keras.optimizers
import numpy as np
import tensorflow as tf
from nets.frcnn import get_train_model
from utils.config import Config
from utils.utils import BboxUtils,calc_iou
from utils.anchors import get_anchors
from nets.frcnn_training import Generate,class_loss_regr,smooth_l1,cls_loss,class_loss_cls
import keras.backend as K
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
if __name__ == '__main__':
    EPOCH = 20
    EPOCH_LENGTH = 1000
    NUM_CLASS = 2+1
    config = Config()
    base_net_weights = "model_data/voc_weights.h5"
    rpn_model,cls_model,model_all  = get_train_model(config,NUM_CLASS)
    data_path = 'label.txt'

    model_all.summary()
    #rpn_model.load_weights(base_net_weights)
    #cls_model.load_weights(base_net_weights)

    with open(data_path) as f:
        lines = f.readlines()

    np.random.seed(101)
    np.random.shuffle(lines)
    np.random.seed(None)
    anchors = get_anchors(38,38,config)
    bboxUtils = BboxUtils(anchors,lower_threshold=config.min_threshold,higher_threshold=config.max_threshold)
    gen = Generate(bboxUtils,lines,NUM_CLASS,config,solid=True)
    rpn_train = gen.generate()

    log_dir = 'logs'
    callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    callback.set_model(model_all)
    rpn_model.compile(
        loss={
            'rpn_logits':   smooth_l1(),
            'rpn_class':  cls_loss()
        },optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    cls_model.compile(loss=[
        class_loss_cls,
        class_loss_regr(NUM_CLASS-1)
    ],optimizer=keras.optimizers.Adam(lr=1e-5))
    model_all.compile(optimizer='sgd',loss='mae')

    # 初始化参数
    iter_num = 0
    train_step = 0
    losses = np.zeros((EPOCH_LENGTH, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()
    # 最佳loss
    best_loss = np.Inf
    # 数字到类的映射
    print('Starting training')
    for i in range(EPOCH):
        progbar = keras.utils.generic_utils.Progbar(EPOCH_LENGTH)
        print('Epoch:{}/{}'.format(i+1,EPOCH))
        while True:
            X, Y, boxes = next(rpn_train)

            loss_rpn=rpn_model.train_on_batch(X,Y)
            write_log(callback, ['rpn_cls_loss', 'rpn_reg_loss'], loss_rpn, train_step)
            P = rpn_model.predict_on_batch(X)
            width,height,_ = np.shape(X[0])
            anchors = get_anchors(width/config.rpn_stride,height/config.rpn_stride,config)
            result = bboxUtils.dection_out(P,anchors,1,confidence_threshold=0)

            R = result[0][:,2:]

            X2,Y1,Y2,_ = calc_iou(R,boxes[0],config,width,height,NUM_CLASS)
            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue
            num_rois = config.rois

            pos_sample = np.where(Y1[0,:,-1] == 0)
            neg_sample = np.where(Y1[0,:,-1] == 1)

            if len(pos_sample) > 0:
                pos_sample = pos_sample[0]
            else:
                pos_sample = []

            if len(neg_sample) > 0:
                neg_sample = neg_sample[0]
            else:
                neg_sample = []

            if len(pos_sample) < num_rois/2 :
                select_pos = pos_sample.tolist()
            else:
                select_pos = np.random.choice(pos_sample,int(num_rois/2),replace=False).tolist()
            try:
                selet_neg = np.random.choice(neg_sample,int(num_rois - len(select_pos)),replace=False).tolist()
            except:
                selet_neg = np.random.choice(neg_sample,int(num_rois - len(select_pos)),replace=True).tolist()
            select = select_pos + selet_neg
            loss_class =cls_model.train_on_batch([X,X2[:,select,:]],[Y1[:,select,:],Y2[:,select,:]])
            iter_num += 1

            write_log(callback, ['detection_cls_loss', 'detection_reg_loss', 'detection_acc'], loss_class, train_step)

            losses[iter_num, 0] = loss_rpn[0]
            losses[iter_num, 1] = loss_rpn[1]
            losses[iter_num, 2] = loss_class[0]
            losses[iter_num, 3] = loss_class[1]
            losses[iter_num, 4] = loss_class[2]

            train_step += 1
            iter_num += 1
            progbar.update(iter_num,
                           [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('detector_cls', np.mean(losses[:iter_num, 2])),
                            ('detector_regr', np.mean(losses[:iter_num, 3]))])
            if iter_num == EPOCH_LENGTH:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / (len(rpn_accuracy_for_epoch)+1e-6)
                rpn_accuracy_for_epoch = []

                if config.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                write_log(callback,
                          ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
                           'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
                          [time.time() - start_time, mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr,
                           loss_class_cls, loss_class_regr, class_acc, curr_loss], i)

                if config.verbose:
                    print('The best loss is {}. The current loss is {}. Saving weights'.format(best_loss, curr_loss))
                if curr_loss < best_loss:
                    best_loss = curr_loss
                model_all.save_weights(log_dir + "/epoch{:03d}-loss{:.3f}-rpn{:.3f}-roi{:.3f}".format(i, curr_loss,
                                                                                                      loss_rpn_cls + loss_rpn_regr,
                                                                                                      loss_class_cls + loss_class_regr) + ".h5")

                break


