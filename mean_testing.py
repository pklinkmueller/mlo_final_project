# External libraries:
import numpy as np
from sklearn.model_selection import train_test_split

# Internal libraries:
import datasets.data as data
from descent_algorithms import *
from learning_rates import *
from models import *
from util import *

def logreg_wbc_fixed_runs(gd_loss, sgd_1_loss, sgd_10_loss, sgd_100_loss, agd_loss,
  svrg_loss, md_loss, wbc_n, wbc_X_train, wbc_y_train):
  rel_conv = 0.000001
  lr = FixedRate(0.01)
  gd_loss_counts = np.sign(gd_loss)
  sgd_1_loss_counts = np.sign(sgd_1_loss)
  sgd_10_loss_counts = np.sign(sgd_10_loss)
  sgd_100_loss_counts = np.sign(sgd_100_loss)
  agd_loss_counts = np.sign(agd_loss)
  svrg_loss_counts = np.sign(svrg_loss)
  md_loss_counts = np.sign(md_loss)
  gd_time = 0
  sgd_1_time = 0
  sgd_10_time = 0
  sgd_100_time = 0
  agd_time = 0
  svrg_time = 0
  md_time = 0

  for i in range(0,99):
      gd = GradientDescent()
      sgd_1 = GradientDescent()
      sgd_10 = GradientDescent()
      sgd_100 = GradientDescent()
      agd = NesterovAcceleratedDescent()
      svrg = StochasticVarianceReducedGradientDescent()
      md = MirrorDescent()
      gd_log = LogisticRegression(gd, lr, 5000, wbc_n, rel_conv)
      sgd_1_log = LogisticRegression(sgd_1, lr, 2000, 1, rel_conv)
      sgd_10_log = LogisticRegression(sgd_10, lr, 4000, 10, rel_conv)
      sgd_100_log = LogisticRegression(sgd_100, lr, 4000, 100, rel_conv)
      agd_log = LogisticRegression(agd, lr, 400, wbc_n, rel_conv)
      svrg_log = LogisticRegression(svrg, lr, 20, wbc_n, rel_conv)
      md_log = LogisticRegression(md, lr, 2000, wbc_n, rel_conv)

      tmp_loss, tmp_time = gd_log.fit(wbc_X_train, wbc_y_train)
      gd_loss_counts += np.sign(tmp_loss)
      gd_loss += tmp_loss
      gd_time += tmp_time
      tmp_loss, tmp_time = sgd_1_log.fit(wbc_X_train, wbc_y_train)
      sgd_1_loss_counts += np.sign(tmp_loss)
      sgd_1_loss += tmp_loss
      sgd_1_time += tmp_time
      tmp_loss, tmp_time = sgd_10_log.fit(wbc_X_train, wbc_y_train)
      sgd_10_loss_counts += np.sign(tmp_loss)
      sgd_10_loss += tmp_loss
      sgd_10_time += tmp_time
      tmp_loss, tmp_time = sgd_100_log.fit(wbc_X_train, wbc_y_train)
      sgd_100_loss_counts += np.sign(tmp_loss)
      sgd_100_loss += tmp_loss
      sgd_100_time += tmp_time
      tmp_loss, tmp_time = agd_log.fit(wbc_X_train, wbc_y_train)
      agd_loss_counts += np.sign(tmp_loss)
      agd_loss += tmp_loss
      agd_time += tmp_time
      tmp_loss, tmp_time = svrg_log.fit(wbc_X_train, wbc_y_train)
      svrg_loss_counts += np.sign(tmp_loss)
      svrg_loss += tmp_loss
      svrg_time += tmp_time
      tmp_loss, tmp_time = md_log.fit(wbc_X_train, wbc_y_train)
      md_loss_counts += np.sign(tmp_loss)
      md_loss += tmp_loss
      md_time += tmp_time
  gd_loss /= gd_loss_counts
  sgd_1_loss /= sgd_1_loss_counts
  sgd_10_loss /= sgd_10_loss_counts
  sgd_100_loss /= sgd_100_loss_counts
  agd_loss /= agd_loss_counts
  svrg_loss /= svrg_loss_counts
  md_loss /= md_loss_counts
  gd_time /= 99
  sgd_1_time /= 99
  sgd_10_time /= 99
  sgd_100_time /= 99
  agd_time /= 99
  svrg_time /= 99
  md_time /= 99
  gd_loss = gd_loss[gd_loss.nonzero()]
  sgd_1_loss = sgd_1_loss[sgd_1_loss.nonzero()]
  sgd_10_loss = sgd_10_loss[sgd_10_loss.nonzero()]
  sgd_100_loss = sgd_100_loss[sgd_100_loss.nonzero()]
  agd_loss = agd_loss[agd_loss.nonzero()]
  svrg_loss = svrg_loss[svrg_loss.nonzero()]
  md_loss = md_loss[md_loss.nonzero()]

  print('GD average runtime: {}'.format(gd_time))
  print('SGD_1 average runtime: {}'.format(sgd_1_time))
  print('SGD_10 average runtime: {}'.format(sgd_10_time))
  print('SGD_100 average runtime: {}'.format(sgd_100_time))
  print('AGD average runtime: {}'.format(agd_time))
  print('SVRG average runtime: {}'.format(svrg_time))
  print('MD average runtime: {}'.format(md_time))

  return (gd_loss, sgd_1_loss, sgd_10_loss, sgd_100_loss, agd_loss, svrg_loss, md_loss)



def logreg_wbc_exp_runs(gd_loss, sgd_1_loss, md_loss, wbc_n, wbc_X_train, wbc_y_train):
  rel_conv = 0.000001
  gd_loss_counts = np.sign(gd_loss)
  sgd_1_loss_counts = np.sign(sgd_1_loss)
  md_loss_counts = np.sign(md_loss)
  gd_time = 0
  sgd_1_time = 0
  md_time = 0

  for i in range(0,99):
    lr_gd = ExpDecayRate(0.1, 0.0001)
    lr_sgd = ExpDecayRate(0.01, 0.00001)
    lr_md = ExpDecayRate(0.1, 0.00001)
    gd = GradientDescent()
    sgd_1 = GradientDescent()
    md = MirrorDescent()
    gd_log = LogisticRegression(gd, lr_gd, 2000, wbc_n, rel_conv)
    sgd_1_log = LogisticRegression(sgd_1, lr_sgd, 2000, 1, rel_conv)
    md_log = LogisticRegression(md, lr_md, 2000, wbc_n, rel_conv)

    tmp_loss, tmp_time = gd_log.fit(wbc_X_train, wbc_y_train)
    gd_loss_counts += np.sign(tmp_loss)
    gd_loss += tmp_loss
    gd_time += tmp_time
    tmp_loss, tmp_time = sgd_1_log.fit(wbc_X_train, wbc_y_train)
    sgd_1_loss_counts += np.sign(tmp_loss)
    sgd_1_loss += tmp_loss
    sgd_1_time += tmp_time
    tmp_loss, tmp_time = md_log.fit(wbc_X_train, wbc_y_train)
    md_loss_counts += np.sign(tmp_loss)
    md_loss += tmp_loss
    md_time += tmp_time

  gd_loss /= gd_loss_counts
  sgd_1_loss /= sgd_1_loss_counts
  md_loss /= md_loss_counts
  gd_time /= 99
  sgd_1_time /= 99
  md_time /= 99
  gd_loss = gd_loss[gd_loss.nonzero()]
  sgd_1_loss = sgd_1_loss[sgd_1_loss.nonzero()]
  md_loss = md_loss[md_loss.nonzero()]

  print('GD average runtime: {}'.format(gd_time))
  print('SGD_1 average runtime: {}'.format(sgd_1_time))
  print('MD average runtime: {}'.format(md_time))

  return (gd_loss, sgd_1_loss, md_loss)


def logreg_wbc_sqrt_runs(gd_loss, sgd_1_loss, md_loss, wbc_n, wbc_X_train, wbc_y_train):
  rel_conv = 0.000001
  gd_loss_counts = np.sign(gd_loss)
  sgd_1_loss_counts = np.sign(sgd_1_loss)
  md_loss_counts = np.sign(md_loss)
  gd_time = 0
  sgd_1_time = 0
  md_time = 0

  for i in range(0,99):
    lr_gd = SqrtDecayRate(0.001,1)
    lr_sgd = SqrtDecayRate(0.0001,1)
    lr_md = SqrtDecayRate(0.001,1)
    gd = GradientDescent()
    sgd_1 = GradientDescent()
    md = MirrorDescent()
    gd_log = LogisticRegression(gd, lr_gd, 4000, wbc_n, rel_conv)
    sgd_1_log = LogisticRegression(sgd_1, lr_sgd, 4000, 1, rel_conv)
    md_log = LogisticRegression(md, lr_md, 2000, wbc_n, rel_conv)

    tmp_loss, tmp_time = gd_log.fit(wbc_X_train, wbc_y_train)
    gd_loss_counts += np.sign(tmp_loss)
    gd_loss += tmp_loss
    gd_time += tmp_time
    tmp_loss, tmp_time = sgd_1_log.fit(wbc_X_train, wbc_y_train)
    sgd_1_loss_counts += np.sign(tmp_loss)
    sgd_1_loss += tmp_loss
    sgd_1_time += tmp_time
    tmp_loss, tmp_time = md_log.fit(wbc_X_train, wbc_y_train)
    md_loss_counts += np.sign(tmp_loss)
    md_loss += tmp_loss
    md_time += tmp_time

  gd_loss /= gd_loss_counts
  sgd_1_loss /= sgd_1_loss_counts
  md_loss /= md_loss_counts
  gd_time /= 99
  sgd_1_time /= 99
  md_time /= 99
  gd_loss = gd_loss[gd_loss.nonzero()]
  sgd_1_loss = sgd_1_loss[sgd_1_loss.nonzero()]
  md_loss = md_loss[md_loss.nonzero()]

  print('GD average runtime: {}'.format(gd_time))
  print('SGD_1 average runtime: {}'.format(sgd_1_time))
  print('MD average runtime: {}'.format(md_time))

  return (gd_loss, sgd_1_loss, md_loss)


def logreg_cod_fixed_runs(gd_loss, sgd_1_loss, sgd_10_loss, sgd_100_loss, agd_loss,
  svrg_loss, md_loss, wbc_n, wbc_X_train, wbc_y_train):
  rel_conv = 0.000001
  lr = FixedRate(0.00001)
  gd_loss_counts = np.sign(gd_loss)
  sgd_1_loss_counts = np.sign(sgd_1_loss)
  sgd_10_loss_counts = np.sign(sgd_10_loss)
  sgd_100_loss_counts = np.sign(sgd_100_loss)
  agd_loss_counts = np.sign(agd_loss)
  svrg_loss_counts = np.sign(svrg_loss)
  md_loss_counts = np.sign(md_loss)
  gd_time = 0
  sgd_1_time = 0
  sgd_10_time = 0
  sgd_100_time = 0
  agd_time = 0
  svrg_time = 0
  md_time = 0

  for i in range(0,1):
      gd = GradientDescent()
      sgd_1 = GradientDescent()
      sgd_10 = GradientDescent()
      sgd_100 = GradientDescent()
      agd = NesterovAcceleratedDescent()
      svrg = StochasticVarianceReducedGradientDescent()
      md = MirrorDescent()
      gd_log = LogisticRegression(gd, lr, 9000, wbc_n, rel_conv)
      sgd_1_log = LogisticRegression(sgd_1, lr, 8000, 1, rel_conv)
      sgd_10_log = LogisticRegression(sgd_10, lr, 8000, 10, rel_conv)
      sgd_100_log = LogisticRegression(sgd_100, lr, 8000, 100, rel_conv)
      agd_log = LogisticRegression(agd, lr, 400, wbc_n, rel_conv)
      svrg_log = LogisticRegression(svrg, lr, 20, wbc_n, rel_conv)
      md_log = LogisticRegression(md, lr, 6000, wbc_n, rel_conv)

      tmp_loss, tmp_time = gd_log.fit(wbc_X_train, wbc_y_train)
      gd_loss_counts += np.sign(tmp_loss)
      gd_loss += tmp_loss
      gd_time += tmp_time
      tmp_loss, tmp_time = sgd_1_log.fit(wbc_X_train, wbc_y_train)
      sgd_1_loss_counts += np.sign(tmp_loss)
      sgd_1_loss += tmp_loss
      sgd_1_time += tmp_time
      tmp_loss, tmp_time = sgd_10_log.fit(wbc_X_train, wbc_y_train)
      sgd_10_loss_counts += np.sign(tmp_loss)
      sgd_10_loss += tmp_loss
      sgd_10_time += tmp_time
      tmp_loss, tmp_time = sgd_100_log.fit(wbc_X_train, wbc_y_train)
      sgd_100_loss_counts += np.sign(tmp_loss)
      sgd_100_loss += tmp_loss
      sgd_100_time += tmp_time
      tmp_loss, tmp_time = agd_log.fit(wbc_X_train, wbc_y_train)
      agd_loss_counts += np.sign(tmp_loss)
      agd_loss += tmp_loss
      agd_time += tmp_time
      tmp_loss, tmp_time = svrg_log.fit(wbc_X_train, wbc_y_train)
      svrg_loss_counts += np.sign(tmp_loss)
      svrg_loss += tmp_loss
      svrg_time += tmp_time
      tmp_loss, tmp_time = md_log.fit(wbc_X_train, wbc_y_train)
      md_loss_counts += np.sign(tmp_loss)
      md_loss += tmp_loss
      md_time += tmp_time
  gd_loss /= gd_loss_counts
  sgd_1_loss /= sgd_1_loss_counts
  sgd_10_loss /= sgd_10_loss_counts
  sgd_100_loss /= sgd_100_loss_counts
  agd_loss /= agd_loss_counts
  svrg_loss /= svrg_loss_counts
  md_loss /= md_loss_counts
  # gd_time /= 1
  # sgd_1_time /= 4
  # sgd_10_time /= 4
  # sgd_100_time /= 4
  # agd_time /= 4
  # svrg_time /= 4
  # md_time /= 4
  gd_loss = gd_loss[gd_loss.nonzero()]
  sgd_1_loss = sgd_1_loss[sgd_1_loss.nonzero()]
  sgd_10_loss = sgd_10_loss[sgd_10_loss.nonzero()]
  sgd_100_loss = sgd_100_loss[sgd_100_loss.nonzero()]
  agd_loss = agd_loss[agd_loss.nonzero()]
  svrg_loss = svrg_loss[svrg_loss.nonzero()]
  md_loss = md_loss[md_loss.nonzero()]

  print('GD average runtime: {}'.format(gd_time))
  print('SGD_1 average runtime: {}'.format(sgd_1_time))
  print('SGD_10 average runtime: {}'.format(sgd_10_time))
  print('SGD_100 average runtime: {}'.format(sgd_100_time))
  print('AGD average runtime: {}'.format(agd_time))
  print('SVRG average runtime: {}'.format(svrg_time))
  print('MD average runtime: {}'.format(md_time))

  return (gd_loss, sgd_1_loss, sgd_10_loss, sgd_100_loss, agd_loss, svrg_loss, md_loss)



def logreg_cod_exp_runs(gd_loss, sgd_1_loss, md_loss, wbc_n, wbc_X_train, wbc_y_train):
  rel_conv = 0.000001
  gd_loss_counts = np.sign(gd_loss)
  sgd_1_loss_counts = np.sign(sgd_1_loss)
  md_loss_counts = np.sign(md_loss)
  gd_time = 0
  sgd_1_time = 0
  md_time = 0

  for i in range(0,2):
    lr_gd = ExpDecayRate(0.0001, 0.00001)
    lr_sgd = ExpDecayRate(0.00001, 0.000001)
    lr_md = ExpDecayRate(0.00001, 0.00001)
    gd = GradientDescent()
    sgd_1 = GradientDescent()
    md = MirrorDescent()
    gd_log = LogisticRegression(gd, lr_gd, 4000, wbc_n, rel_conv)
    sgd_1_log = LogisticRegression(sgd_1, lr_sgd, 6000, 1, rel_conv)
    md_log = LogisticRegression(md, lr_md, 4000, wbc_n, rel_conv)

    tmp_loss, tmp_time = gd_log.fit(wbc_X_train, wbc_y_train)
    gd_loss_counts += np.sign(tmp_loss)
    gd_loss += tmp_loss
    gd_time += tmp_time
    tmp_loss, tmp_time = sgd_1_log.fit(wbc_X_train, wbc_y_train)
    sgd_1_loss_counts += np.sign(tmp_loss)
    sgd_1_loss += tmp_loss
    sgd_1_time += tmp_time
    tmp_loss, tmp_time = md_log.fit(wbc_X_train, wbc_y_train)
    md_loss_counts += np.sign(tmp_loss)
    md_loss += tmp_loss
    md_time += tmp_time

  gd_loss /= gd_loss_counts
  sgd_1_loss /= sgd_1_loss_counts
  md_loss /= md_loss_counts
  gd_time /= 2
  sgd_1_time /= 2
  md_time /= 2
  gd_loss = gd_loss[gd_loss.nonzero()]
  sgd_1_loss = sgd_1_loss[sgd_1_loss.nonzero()]
  md_loss = md_loss[md_loss.nonzero()]

  print('GD average runtime: {}'.format(gd_time))
  print('SGD_1 average runtime: {}'.format(sgd_1_time))
  print('MD average runtime: {}'.format(md_time))

  return (gd_loss, sgd_1_loss, md_loss)



def logreg_cod_sqrt_runs(gd_loss, sgd_1_loss, md_loss, wbc_n, wbc_X_train, wbc_y_train):
  rel_conv = 0.000001
  gd_loss_counts = np.sign(gd_loss)
  sgd_1_loss_counts = np.sign(sgd_1_loss)
  md_loss_counts = np.sign(md_loss)
  gd_time = 0
  sgd_1_time = 0
  md_time = 0

  for i in range(0,1):
    lr_gd = SqrtDecayRate(0.000001, 1.)
    lr_sgd = SqrtDecayRate(0.0000001, 2.)
    lr_md = SqrtDecayRate(0.000001, 10.)
    gd = GradientDescent()
    sgd_1 = GradientDescent()
    md = MirrorDescent()
    gd_log = LogisticRegression(gd, lr_gd, 4000, wbc_n, rel_conv)
    sgd_1_log = LogisticRegression(sgd_1, lr_sgd, 6000, 1, rel_conv)
    md_log = LogisticRegression(md, lr_md, 4000, wbc_n, rel_conv)

    tmp_loss, tmp_time = gd_log.fit(wbc_X_train, wbc_y_train)
    gd_loss_counts += np.sign(tmp_loss)
    gd_loss += tmp_loss
    gd_time += tmp_time
    tmp_loss, tmp_time = sgd_1_log.fit(wbc_X_train, wbc_y_train)
    sgd_1_loss_counts += np.sign(tmp_loss)
    sgd_1_loss += tmp_loss
    sgd_1_time += tmp_time
    tmp_loss, tmp_time = md_log.fit(wbc_X_train, wbc_y_train)
    md_loss_counts += np.sign(tmp_loss)
    md_loss += tmp_loss
    md_time += tmp_time

  gd_loss /= gd_loss_counts
  sgd_1_loss /= sgd_1_loss_counts
  md_loss /= md_loss_counts
  # gd_time /= 1
  # sgd_1_time /= 4
  # md_time /= 4
  gd_loss = gd_loss[gd_loss.nonzero()]
  sgd_1_loss = sgd_1_loss[sgd_1_loss.nonzero()]
  md_loss = md_loss[md_loss.nonzero()]

  print('GD average runtime: {}'.format(gd_time))
  print('SGD_1 average runtime: {}'.format(sgd_1_time))
  print('MD average runtime: {}'.format(md_time))

  return (gd_loss, sgd_1_loss, md_loss)




########## SVD : #################################################






def svm_wbc_fixed_runs(gd_loss, sgd_100_loss, agd_loss, svrg_loss, md_loss, wbc_n, wbc_X_train, wbc_y_train):
  rel_conv = 0.000001
  c = 0.00001
  lr = FixedRate(0.0005)
  gd_loss_counts = np.sign(gd_loss)
  sgd_100_loss_counts = np.sign(sgd_100_loss)
  agd_loss_counts = np.sign(agd_loss)
  svrg_loss_counts = np.sign(svrg_loss)
  md_loss_counts = np.sign(md_loss)
  gd_time = 0
  sgd_100_time = 0
  agd_time = 0
  svrg_time = 0
  md_time = 0

  for i in range(0,4):
    gd = GradientDescent()
    sgd_100 = GradientDescent()
    agd = NesterovAcceleratedDescent()
    svrg = StochasticVarianceReducedGradientDescent()
    md = MirrorDescent()
    gd_svm = SVM(gd, lr, c, 20000, wbc_n, rel_conv)
    sgd_100_svm = SVM(sgd_100, lr, c, 20000, 100, rel_conv)
    agd_svm = SVM(agd, lr, c, 20000, wbc_n, rel_conv)
    svrg_svm = SVM(svrg, lr, c, 3000, wbc_n, rel_conv)
    md_svm = SVM(md, lr, c, 2000, wbc_n, rel_conv)

    tmp_loss, tmp_time = gd_svm.fit(wbc_X_train, wbc_y_train)
    gd_loss_counts += np.sign(tmp_loss)
    gd_loss += tmp_loss
    gd_time += tmp_time
    tmp_loss, tmp_time = sgd_100_svm.fit(wbc_X_train, wbc_y_train)
    sgd_100_loss_counts += np.sign(tmp_loss)
    sgd_100_loss += tmp_loss
    sgd_100_time += tmp_time
    tmp_loss, tmp_time = agd_svm.fit(wbc_X_train, wbc_y_train)
    agd_loss_counts += np.sign(tmp_loss)
    agd_loss += tmp_loss
    agd_time += tmp_time
    tmp_loss, tmp_time = svrg_svm.fit(wbc_X_train, wbc_y_train)
    svrg_loss_counts += np.sign(tmp_loss)
    svrg_loss += tmp_loss
    svrg_time += tmp_time
    tmp_loss, tmp_time = md_svm.fit(wbc_X_train, wbc_y_train)
    md_loss_counts += np.sign(tmp_loss)
    md_loss += tmp_loss
    md_time += tmp_time
  gd_loss /= gd_loss_counts
  sgd_100_loss /= sgd_100_loss_counts
  agd_loss /= agd_loss_counts
  svrg_loss /= svrg_loss_counts
  md_loss /= md_loss_counts
  gd_time /= 4
  sgd_100_time /= 4
  agd_time /= 4
  svrg_time /= 4
  md_time /= 4
  gd_loss = gd_loss[gd_loss.nonzero()]
  sgd_100_loss = sgd_100_loss[sgd_100_loss.nonzero()]
  agd_loss = agd_loss[agd_loss.nonzero()]
  svrg_loss = svrg_loss[svrg_loss.nonzero()]
  md_loss = md_loss[md_loss.nonzero()]

  print('GD average runtime: {}'.format(gd_time))
  print('SGD_100 average runtime: {}'.format(sgd_100_time))
  print('AGD average runtime: {}'.format(agd_time))
  print('SVRG average runtime: {}'.format(svrg_time))
  print('MD average runtime: {}'.format(md_time))

  return (gd_loss, sgd_100_loss, agd_loss, svrg_loss, md_loss)



def svm_wbc_exp_runs(gd_loss, sgd_100_loss, md_loss, wbc_n, wbc_X_train, wbc_y_train):
  rel_conv = 0.000001
  c = 0.00001
  gd_loss_counts = np.sign(gd_loss)
  sgd_100_loss_counts = np.sign(sgd_100_loss)
  md_loss_counts = np.sign(md_loss)
  gd_time = 0
  sgd_100_time = 0
  md_time = 0

  for i in range(0,19):
    lr_gd = ExpDecayRate(0.001, 0.0001)
    lr_sgd = ExpDecayRate(0.001, 0.00001)
    lr_md = ExpDecayRate(0.0001, 0.00001)
    gd = GradientDescent()
    sgd_100 = GradientDescent()
    md = MirrorDescent()
    gd_svm = SVM(gd, lr_gd, c, 4000, wbc_n, rel_conv)
    sgd_100_svm = SVM(sgd_100, lr_sgd, c, 6000, 100, rel_conv)
    md_svm = SVM(md, lr_md, c, 4000, wbc_n, rel_conv)

    tmp_loss, tmp_time = gd_svm.fit(wbc_X_train, wbc_y_train)
    gd_loss_counts += np.sign(tmp_loss)
    gd_loss += tmp_loss
    gd_time += tmp_time
    tmp_loss, tmp_time = sgd_100_svm.fit(wbc_X_train, wbc_y_train)
    sgd_100_loss_counts += np.sign(tmp_loss)
    sgd_100_loss += tmp_loss
    sgd_100_time += tmp_time
    tmp_loss, tmp_time = md_svm.fit(wbc_X_train, wbc_y_train)
    md_loss_counts += np.sign(tmp_loss)
    md_loss += tmp_loss
    md_time += tmp_time

  gd_loss /= gd_loss_counts
  sgd_100_loss /= sgd_100_loss_counts
  md_loss /= md_loss_counts
  gd_time /= 19
  sgd_100_time /= 19
  md_time /= 19
  gd_loss = gd_loss[gd_loss.nonzero()]
  sgd_100_loss = sgd_100_loss[sgd_100_loss.nonzero()]
  md_loss = md_loss[md_loss.nonzero()]

  print('GD average runtime: {}'.format(gd_time))
  print('SGD_100 average runtime: {}'.format(sgd_100_time))
  print('MD average runtime: {}'.format(md_time))

  return (gd_loss, sgd_100_loss, md_loss)


def svm_wbc_sqrt_runs(gd_loss, sgd_100_loss, md_loss, wbc_n, wbc_X_train, wbc_y_train):
  rel_conv = 0.000001
  c = 0.00001
  gd_loss_counts = np.sign(gd_loss)
  sgd_100_loss_counts = np.sign(sgd_100_loss)
  md_loss_counts = np.sign(md_loss)
  gd_time = 0
  sgd_100_time = 0
  md_time = 0

  for i in range(0,19):
    lr_gd = SqrtDecayRate(0.00001, 1.)
    lr_sgd = SqrtDecayRate(0.01, 5.)
    lr_md = SqrtDecayRate(0.0000001, 10.)
    gd = GradientDescent()
    sgd_100 = GradientDescent()
    md = MirrorDescent()
    gd_svm = SVM(gd, lr_gd, c, 4000, wbc_n, rel_conv)
    sgd_100_svm = SVM(sgd_100, lr_sgd, c, 200, 1, rel_conv)
    md_svm = SVM(md, lr_md, c, 4000, wbc_n, rel_conv)

    tmp_loss, tmp_time = gd_svm.fit(wbc_X_train, wbc_y_train)
    gd_loss_counts += np.sign(tmp_loss)
    gd_loss += tmp_loss
    gd_time += tmp_time
    tmp_loss, tmp_time = sgd_100_svm.fit(wbc_X_train, wbc_y_train)
    sgd_100_loss_counts += np.sign(tmp_loss)
    sgd_100_loss += tmp_loss
    sgd_100_time += tmp_time
    tmp_loss, tmp_time = md_svm.fit(wbc_X_train, wbc_y_train)
    md_loss_counts += np.sign(tmp_loss)
    md_loss += tmp_loss
    md_time += tmp_time

  gd_loss /= gd_loss_counts
  sgd_100_loss /= sgd_100_loss_counts
  md_loss /= md_loss_counts
  gd_time /= 19
  sgd_100_time /= 19
  md_time /= 19
  gd_loss = gd_loss[gd_loss.nonzero()]
  sgd_100_loss = sgd_100_loss[sgd_100_loss.nonzero()]
  md_loss = md_loss[md_loss.nonzero()]

  print('GD average runtime: {}'.format(gd_time))
  print('SGD_100 average runtime: {}'.format(sgd_100_time))
  print('MD average runtime: {}'.format(md_time))

  return (gd_loss, sgd_100_loss, md_loss)
