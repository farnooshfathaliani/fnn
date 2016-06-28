import numpy

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities  import percentError
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer


from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

#ourdataset=SupervisedDataSet(4,1)
ourdataset=ClassificationDataSet(4, 1, nb_classes=3)



with open('newdata.txt') as fp:
	for line in fp:
		splitedline=line.split(",")
		ourclass=splitedline[4].split("\n")[0]
		if "Iris-virginica" in ourclass:
			nameclass=0
				
		elif "Iris-setosa" in ourclass:
			nameclass=1

		else:
			nameclass=2
				
		oursample=splitedline[0:4]
		ourdataset.addSample(oursample,nameclass)
tstdata, trndata = ourdataset.splitWithProportion( 0.25 )


trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

#print trndata['class']
#print len(trndata)

fnn = buildNetwork( trndata.indim, 15, trndata.outdim, outclass=SoftmaxLayer )

trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

for i in range(200):

	 trainer.trainEpochs( i )

trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )

print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult






#print ourdataset['input']

#ourdataset2=ourdataset['input'][0:30]
#ourdataset3=ourdataset['input'][50:80]
#ourdataset4=ourdataset['input'][100:130]

#traindataset=numpy.concatenate((ourdataset2,ourdataset3,ourdataset4),axis=0)


 


# trainer = BackpropTrainer(net, ourdataset)


# rnresult = percentError(trainer.testOnClassData(),ourdataset['input'])
# tstresult = percentError(trainer.testOnClassData(dataset=ourdataset ), ourdataset['input'])

# print "epoch: %4d" % trainer.totalepochs, \
#           "  train error: %5.2f%%" % rnresult, \
#           "  test error: %5.2f%%" % tstresult


#print len(testdataset)

# net = buildNetwork(4, 5, 1)
# trainer = BackpropTrainer(net, ourdataset)
# trainer.train()




#print oursample

