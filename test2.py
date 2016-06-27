import numpy


from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities  import percentError
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pybrain.supervised.trainers import BackpropTrainer

ourdataset=SupervisedDataSet(4,1)

with open('newdata.txt') as fp:
	for line in fp:
		x=line.split(",")
		ourclass=x[4].split("\n")[0]
		if "Iris-virginica" in ourclass:
			nameclass=0
				
		elif "Iris-setosa" in ourclass:
			nameclass=1

		else:
			nameclass=2
				
		oursample=x[0:4]
		ourdataset.addSample(oursample,nameclass)
#print ourdataset['input']

ourdataset2=ourdataset['input'][0:30]
ourdataset3=ourdataset['input'][50:80]
ourdataset4=ourdataset['input'][100:130]

traindataset=numpy.concatenate((ourdataset2,ourdataset3,ourdataset4),axis=0)


ourdataset5=ourdataset['input'][30:50]
ourdataset6=ourdataset['input'][80:100]
ourdataset7=ourdataset['input'][130:150]
net = buildNetwork(4, 5, 1)
testdataset=numpy.concatenate((ourdataset5,ourdataset6,ourdataset7),axis=0)


trainer = BackpropTrainer(net, ourdataset)


rnresult = percentError(trainer.testOnClassData(),ourdataset['input'])
tstresult = percentError(trainer.testOnClassData(dataset=ourdataset ), ourdataset['input'])

print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % rnresult, \
          "  test error: %5.2f%%" % tstresult


#print len(testdataset)

# net = buildNetwork(4, 5, 1)
# trainer = BackpropTrainer(net, ourdataset)
# trainer.train()




#print oursample

