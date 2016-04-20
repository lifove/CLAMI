libs="/bigstore/msr3/jc/STDP/bin/SelfTrainer/dist/SelfTrainer2.jar:/bigstore/msr3/jc/mybin/JCTools/dist/JCTools.jar:/bigstore/msr3/jc/mybin/JCTools/lib/*"

server=`hostname`

dataPath=$1 #"../../CDDP/CDDP/data"
groupName=$2 #Relink #AEEEM
datasets=$3 #"Apache Safe Zxing"
#mlAlg=$4 #weka.classifiers.functions.Logistic
labelName=$4
posLabel=$5 #TRUEA
numMaxClusters=$6

percentile=50
generateArffOnly=false
generateLPUSourceArff=false

cutoffType=MEDIAN

saveDetailedFoldResult=false
applyFeatureSelectionByCLAMIForSupervisedLearning=false
isSinglePredictionOption=RS_only # Single or any

numClusters=-1

mlAlgs="weka.classifiers.functions.Logistic" #weka.classifiers.trees.J48 weka.classifiers.trees.RandomForest weka.classifiers.bayes.NaiveBayes weka.classifiers.bayes.BayesNet weka.classifiers.functions.SMO"

#for numClusters in $(seq 1 1 $numMaxClusters)
#do

for mlAlg in $mlAlgs
do
for tgt in $datasets
do
	java -cp $libs hk.ust.cse.ipam.jc.selftrainer.CFIBased $dataPath/$groupName/$tgt.arff $groupName $tgt $labelName $posLabel $numClusters $numClusters $percentile $generateArffOnly $generateLPUSourceArff $cutoffType $saveDetailedFoldResult $applyFeatureSelectionByCLAMIForSupervisedLearning false $mlAlg $isSinglePredictionOption > Results/$groupName\_$tgt\_raw_$mlAlg\_2.txt 

#grep "^CLAMI," Results/$groupName\_$tgt\_raw.txt > Results/$groupName\_$tgt.txt
#grep "^[^CLAMI,]" Results/$groupName\_$tgt\_raw.txt > Results/$groupName\_$tgt\_detailed.txt
#rm Results/$groupName\_$tgt\_raw.txt



done
#done
mail -s "$server CLAMI $groupName $mlAlg finished!" jaechang.nam@gmail.com < /dev/null
done
