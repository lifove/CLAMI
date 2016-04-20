libs="/bigstore/msr3/jc/STDP/bin/SelfTrainer/dist/SelfTrainer.jar:/bigstore/msr3/jc/mybin/JCTools/dist/JCTools.jar:/bigstore/msr3/jc/mybin/JCTools/lib/*"

server=`hostname`

dataPath=$1 #"../../CDDP/CDDP/data"
groupName=$2 #Relink #AEEEM
datasets=$3 #"Apache Safe Zxing"
#mlAlg=$4 #weka.classifiers.functions.Logistic
labelName=$4
posLabel=$5 #TRUEA
numMaxClusters=$6
mlAlg=$7

percentile=50
generateArffOnly=false
generateLPUSourceArff=false

cutoffType=MEDIAN

saveDetailedFoldResult=false
applyFeatureSelectionByCLAMIForSupervisedLearning=false
isSinglePredictionOption=Table1 #Single or any


numClusters=-1


#for numClusters in $(seq 1 1 $numMaxClusters)
#do

for tgt in $datasets
do
	java -cp $libs hk.ust.cse.ipam.jc.selftrainer.CFIBased $dataPath/$groupName/$tgt.arff $groupName $tgt $labelName $posLabel $numClusters $numClusters $percentile $generateArffOnly $generateLPUSourceArff $cutoffType $saveDetailedFoldResult $applyFeatureSelectionByCLAMIForSupervisedLearning false $mlAlg $isSinglePredictionOption > Results/$isSinglePredictionOption\_$groupName\_$tgt\_raw_$mlAlg\.txt 

#grep "^CLAMI," Results/$groupName\_$tgt\_raw.txt > Results/$groupName\_$tgt.txt
#grep "^[^CLAMI,]" Results/$groupName\_$tgt\_raw.txt > Results/$groupName\_$tgt\_detailed.txt
#rm Results/$groupName\_$tgt\_raw.txt



mail -s "$server $isSinglePredictionOption CLAMI $groupName $tgt $mlAlg finished!" jaechang.nam@gmail.com < /dev/null

done
