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

percentiles="50" # 5 35 65 95" #"10 90" #20 30 40 50 60 70 80 90" # "25 75 85 15 45 55"
generateArffOnly=false
generateLPUSourceArff=false

cutoffType=MEDIAN

saveDetailedFoldResult=false
applyFeatureSelectionByCLAMIForSupervisedLearning=false
isSinglePredictionOption=NotSingle #Table1 #Single or any


numClusters=-1


for percentile in $percentiles #$(seq 1 1 $numMaxClusters)
do

for tgt in $datasets
do
	java -cp $libs hk.ust.cse.ipam.jc.selftrainer.CFIBased $dataPath/$groupName/$tgt.arff $groupName $tgt $labelName $posLabel $numClusters $numClusters $percentile $generateArffOnly $generateLPUSourceArff $cutoffType $saveDetailedFoldResult $applyFeatureSelectionByCLAMIForSupervisedLearning false $mlAlg $isSinglePredictionOption > Results/$isSinglePredictionOption\_$groupName\_$tgt\_raw_$mlAlg\_$cutoffType\_$percentile.txt 

#grep "^CLAMI," Results/$groupName\_$tgt\_raw.txt > Results/$groupName\_$tgt.txt
#grep "^[^CLAMI,]" Results/$groupName\_$tgt\_raw.txt > Results/$groupName\_$tgt\_detailed.txt
#rm Results/$groupName\_$tgt\_raw.txt



mail -s "$server $isSinglePredictionOption $cutoffType$percentile CLAMI $groupName $tgt $mlAlg finished!" jaechang.nam@gmail.com < /dev/null

done

done
