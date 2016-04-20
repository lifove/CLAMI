libs="/bigstore/msr3/jc/mybin/JCTools/dist/JCTools.jar:/bigstore/msr3/jc/mybin/JCTools/lib/*"

server=`hostname`

dataPath=$1 #"../../CDDP/CDDP/data"
groupName=$2 #Relink #AEEEM
datasets=$3 #"Apache Safe Zxing"
#mlAlg=$4 #weka.classifiers.functions.Logistic
labelName=$4
posLabel=$5 #TRUE

repeat=500
folds=2

for tgt in $datasets
do
	java -cp $libs hk.ust.cse.ipam.utils.SimpleCrossPredictor $groupName\,$tgt\_lpu lpudata/arff_$groupName/$tgt.arff $dataPath/$groupName/$tgt.arff $labelName $posLabel $repeat $folds >> Results/lpu/$groupName.txt
done
mail -s "$server lpu predictor $groupName finished!" jaechang.nam@gmail.com < /dev/null
