libs="/bigstore/msr3/jc/mybin/JCTools/dist/JCTools.jar:/bigstore/msr3/jc/mybin/JCTools/lib/*"
server=`hostname`

dataPath=$1 #"../../CDDP/CDDP/data"
groupName=$2 #Relink #AEEEM
datasets=$3 #"Apache Safe Zxing"
attributeName=$4
posLabel=$5
#mlAlg=$4 #weka.classifiers.functions.Logistic



for tgt in $datasets
do
	java -cp $libs hk.ust.cse.ipam.weka.lpuutil.Main -transform lpu $dataPath/$groupName/$tgt.arff $dataPath/$groupName/$tgt $attributeName $posLabel
done
mail -s "$server lpu data $groupName finished!" jaechang.nam@gmail.com < /dev/null
