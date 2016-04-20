libs="/bigstore/msr3/jc/mybin/JCTools/dist/JCTools.jar:/bigstore/msr3/jc/mybin/JCTools/lib/*"

server=`hostname`

dataPath=$1 #"../../CDDP/CDDP/data"
groupName=$2 #Relink #AEEEM
datasets=$3 #"Apache Safe Zxing"
#mlAlg=$4 #weka.classifiers.functions.Logistic
labelName=$4
posLabel=$5 #TRUEA



for tgt in $datasets
do
	java -cp $libs hk.ust.cse.ipam.utils.InstanceFilteringBasedPrediction $dataPath/$groupName/$tgt.arff $groupName $tgt $labelName $posLabel >> Results/$groupName\_$tgt.txt 
done

mail -s "$server STPD $groupName $dataset finished!" jaechang.nam@gmail.com < /dev/null
