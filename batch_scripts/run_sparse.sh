libs="/bigstore/msr3/jc/mybin/JCTools/dist/JCTools.jar:/bigstore/msr3/jc/mybin/JCTools/lib/*"
server=`hostname`

dataPath=$1 #"../../CDDP/CDDP/data"
groupName=$2 #Relink #AEEEM
datasets=$3 #"Apache Safe Zxing"
#mlAlg=$4 #weka.classifiers.functions.Logistic



for tgt in $datasets
do
	java -cp $libs weka.filters.unsupervised.instance.NonSparseToSparse -i $dataPath/$groupName/$tgt\_LPU.arff > sparsedata/$groupName/$tgt.arff
done
mail -s "$server sparse $groupName finished!" jaechang.nam@gmail.com < /dev/null
