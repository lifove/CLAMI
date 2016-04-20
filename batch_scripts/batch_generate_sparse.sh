#mlAlg=weka.classifiers.functions.Logistic
dataPath="data"

sh run_sparse.sh $dataPath AEEEM "EQ JDT PDE LC ML" class buggy $applyFeatureSelection &

sh run_sparse.sh $dataPath Relink "Apache Safe Zxing" isDefective TRUE $applyFeatureSelection &

sh run_sparse.sh $dataPath NASA "cm1 mw1 pc1 pc3 pc4"  Defective Y $applyFeatureSelection &

sh run_sparse.sh $dataPath SOFTLAB "ar1 ar3 ar4 ar5 ar6" defects true $applyFeatureSelection &

sh run_sparse.sh $dataPath promise "ant-1.3 arc camel-1.0 poi-1.5 redaktor skarbonka tomcat velocity-1.4 xalan-2.4 xerces-1.2" bug buggy $applyFeatureSelection &
