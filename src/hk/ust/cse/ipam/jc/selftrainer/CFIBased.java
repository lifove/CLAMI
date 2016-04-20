package hk.ust.cse.ipam.jc.selftrainer;

import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.inference.WilcoxonSignedRankTest;

import com.google.common.primitives.Doubles;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveRange;
import hk.ust.cse.ipam.utils.ArrayListUtil;
import hk.ust.cse.ipam.utils.FileUtil;
import hk.ust.cse.ipam.utils.Measure;
import hk.ust.cse.ipam.utils.Measures;
import hk.ust.cse.ipam.utils.SimpleCrossPredictor;
import hk.ust.cse.ipam.utils.WekaUtils;

public class CFIBased {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		new CFIBased().run(args);
	}

	String dataPath = "";
	String projectName = "";
	String groupName = "";
	int percentile = 50;
	String mlAlgorithm = "weka.classifiers.functions.Logistic";
	boolean saveDetailedFoldResult = false;
	boolean featureSelectionByCLAMIForSupervisedLearning = false;
	boolean verbose=true;
	boolean onlyCLA = false;
	boolean singlePrediction = false;

	String selectedAttributeIndices;

	public enum cutoffType {
		MEAN,MEDIAN,MAXMIN,P
	}

	public void run(String[] args) {

		// load parameters
		dataPath = args[0];
		groupName = args[1];
		projectName = args[2];
		String classAttributeName = args[3];
		String positiveLabel = args[4];
		percentile = Integer.parseInt(args[7]);
		String strCuttOffType = args[10];
		saveDetailedFoldResult = Boolean.parseBoolean(args[11]);
		featureSelectionByCLAMIForSupervisedLearning = Boolean.parseBoolean(args[12]);
		verbose = Boolean.parseBoolean(args[13]);
		mlAlgorithm = args[14];
		singlePrediction = args[15].equals("Single")?true:false;

		// load arff file
		Instances instances = WekaUtils.loadArff(dataPath, classAttributeName);

		if(singlePrediction){
			// single prediction
			ArrayList<Measure> CLAMIMeasures = CFIBasedPrediction(instances, positiveLabel, strCuttOffType);
			Measure CLAMeasure = CLAMIMeasures.get(0);
			Measure CLAMIMeasure = CLAMIMeasures.get(1);
			
			String clusterCufoff = strCuttOffType.equals("P")?percentile + ",":"";
			
			System.out.println("CLAMI,D," + groupName + "," + projectName + ","+ mlAlgorithm + "," + clusterCufoff +
						"-," +
						CLAMeasure.getTP() + "," + 
						"-," +
						CLAMeasure.getFP() + "," + 
						"-," +
						CLAMeasure.getTN() + "," + 
						"-," +
						CLAMeasure.getFN() + "," + 
						"-," +
						CLAMeasure.getFPR() + "," + 
						"-," +
						CLAMeasure.getFNR() + "," + 
						"-," +
						CLAMeasure.getPrecision() + "," + 
						"-," +
						CLAMeasure.getRecall() + "," + 
						"-," +
						"-," +
						CLAMeasure.getFmeasure() + "," + 
						"-," +
						CLAMIMeasure.getPrecision() + "," + 
						"-," +
						"-," +
						CLAMIMeasure.getRecall() + "," + 
						"-," +
						"-," +
						CLAMIMeasure.getFmeasure() + "," + 
						"-," +
						"-," +
						CLAMIMeasure.getAUC()+
						",-," + selectedAttributeIndices.replace(",", "|"));
		}
		else{
			// compare randomSplit vs CLAMI
			compareRandomSplitsVSCLAMI(instances,positiveLabel,strCuttOffType);
		}
	}

	private void compareRandomSplitsVSCLAMI(Instances instances,
			String positiveLabel, String strCuttOffType) {
		int folds=2;
		int repeat=500;

		try {
			Classifier classifier = (Classifier) Utils.forName(Classifier.class, mlAlgorithm, null);

			Measures randomSplitsMeasures = new Measures();
			Measures CLAMeasures = new Measures();
			Measures CLAMIMeasures = new Measures();
			Measures THDMeasures = new Measures();
			Measures EXPMeasures = new Measures();


			int posClassIndex = WekaUtils.getClassValueIndex(instances, positiveLabel);

			for(int i=0; i<repeat;i++){
				// randomize with different seed for each iteration
				instances.randomize(new Random(i)); 
				instances.stratify(folds);
				Evaluation eval;

				for(int n=0;n<folds;n++){
					Instances trainingSet = instances.trainCV(folds, n);
					Instances testSet = instances.testCV(folds, n);

					// run CLAMI for testSet
					ArrayList<Measure> CLAandCLAMIMeasure = CFIBasedPrediction(testSet, positiveLabel, strCuttOffType);
					Measure CLAMeasure = CLAandCLAMIMeasure.get(0);
					Measure CLAMIMeasure = CLAandCLAMIMeasure.get(1);

					CLAMeasures.getPrecisions().add(CLAMeasure.getPrecision());
					CLAMeasures.getRecalls().add(CLAMeasure.getRecall());
					CLAMeasures.getFmeasures().add(CLAMeasure.getFmeasure());
					CLAMeasures.getAUCs().add(CLAMeasure.getAUC());
					CLAMeasures.getFPRs().add(CLAMeasure.getFPR());
					CLAMeasures.getFNRs().add(CLAMeasure.getFNR());

					CLAMIMeasures.getPrecisions().add(CLAMIMeasure.getPrecision());
					CLAMIMeasures.getRecalls().add(CLAMIMeasure.getRecall());
					CLAMIMeasures.getFmeasures().add(CLAMIMeasure.getFmeasure());
					CLAMIMeasures.getAUCs().add(CLAMIMeasure.getAUC());
					CLAMIMeasures.getFPRs().add(CLAMIMeasure.getFPR());
					CLAMIMeasures.getFNRs().add(CLAMIMeasure.getFNR());

					// build/test a supervised learning model
					if(featureSelectionByCLAMIForSupervisedLearning){ // generate training and test sets based on metrics selected by CLAMI
						trainingSet = WekaUtils.getInstancesByRemovingSpecificAttributes(trainingSet, selectedAttributeIndices, true);
						testSet = WekaUtils.getInstancesByRemovingSpecificAttributes(testSet, selectedAttributeIndices, true);
					}
					Classifier clsCopy = AbstractClassifier.makeCopy(classifier);
					clsCopy.buildClassifier(trainingSet);
					eval = new Evaluation(instances);
					eval.evaluateModel(clsCopy, testSet);

					randomSplitsMeasures.getPrecisions().add(eval.precision(posClassIndex));
					randomSplitsMeasures.getRecalls().add(eval.recall(posClassIndex));
					randomSplitsMeasures.getFmeasures().add(eval.fMeasure(posClassIndex));
					randomSplitsMeasures.getAUCs().add(eval.areaUnderROC(posClassIndex));
					randomSplitsMeasures.getFPRs().add(eval.falsePositiveRate(posClassIndex));
					randomSplitsMeasures.getFNRs().add(eval.falseNegativeRate(posClassIndex));
					
					// THD model
					ThresholdBased thd = new ThresholdBased();
					Measure thdResult = thd.thdModel(positiveLabel, testSet);
					
					THDMeasures.getPrecisions().add(thdResult.getPrecision());
					THDMeasures.getRecalls().add(thdResult.getRecall());
					THDMeasures.getFmeasures().add(thdResult.getFmeasure());
					THDMeasures.getFPRs().add(thdResult.getFPR());
					THDMeasures.getFNRs().add(thdResult.getFNR());
					
					//EXP model
					ClusteringAndExpertBased exp = new ClusteringAndExpertBased();
					Measure expResult = exp.expModel(positiveLabel, testSet);
					EXPMeasures.getPrecisions().add(expResult.getPrecision());
					EXPMeasures.getRecalls().add(expResult.getRecall());
					EXPMeasures.getFmeasures().add(expResult.getFmeasure());
					EXPMeasures.getFPRs().add(expResult.getFPR());
					EXPMeasures.getFNRs().add(expResult.getFNR());
					
					// selected attributes
					String[] selectedAttributes = selectedAttributeIndices.split(",");

					if(verbose)
						System.out.println("CLAMI,D," + groupName + "," + projectName + ","+ 
								eval.precision(posClassIndex) + "," +
								CLAMIMeasure.getPrecision() + "," + 
								"-," +
								eval.recall(posClassIndex) + "," +
								CLAMIMeasure.getRecall() + "," + 
								"-," +
								eval.fMeasure(posClassIndex) + "," +
								CLAMIMeasure.getFmeasure() + "," + 
								"-," +
								eval.areaUnderROC(posClassIndex) + "," +
								CLAMIMeasure.getAUC()+ "," +
								"-," + selectedAttributeIndices.replace(",", "|") + "," + (selectedAttributes.length-1));
				}
			}

			String result =  "CLAMI,A," + groupName+ "," + projectName +  "," + "," + mlAlgorithm + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getPrecisions(),CLAMeasures.getPrecisions()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getRecalls(),CLAMeasures.getRecalls()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getFmeasures(),CLAMeasures.getFmeasures()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getPrecisions(),CLAMIMeasures.getPrecisions()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getRecalls(),CLAMIMeasures.getRecalls()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getFmeasures(),CLAMIMeasures.getFmeasures()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getAUCs(),CLAMIMeasures.getAUCs()) + "," +
					
					SimpleCrossPredictor.getWinTieLoss(THDMeasures.getPrecisions(),CLAMeasures.getPrecisions()) + "," +
					SimpleCrossPredictor.getWinTieLoss(THDMeasures.getRecalls(),CLAMeasures.getRecalls()) + "," +
					SimpleCrossPredictor.getWinTieLoss(THDMeasures.getFmeasures(),CLAMeasures.getFmeasures()) + "," +
					SimpleCrossPredictor.getWinTieLoss(THDMeasures.getPrecisions(),CLAMIMeasures.getPrecisions()) + "," +
					SimpleCrossPredictor.getWinTieLoss(THDMeasures.getRecalls(),CLAMIMeasures.getRecalls()) + "," +
					SimpleCrossPredictor.getWinTieLoss(THDMeasures.getFmeasures(),CLAMIMeasures.getFmeasures()) + "," +
					
					SimpleCrossPredictor.getWinTieLoss(EXPMeasures.getPrecisions(),CLAMeasures.getPrecisions()) + "," +
					SimpleCrossPredictor.getWinTieLoss(EXPMeasures.getRecalls(),CLAMeasures.getRecalls()) + "," +
					SimpleCrossPredictor.getWinTieLoss(EXPMeasures.getFmeasures(),CLAMeasures.getFmeasures()) + "," +
					SimpleCrossPredictor.getWinTieLoss(EXPMeasures.getPrecisions(),CLAMIMeasures.getPrecisions()) + "," +
					SimpleCrossPredictor.getWinTieLoss(EXPMeasures.getRecalls(),CLAMIMeasures.getRecalls()) + "," +
					SimpleCrossPredictor.getWinTieLoss(EXPMeasures.getFmeasures(),CLAMIMeasures.getFmeasures()) + "," +
					
					SimpleCrossPredictor.getWinTieLoss(CLAMeasures.getPrecisions(),CLAMIMeasures.getPrecisions()) + "," +
					SimpleCrossPredictor.getWinTieLoss(CLAMeasures.getRecalls(),CLAMIMeasures.getRecalls()) + "," +
					SimpleCrossPredictor.getWinTieLoss(CLAMeasures.getFmeasures(),CLAMIMeasures.getFmeasures()) +"," +
					
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getPrecisions(),THDMeasures.getPrecisions()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getRecalls(),THDMeasures.getRecalls()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getFmeasures(),THDMeasures.getFmeasures()) + "," +
					
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getPrecisions(),EXPMeasures.getPrecisions()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getRecalls(),EXPMeasures.getRecalls()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getFmeasures(),EXPMeasures.getFmeasures()) + "," +
					
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getFPRs(),THDMeasures.getFPRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getFNRs(),THDMeasures.getFNRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getFPRs(),EXPMeasures.getFPRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getFNRs(),EXPMeasures.getFNRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getFPRs(),CLAMeasures.getFPRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getFNRs(),CLAMeasures.getFNRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getFPRs(),CLAMIMeasures.getFPRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(randomSplitsMeasures.getFNRs(),CLAMIMeasures.getFNRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(THDMeasures.getFPRs(),CLAMeasures.getFPRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(THDMeasures.getFNRs(),CLAMeasures.getFNRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(THDMeasures.getFPRs(),CLAMIMeasures.getFPRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(THDMeasures.getFNRs(),CLAMIMeasures.getFNRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(EXPMeasures.getFPRs(),CLAMeasures.getFPRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(EXPMeasures.getFNRs(),CLAMeasures.getFNRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(EXPMeasures.getFPRs(),CLAMIMeasures.getFPRs()) + "," +
					SimpleCrossPredictor.getWinTieLoss(EXPMeasures.getFNRs(),CLAMIMeasures.getFNRs());
					
					
					

			System.out.println(result);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public ArrayList<Measure> CFIBasedPrediction(Instances instances,String positiveLabel,String strCuttOffType) {

		ArrayList<Measure> measures = new ArrayList<Measure>(); // 0: CLA 1 :CLAMI


		// create HashMaps that contain cluster information and list of instance indices
		HashMap<Integer,ClusterInfo> numPositivelyAgreedFeatures = new HashMap<Integer,ClusterInfo>(); // key: instance index, value: ClusterInfo
		HashMap<Integer,String> instIdxByAgreedFeatureNum = new HashMap<Integer,String>();// key: numAgreedFeatures value; list of instance indices, instance index starts from 1

		// Retrieve each instance and fill HashMaps
		for(int instIdx=0;instIdx<instances.numInstances();instIdx++){
			int numAgreed = 0;
			String associationKey = "";
			for(int attrIdx=0;attrIdx<instances.numAttributes()-1;attrIdx++){
				if(attrIdx==instances.classIndex())
					continue;

				double attributeValue = instances.instance(instIdx).value(attrIdx);

				double attributeCutoffValue = getCutoffValue(strCuttOffType,instances,attrIdx);

				if(attributeValue>attributeCutoffValue){
					associationKey += attrIdx + "-";
					numAgreed++;
				}
			}
			ClusterInfo clusterInfo = new ClusterInfo();
			clusterInfo.numOfAgreedFeautures = numAgreed;
			clusterInfo.associatedFeatures = associationKey;
			//instances.attribute(labelName).indexOfValue(posClassValue);
			clusterInfo.originalLabel = WekaUtils.getStringValueOfInstanceLabel(instances,instIdx);
			numPositivelyAgreedFeatures.put(instIdx, clusterInfo);
			if(instIdxByAgreedFeatureNum.containsKey(numAgreed))
				instIdxByAgreedFeatureNum.put(numAgreed,instIdxByAgreedFeatureNum.get(numAgreed) + (instIdx+1) +",");
			else{
				instIdxByAgreedFeatureNum.put(numAgreed,(instIdx+1)+",");
			}
		}

		// get the number of all clusters
		int numAllClustersGenerated = instIdxByAgreedFeatureNum.size();
		// set the number of clusters by top half and bottom half
		int numSelectedMaxClusters = numAllClustersGenerated/2;
		int numSelectedMinClusters = numAllClustersGenerated-numSelectedMaxClusters;

		// get instance index in top `n' clusters, n=numSelectedMaxClusters
		// instIdx starts from 1
		String instIdxWithMaxAgreedNum = getInstancesWithMaxAgreedNum(instIdxByAgreedFeatureNum,instances.numAttributes(),numSelectedMaxClusters);
		// get instance index in bottom `n' clusters, n=numSelectedMinClusters
		// instIdx starts from 1
		String instIdxWithMinAgreedNum = getInstancesWithMinAgreedNum(instIdxByAgreedFeatureNum,instances.numAttributes(),numSelectedMinClusters);


		// compute prediction performance only after CLA
		Measure reslutsCLA = getPredictionPerformanceOfCLA(instances,instIdxWithMaxAgreedNum,instIdxWithMinAgreedNum,positiveLabel);
		measures.add(reslutsCLA);


		// feature selection
		//selectedAttributeIndices = getselectedAttributeIndices(numPositivelyAgreedFeatures,
		//		instIdxByAgreedFeatureNum,
		//		numSelectedMaxClusters,numSelectedMinClusters,
		//		instances.classIndex());
		int nthBestConflictScore = 1;
		selectedAttributeIndices = getSelectedAttriuteIndicesBasedOnConflictScore(instances,instIdxWithMaxAgreedNum,instIdxWithMinAgreedNum,instances.classIndex(),nthBestConflictScore);


		// generate and label the training dataset from clusters
		Instances rawSourceInstances = generateAndLabelNewSourceData(instances,instIdxWithMaxAgreedNum,instIdxWithMinAgreedNum,positiveLabel,false);
		Instances newSourceInstances = null;
		// do prediction or save dataset files
		// don't conduct STDP in case that selected clusters are greater than the the number of all clusters.

		if(numAllClustersGenerated < numSelectedMaxClusters + numSelectedMinClusters){
			System.out.println("Terminated since numAllClustersGenerated < numSelectedMaxClusters + numSelectedMinClusters");
			System.exit(0);
		}

		Instances tarInstances = new Instances(instances);

		// initial feature selection
		newSourceInstances = WekaUtils.getInstancesByRemovingSpecificAttributes(rawSourceInstances,
				this.selectedAttributeIndices, true);
		tarInstances = WekaUtils.getInstancesByRemovingSpecificAttributes(instances,
				this.selectedAttributeIndices, true);

		//System.out.println(selectedAttributeIndices);


		//--------------------
		// instance selection
		//--------------------
		while(true){

			// instance selection based on instances whose all feature values are greater than mean attribute value (buggy) or less than / same as mean attribute value 
			newSourceInstances = selectInstances(newSourceInstances,tarInstances, positiveLabel);
			int numBuggyNewInstances = WekaUtils.getNumInstancesByClass(newSourceInstances, positiveLabel);

			//if(nthBestConflictScore==selectedAttributeIndices.split(",").length-1){ // if n-th is the last two feature
			//newSourceInstances = WekaUtils.getInstancesByRemovingSpecificAttributes(rawSourceInstances, selectedAttributeIndices, true);
			//tarInstances = WekaUtils.getInstancesByRemovingSpecificAttributes(instances, selectedAttributeIndices, true);;
			//break;
			//}

			if(numBuggyNewInstances==0 || newSourceInstances.numInstances()-numBuggyNewInstances==0){
				// select less features to get more instances, repeat this until both buggy and clean instances are selected.
				selectedAttributeIndices = getSelectedAttriuteIndicesBasedOnConflictScore(instances,instIdxWithMaxAgreedNum,instIdxWithMinAgreedNum,instances.classIndex(),++nthBestConflictScore);
				//System.out.println(selectedAttributeIndices);

				newSourceInstances = WekaUtils.getInstancesByRemovingSpecificAttributes(rawSourceInstances,
						this.selectedAttributeIndices, true);
				tarInstances = WekaUtils.getInstancesByRemovingSpecificAttributes(instances,
						this.selectedAttributeIndices, true);
				
				// if there are no other conflict scores, stop the lopp. CLAMI can't work! 
				if(nthBestConflictScore > instances.numAttributes()){
					Measure measure = new Measure();
					measure.addPrecision(0.0);
					measure.addRecall(0.0);
					measure.addAUC(0.0);
					measure.addFmeasure(0.0);
					measure.addFPR(0.0);
					measure.addFNR(0.0);
					
					measures.add(measure);
					
					//System.out.println("No instances selected!");
					//System.exit(0);
					return measures;
				}

				continue;
			}
			break;
		}

		//System.out.println("training dataset ready!");
		// run cross-prediction
		//System.out.println(groupName + "," + projectName + "," + SimpleCrossPredictor.crossPredictionOnTheSameSplit(newSourceInstances, tarInstances, instances, positiveLabel, repeat, folds));

		Measure measure = SimpleCrossPredictor.crossPrediction(newSourceInstances, tarInstances, mlAlgorithm, positiveLabel,saveDetailedFoldResult);

		/*System.out.println("CLAMI,D," + groupName + "," + projectName + ","+ 
							"-," +
							measure.getPrecision() + "," + 
							"-," +
							"-," +
							measure.getRecall() + "," + 
							"-," +
							"-," +
							measure.getFmeasure() + "," + 
							"-," +
							"-," +
							measure.getAUC()+
							"-," + selectedAttributeIndices.replace(",", "|"));*/

		//ArrayList<Double> means = new ArrayList<Double>();
		//ArrayList<Double> variances = new ArrayList<Double>();


		// remove attributes who small variance
		/*for(int attrIdx=0;attrIdx<tarInstances.numAttributes();attrIdx++){
			if(attrIdx==tarInstances.classIndex()){
				continue;
			}

			double[] values= tarInstances.attributeToDoubleArray(attrIdx);
			DescriptiveStatistics stat = new DescriptiveStatistics(values);

			if (((Double)stat.getMean()).equals(Double.NaN) || ((Double)stat.getVariance()).equals(Double.NaN ))
				continue;

			//System.out.println(attrIdx + ":" + stat.getMean() + "," + stat.getVariance());

			means.add(stat.getMean());
			variances.add(stat.getVariance());
		}

		double meanOfMeans = ArrayListUtil.getMedian(means);
		double meanOFVariances = ArrayListUtil.getMedian(variances);*/

		//System.out.println(projectName + "," + meanOFVariances +"," + meanOfMeans + "," + means.size());

		measures.add(measure);

		return measures;

		//System.out.println(groupName + "," + projectName + "," + SimpleCrossPredictor.crossPrediction(newSourceInstances, tarInstances, "weka.classifiers.functions.Logistic", positiveLabel));


		//crossPredictionOnTheSameSplit(newSourceInstances,instances,positiveLabel,applyFeatureSelection);
	}

	private Measure getPredictionPerformanceOfCLA(Instances instances,
			String instIdxLabeledAsBuggyByCLA, String instIdxLabeledAsCleanByCLA,String positiveLabel) {

		Measure measure = new Measure();
		int posClassIndex = WekaUtils.getClassValueIndex(instances, positiveLabel);

		int TP=0,FP=0,TN=0,FN=0;

		String[] buggyInstanceIdx = instIdxLabeledAsBuggyByCLA.split(",");
		String[] cleanInstanceIdx = instIdxLabeledAsCleanByCLA.split(",");

		for(String idx:buggyInstanceIdx){
			int instanceIdx = Integer.parseInt(idx)-1;
			double originalLabel = instances.get(instanceIdx).value(instances.classAttribute());
			if(originalLabel==posClassIndex)
				TP++;
			else
				FP++;
		}

		for(String idx:cleanInstanceIdx){
			int instanceIdx = Integer.parseInt(idx)-1;
			double originalLabel = instances.get(instanceIdx).value(instances.classAttribute());
			if(originalLabel==posClassIndex)
				FN++;
			else
				TN++;
		}

		measure.addPrecision(WekaUtils.getPrecision(TP, FP, TN, FN));
		measure.addRecall(WekaUtils.getRecall(TP, FP, TN, FN));
		measure.addFmeasure(WekaUtils.getFmeasure(TP, FP, TN, FN));
		measure.addFPR(WekaUtils.getFalsePositiveRate(TP, FP, TN, FN));
		measure.addFNR(WekaUtils.getFalseNegativeRate(TP, FP, TN, FN));
		measure.addTP(TP);
		measure.addFP(FP);
		measure.addTN(TN);
		measure.addFN(FN);


		return measure;
	}

	private Instances selectInstances(Instances newSourceInstances, Instances tarInstances, String strPosLabel) {

		String instanceIndicesToBeRemoved = "";
		for(int instIdx=0;instIdx<newSourceInstances.numInstances();instIdx++){
			for(int attrIdx=0;attrIdx<newSourceInstances.numAttributes();attrIdx++){
				if(attrIdx==newSourceInstances.classIndex())
					continue;

				//double attrMean = tarInstances.attributeStats(attrIdx).numericStats.mean;
				double attrMedian = tarInstances.kthSmallestValue(attrIdx, tarInstances.numInstances()/2);

				// if buggy
				if(newSourceInstances.instance(instIdx).value(newSourceInstances.classIndex())==WekaUtils.getClassValueIndex(newSourceInstances, strPosLabel)){
					if(newSourceInstances.instance(instIdx).value(attrIdx)<=attrMedian){
						instanceIndicesToBeRemoved += (instIdx+1) + ",";
						break;
					}
				}
				else{ // if clean instance
					if(newSourceInstances.instance(instIdx).value(attrIdx)>attrMedian){
						instanceIndicesToBeRemoved += (instIdx+1) + ",";
						break;
					}
				}
			}
		}

		return WekaUtils.getInstancesByRemovingSpecificInstances(newSourceInstances,instanceIndicesToBeRemoved,false);
	}

	/**
	 * get selected attribute indices based on conflict score
	 * Conflict score is computed by (# of instances whose feature values > Mean in top clusters)/# of instances in top clusters
	 * @param instances
	 * @param instIdxWithMaxAgreedNum
	 * @param instIdxWithMinAgreedNum
	 * @param classIndex
	 * @return
	 */
	private String getSelectedAttriuteIndicesBasedOnConflictScore(
			Instances instances, String instIdxWithMaxAgreedNum,String instIdxWithMinAgreedNum, int classIndex,int nthBestConflictScore) {

		String selectedFeatures = "";
		String[] instIndicesInTopClusters = instIdxWithMaxAgreedNum.split(",");
		String[] instIndicesInBottomClusters = instIdxWithMinAgreedNum.split(",");
		double[] conflictScore = new double[instances.numAttributes()-1];

		for(int attrIdx=0;attrIdx<instances.numAttributes();attrIdx++){	// attrIdx starts from 0
			if(attrIdx==classIndex)
				continue;

			double attrMedian = instances.kthSmallestValue(attrIdx, instances.numInstances()/2);//instances.attributeStats(attrIdx).numericStats.mean;
			int conflictCounter = 0;
			// conflicts on top clusters
			for(int i=0;i<instIndicesInTopClusters.length;i++){
				int instIdx = Integer.parseInt(instIndicesInTopClusters[i])-1; // index in instIndicesInTopClusters starts from 1
				if(instances.get(instIdx).value(attrIdx)<=attrMedian)
					conflictCounter++;
			}
			// conflicts on bottom clusters
			for(int i=0;i<instIndicesInBottomClusters.length;i++){
				int instIdx = Integer.parseInt(instIndicesInBottomClusters[i])-1; // index in instIndicesInTopClusters starts from 1
				if(instances.get(instIdx).value(attrIdx)>attrMedian)
					conflictCounter++;
			}

			conflictScore[attrIdx] = (double)conflictCounter/(instIndicesInTopClusters.length+instIndicesInBottomClusters.length);
		}

		double cutoffOfConflictScore = getMinCscore(conflictScore);//getBestFeatureConflictScoreCutoff(conflictScore,instances,instIdxWithMaxAgreedNum,instIdxWithMinAgreedNum,classIndex,nthBestConflictScore);
		//double cutoffOfConflictScore = getBestFeatureConflictScoreCutoff(conflictScore,instances,instIdxWithMaxAgreedNum,instIdxWithMinAgreedNum,classIndex,nthBestConflictScore);

		//System.out.println(cutoffPercentileForConflictScore);
		for(int attrIdx=0;attrIdx<conflictScore.length;attrIdx++){
			if(conflictScore[attrIdx] <= cutoffOfConflictScore)// || conflictScore[attrIdx]==0.0)
				selectedFeatures+= (attrIdx+1) +",";
		}

		return selectedFeatures + (classIndex + 1);
	}

	private double getMinCscore(double[] conflictScore) {

		return (new DescriptiveStatistics(conflictScore)).getMin();
	}

	/**
	 * get the best conflict score cutoff based on minimizing conflict score in both instance and feature conflict score
	 * @param featureConflictScores // array index starts from 0
	 * @param instances
	 * @param instIdxWithMaxAgreedNum
	 * @param instIdxWithMinAgreedNum
	 * @param classIndex
	 * @return
	 */
	private double getBestFeatureConflictScoreCutoff(double[] featureConflictScores,Instances instances,
			String instIdxWithMaxAgreedNum, String instIdxWithMinAgreedNum,
			int classIndex,int nthBestConflictScore) {

		HashMap<Double,Double> totalConflictScores = new HashMap<Double,Double>(); // key: total conflict score, value: "currentConflictScoreCutoff" 

		// compute total conflict scores
		ArrayList<Double> computedConflictScoreCutoff = new ArrayList<Double>(); // to skip already processed score
		for(int attrIdx=0;attrIdx<featureConflictScores.length;attrIdx++){ 
			// to avoid recomputing total conflict score in the case that feature conflict score is same
			if(computedConflictScoreCutoff.contains(featureConflictScores[attrIdx])){
				continue;
			}

			computedConflictScoreCutoff.add(featureConflictScores[attrIdx]);

			ArrayList<Integer> selectedAttributes = new ArrayList<Integer>(); // selected features by using currentFeatureConflictScoreCutoff
			// selected attributes for the current conflict score cutoff
			selectedAttributes = getSelectedAttributes(instances,featureConflictScores,featureConflictScores[attrIdx]);
			//if(selectedAttributes.size()==1) // at least 2 features required for this conflict score based approach
			//	continue;

			// Compute instance conflict score using selected attributes by the current conflictScoreCutoff
			// (1) count conflicts from top clusters
			String[] instIndicesInTopClusters = instIdxWithMaxAgreedNum.split(",");
			int numConflictAttributes = 0;
			int numAllAttributes = 0;
			for(int i = 0;i<instIndicesInTopClusters.length;i++){
				int instIdx = Integer.parseInt(instIndicesInTopClusters[i])-1; // index in instIndicesInTopClusters starts from 1

				for(int selectedAttrIdx:selectedAttributes){
					double attrMedian = instances.kthSmallestValue(selectedAttrIdx, instances.numInstances()/2);//instances.attributeStats(attrIdx).numericStats.mean;
					if(instances.get(instIdx).value(selectedAttrIdx) <=attrMedian){
						numConflictAttributes++;
						//break;
					}
					numAllAttributes++;
				}
			}
			// (2) count conflicts from bottom clusters
			String[] instIndicesInBottomClusters = instIdxWithMinAgreedNum.split(",");
			for(int i = 0;i<instIndicesInBottomClusters.length;i++){
				int instIdx = Integer.parseInt(instIndicesInBottomClusters[i])-1; // index in instIndicesInTopClusters starts from 1

				for(int selectedAttrIdx:selectedAttributes){
					double attrMedian = instances.kthSmallestValue(selectedAttrIdx, instances.numInstances()/2);//instances.attributeStats(attrIdx).numericStats.mean;
					if(instances.get(instIdx).value(selectedAttrIdx) > attrMedian){
						numConflictAttributes++;
						//break;
					}	
					numAllAttributes++;
				}
			}

			// (3) compute 
			double InstanceConflictScore = ((double)numConflictAttributes/numAllAttributes);//(instIndicesInTopClusters.length+instIndicesInBottomClusters.length); // always > 0

			// compute the total conflict score and put it into totalConflictScores
			//totalConflictScores.put(((instances.numAttributes())-selectedAttributes.size())*(currentFeatureConflictScoreCutoff + InstanceConflictScore)/2,currentFeatureConflictScoreCutoff);
			totalConflictScores.put(((featureConflictScores[attrIdx] + InstanceConflictScore)/2)/selectedAttributes.size(),featureConflictScores[attrIdx]);
			//System.out.println(selectedAttributes.size() + "," + featureConflictScores[attrIdx] + "," + InstanceConflictScore + "," + ((featureConflictScores[attrIdx] + InstanceConflictScore)/2));

		}

		// find minimum total conflict scores and its key is the best cutoff for the feature conflict score.
		SortedSet<Double> keys = new TreeSet<Double>(totalConflictScores.keySet()); // keys are in ascending order.

		double bestCutoff = -1;
		int nth=0;

		/*for(Double key:keys){
			System.out.println(key + " " + totalConflictScores.get(key));
		}*/

		for(Double key:keys){
			bestCutoff = totalConflictScores.get(key);
			nth++;
			if(nthBestConflictScore==nth)
				break;
		}

		return bestCutoff;
	}

	private ArrayList<Integer> getSelectedAttributes(Instances instances,
			double[] featureConflictScores,double cutoff) {

		ArrayList<Integer> selectedFeatures = new ArrayList<Integer>();
		for(int attrIdx=0;attrIdx<featureConflictScores.length;attrIdx++){
			if(featureConflictScores[attrIdx] <= cutoff)// || conflictScore[attrIdx]==0.0)
				selectedFeatures.add(attrIdx);
		}
		return selectedFeatures;
	}

	public String getSelectedAttributeIndices(){
		return selectedAttributeIndices;
	}

	private double getCutoffValue(String strCuttOffType,Instances instances,int attrIdx) {
		DescriptiveStatistics stat = null;
		switch(cutoffType.valueOf(strCuttOffType)){
		case MEAN:
			return instances.attributeStats(attrIdx).numericStats.mean;
		case MEDIAN:
			stat = new DescriptiveStatistics(instances.attributeToDoubleArray(attrIdx));
			return stat.getPercentile(50);
		case MAXMIN:
			return (instances.attributeStats(attrIdx).numericStats.max+
					instances.attributeStats(attrIdx).numericStats.min)/2;
		case P:
			stat = new DescriptiveStatistics(instances.attributeToDoubleArray(attrIdx));
			return stat.getPercentile(percentile);
		}

		return instances.attributeStats(attrIdx).numericStats.mean;
	}

	private String getInstancesWithMaxAgreedNum(
			HashMap<Integer, String> instIdxByAgreedFeatureNum,int numFeatures,int numSelectedMaxClusters) {

		ArrayList<Integer> maxKeys = new ArrayList<Integer>(); // key for instances with the same agreed features.

		int clusterCount = 0;
		for(int key=numFeatures;key>=0;key--){
			if(instIdxByAgreedFeatureNum.containsKey(key)){
				maxKeys.add(key);
				if(clusterCount==numSelectedMaxClusters-1)
					break;
				clusterCount++;
			}
		}

		return getInstIdxByAgreedFeatureNum(instIdxByAgreedFeatureNum,maxKeys);
	}

	private String getInstancesWithMinAgreedNum(
			HashMap<Integer, String> instIdxByAgreedFeatureNum,int numFeatures,int numSelectedMinClusters) {

		ArrayList<Integer> minKeys = new ArrayList<Integer>();

		int clusterCount = 0;

		for(int key=0;key<=numFeatures;key++){
			if(instIdxByAgreedFeatureNum.containsKey(key)){
				minKeys.add(key);
				if(clusterCount==numSelectedMinClusters-1)
					break;
				clusterCount++;
			}
		}

		return getInstIdxByAgreedFeatureNum(instIdxByAgreedFeatureNum,minKeys);
	}

	private String getInstIdxByAgreedFeatureNum(HashMap<Integer, String> instIdxByAgreedFeatureNum,ArrayList<Integer> keys) {
		String indices = "";

		for(int key:keys){
			indices += instIdxByAgreedFeatureNum.get(key);
		}

		return indices;
	}

	int newPosInstCount = 0;
	int newNegInstCount = 0;

	private Instances generateAndLabelNewSourceData(Instances instances,String selectedInstanceIdxForBuggy,String selectedInstanceIdxForClean,String posLabel,boolean forLPUSource) {

		Instances newInstances = null;

		RemoveRange instFilter = new RemoveRange();
		RemoveRange instFilter2 = new RemoveRange();
		instFilter.setInstancesIndices(selectedInstanceIdxForBuggy);
		instFilter.setInvertSelection(true);

		if(!forLPUSource){
			instFilter2.setInstancesIndices(selectedInstanceIdxForClean);
			instFilter2.setInvertSelection(true);
		}else{
			instFilter2.setInstancesIndices(selectedInstanceIdxForBuggy);
			instFilter2.setInvertSelection(false);
		}

		try {
			instFilter.setInputFormat(instances);
			Instances newPosInstances = Filter.useFilter(instances, instFilter);
			for(int instIdx=0;instIdx<newPosInstances.numInstances();instIdx++){
				newPosInstances.get(instIdx).setClassValue(posLabel);
			}

			newPosInstCount = newPosInstances.numInstances();

			instFilter2.setInputFormat(instances);
			Instances newNegInstances = Filter.useFilter(instances, instFilter2);
			for(int instIdx=0;instIdx<newNegInstances.numInstances();instIdx++){
				newNegInstances.get(instIdx).setClassValue(WekaUtils.getNegClassStringValue(newNegInstances, instances.classAttribute().name(), posLabel));
			}

			newNegInstCount = newNegInstances.numInstances();

			newInstances = new Instances(newPosInstances);

			for(int instIdx=0;instIdx<newNegInstances.numInstances();instIdx++){
				newInstances.add(newNegInstances.get(instIdx));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return newInstances;
	}
}

class ClusterInfo{
	int numOfAgreedFeautures = 0;
	String associatedFeatures = "";
	String originalLabel = "";
}
