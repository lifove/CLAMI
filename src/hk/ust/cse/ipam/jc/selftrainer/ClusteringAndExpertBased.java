package hk.ust.cse.ipam.jc.selftrainer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.ListIterator;

import hk.ust.cse.ipam.utils.Measure;
import hk.ust.cse.ipam.utils.WekaUtils;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;

public class ClusteringAndExpertBased {

	//boolean useBuggyRate = true;
	//int numBuggyClusters = 2;
	ArrayList<Integer> numBuggyInstancesInCluster = new ArrayList<Integer>();

	public static void main(String[] args) {
		new ClusteringAndExpertBased().run(args);
	}

	void run(String[] args) {
		String dataPath = args[0];
		String groupName = args[1];
		String projectName = args[2];
		String classAttributeName = args[3];
		String positiveLabel = args[4];
		//useBuggyRate = Boolean.parseBoolean(args[5]);
		//numBuggyClusters = Integer.parseInt(args[6]);

		// load arff file
		Instances instances = WekaUtils.loadArff(dataPath, classAttributeName);

		Measure measure = expModel(positiveLabel, instances);
		
		double precision = measure.getPrecision();
		double recall = measure.getRecall();
		double fmeasure = measure.getFmeasure();
		
		System.out.println(groupName+","+projectName+"," + precision+"," + recall+"," + fmeasure);
		//System.out.println(TP + " " + TN + " " + FP + " " +FN);

	}

	public Measure expModel(String positiveLabel, Instances instances) {
		Instances instancesForClusters = WekaUtils.getInstancesByRemovingSpecificAttributes(instances, "1-" + (instances.numAttributes()-1), true);

		Measure measure = new Measure();
		
		String[] options = new String[2];
		options[0] = "-I";                 // max. iterations
		options[1] = "100";
		SimpleKMeans kMeans = new SimpleKMeans();  // new instance of clusterer
		try {
			kMeans.setNumClusters(20);
			kMeans.setOptions(options);
			kMeans.setPreserveInstancesOrder(true);
			kMeans.buildClusterer(instancesForClusters);    // build the clusterer

			// print out the cluster centroids
			Instances centroids = kMeans.getClusterCentroids();
			
			int[] assignment = kMeans.getAssignments(); // index is the instance id and the value is the cluster id
			
			/*for (int i = 0; i < centroids.numInstances(); i++) { 
				System.out.println( "Centroid " + (i+1) + ": " + centroids.instance(i)); 
			}*/

			// label clusters by Expert (=use its original label as Expert decides the label)
			// if useBuggyRate is true, always return false but numBuggyInCluster has values
			numBuggyInstancesInCluster.clear();
			boolean[] clusterLabels = new boolean[centroids.numInstances()];
			for (int i = 0; i < kMeans.numberOfClusters(); i++) { 
				//System.out.println( instances.instance(i) + " is in cluster " + (kMeans.clusterInstance(instances.instance(i)) + 1)); 
				clusterLabels[i]=getClusterLabelByProminentLabel(i,instances,assignment,positiveLabel);
			}
			
			boolean noBuggyClusters = !areThereBuggyClusters(clusterLabels);
			
			/*if(noBuggyClusters){
				System.out.println("noBuggyClusters");
			}*/
			
			// no buggy clusters, then find the best number for buggy clusters based on f-measure and report the results for EXP result.
			if(noBuggyClusters){
				ArrayList<ArrayList<Integer>> listIdxOfBuggyCluster = new ArrayList<ArrayList<Integer>>();
				ArrayList<Double> fmeasures = new ArrayList<Double>();
				ArrayList<Integer> TPs = new ArrayList<Integer>();
				ArrayList<Integer> FPs = new ArrayList<Integer>();
				ArrayList<Integer> TNs = new ArrayList<Integer>();
				ArrayList<Integer> FNs = new ArrayList<Integer>();
				// 1 to 19
				for(int numBuggyClusters=1;numBuggyClusters<kMeans.numberOfClusters();numBuggyClusters++){
					//int numBuggyClusters = i;//getNumBuggyClusters(instances,kMeans.numberOfClusters(),positiveLabel);
					ArrayList<Integer> idxOfBuggyCluster = new ArrayList<Integer>();
					
					// init clusterLabels
					for(int i=0;i<clusterLabels.length;i++){
						clusterLabels[i]=false;
					}
					
					// find min clusters
					for(int i=0;i<numBuggyClusters;i++){
						idxOfBuggyCluster.add(getMax(numBuggyInstancesInCluster,idxOfBuggyCluster));
					}
					
					/*for(int i=0; i< numBuggyInstancesInCluster.size();i++){
						System.out.println(i + ": " + numBuggyInstancesInCluster.get(i));
					}*/
					
					for(int i=0;i<numBuggyClusters;i++){
						clusterLabels[idxOfBuggyCluster.get(i)] = true;
						//System.out.println(idxOfBuggyCluster.get(i));
					}
					listIdxOfBuggyCluster.add(idxOfBuggyCluster);
					
					int TP=0,FP=0,TN=0,FN=0;
					
					double buggyValue = WekaUtils.getClassValueIndex(instances, positiveLabel);
					for (int i = 0; i < instances.numInstances(); i++) { 
						double originalLabelValue = instances.instance(i).classValue();
						int clusterIndex = kMeans.clusterInstance(instancesForClusters.instance(i));
						boolean isBuggyCluster = clusterLabels[clusterIndex];
						
						if(isBuggyCluster){
							if(buggyValue==originalLabelValue)
								TP++;
							else
								FP++;
						}else{  // clean cluster
							if(buggyValue==originalLabelValue)
								FN++;
							else
								TN++;
						}
						//System.out.println(instances.instance(i).toString() + " / " + isBuggyCluster);
					}
					TPs.add(TP);
					FPs.add(FP);
					TNs.add(TN);
					FNs.add(FN);
					fmeasures.add(WekaUtils.getFmeasure(TP, FP, TN, FN));
				}
				
				// find index of max fmeasure.
				int maxFmeasureIndex = getMaxIndexFromArrayList(fmeasures);
				
				int TP = TPs.get(maxFmeasureIndex);
				int FP = FPs.get(maxFmeasureIndex);
				int TN = TNs.get(maxFmeasureIndex);
				int FN = FNs.get(maxFmeasureIndex);
				
				measure.addRecall(WekaUtils.getRecall(TP, FP, TN, FN));
				measure.addPrecision(WekaUtils.getPrecision(TP, FP, TN, FN));
				measure.addFmeasure(WekaUtils.getFmeasure(TP, FP, TN, FN));
				measure.addFPR(WekaUtils.getFalsePositiveRate(TP, FP, TN, FN));
				measure.addFNR(WekaUtils.getFalseNegativeRate(TP, FP, TN, FN));
				
				return measure;
			}
			
			// compute Precision, Recall, and F-measure
			int TP=0,FP=0,TN=0,FN=0;
			
			double buggyValue = WekaUtils.getClassValueIndex(instances, positiveLabel);
			for (int i = 0; i < instances.numInstances(); i++) { 
				double originalLabelValue = instances.instance(i).classValue();
				int clusterIndex = kMeans.clusterInstance(instancesForClusters.instance(i));
				boolean isBuggyCluster = clusterLabels[clusterIndex];
				
				if(isBuggyCluster){
					if(buggyValue==originalLabelValue)
						TP++;
					else
						FP++;
				}else{  // clean cluster
					if(buggyValue==originalLabelValue)
						FN++;
					else
						TN++;
				}
				//System.out.println(instances.instance(i).toString() + " / " + isBuggyCluster);
			}
			
			double recall = WekaUtils.getRecall(TP, FP, TN, FN);//(double) TP / (TP + FN);
			double precision = WekaUtils.getPrecision(TP, FP, TN, FN); //(double) TP / (TP + FP);
			double fmeasure = WekaUtils.getFmeasure(TP, FP, TN, FN); //(2*recall*precision)/(recall+precision);
			double accuracy = (double) (TP+TN)/(TP+TN+FP+FN);
			
			//System.out.println("Precision: " + precision);
			
			measure.addRecall(recall);
			measure.addPrecision(precision);
			measure.addFmeasure(fmeasure);
			measure.addFPR(WekaUtils.getFalsePositiveRate(TP, FP, TN, FN));
			measure.addFNR(WekaUtils.getFalseNegativeRate(TP, FP, TN, FN));
			measure.addTP(TP);
			measure.addFP(FP);
			measure.addTN(TN);
			measure.addFN(FN);
			
			
			

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}     // set the options
		return measure;
	}

	private int getMaxIndexFromArrayList(ArrayList<Double> fmeasures) {
		
		int maxIndex = -1;
		double maxFmeasure = 0.0;
		
		for(int i=0; i<fmeasures.size();i++){
			double curFmeasure = fmeasures.get(i);
			if(maxFmeasure<curFmeasure){
				maxFmeasure = curFmeasure;
				maxIndex = i;
			}
			
			//System.out.println(curFmeasure + " " + maxFmeasure);	
		}
		return maxIndex;
	}

	private int getNumBuggyClusters(Instances instances, int numberOfClusters,String posStrLabel) {
		
		int positiveIndex = WekaUtils.getClassValueIndex(instances, posStrLabel);
		int numBuggyInstances = instances.attributeStats(instances.classIndex()).nominalCounts[positiveIndex];
		
		float buggyRate = (float)numBuggyInstances/instances.numInstances();
		
		int numBuggyClusters = Math.round(buggyRate*2*10);
		
		return numBuggyClusters;
	}

	private boolean areThereBuggyClusters(boolean[] clusterLabels) {
		
		for(boolean isBuggy:clusterLabels)
			if(isBuggy)
				return true;
		
		return false;
	}

	private int getMax(ArrayList<Integer> list, ArrayList<Integer> minAlready) {
		
		int maxIndex=-1;
		
		int maxValue = -1;
		
		for(int i=0;i<list.size();i++){
			int curValue = list.get(i);
			if(curValue>maxValue && !minAlready.contains(i)){
				maxValue = curValue;
				maxIndex = i;
			}
		}
		
		return maxIndex;
	}

	private boolean getClusterLabelByProminentLabel(int i, Instances instances, int[] assignment, String posStrLabel) {
		ArrayList<Integer> instIdxInThisCLuster = new ArrayList<Integer>();
		
		// find instances in the cluster i.
		for(int instIdx = 0; instIdx <assignment.length;instIdx++){
			if(assignment[instIdx]==i)
				instIdxInThisCLuster.add(instIdx);
		}
		
		// label the cluster i with the prominent label
		int countBuggy = 0;
		int countClean = 0;
		double buggyLabelValue = WekaUtils.getClassValueIndex(instances, posStrLabel);
		for(int instIdx=0;instIdx<instIdxInThisCLuster.size();instIdx++){
			if(instances.instance(instIdxInThisCLuster.get(instIdx)).classValue()==buggyLabelValue)
				countBuggy++;
			else
				countClean++;
		}
		
		numBuggyInstancesInCluster.add(countBuggy);
		
		//if(useBuggyRate){
		// numBuggyInstancesInCluster.add(countBuggy);
		//	return false;
		//}
		return countBuggy>countClean?true:false;
	}

}
