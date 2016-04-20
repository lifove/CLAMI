package hk.ust.cse.ipam.jc.selftrainer;

import hk.ust.cse.ipam.utils.ArrayListUtil;
import hk.ust.cse.ipam.utils.WekaUtils;

import java.util.ArrayList;
import java.util.List;

import com.google.common.primitives.Doubles;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

public class SimpleClusteringBased {
	public static void main(String[] args) {
		new SimpleClusteringBased().run(args);
	}

	void run(String[] args) {
		String dataPath = args[0];
		String groupName = args[1];
		String projectName = args[2];
		String classAttributeName = args[3];
		String positiveLabel = args[4];

		// load arff file
		Instances instances = WekaUtils.loadArff(dataPath, classAttributeName);

		Instances instancesForClusters = WekaUtils.getInstancesByRemovingSpecificAttributes(instances, "1-" + (instances.numAttributes()-1), true);
		
		String[] options = new String[2];
		options[0] = "-I";                 // max. iterations
		options[1] = "100";
		SimpleKMeans kMeans = new SimpleKMeans();  // new instance of clusterer
		try {
			kMeans.setNumClusters(2);
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
			boolean[] clusterLabels = new boolean[centroids.numInstances()];
			ArrayList<ArrayList<Double>> mediansInClusters = new ArrayList<ArrayList<Double>>();
			for (int i = 0; i < kMeans.numberOfClusters(); i++) { 
				//System.out.println( instances.instance(i) + " is in cluster " + (kMeans.clusterInstance(instances.instance(i)) + 1)); 
				mediansInClusters.add(getClusterLabelByMedianMetricValues(i,instances,assignment,positiveLabel));
				
			}
			
			//compare mediansInClusters.get(0) and mediansInClusters.get(1)
			int winCountForTheFirstCluster = 0;
			for(int attrIdx = 0; attrIdx< instances.numAttributes(); attrIdx++){
				if(attrIdx==instances.classIndex())
					continue;
				
				if(mediansInClusters.get(0).get(attrIdx)>mediansInClusters.get(1).get(attrIdx))
					winCountForTheFirstCluster++;
			}
			
			System.out.println(winCountForTheFirstCluster);
			System.out.println((instances.numAttributes()-1)/2.0);
			
			if(winCountForTheFirstCluster>(instances.numAttributes()-1)/2.0){
				clusterLabels[0] = true;
				clusterLabels[1] = false;
			}
			else{
				clusterLabels[0] = false;
				clusterLabels[1] = true;
			}
			
			System.out.println(clusterLabels[0]);
			
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
			
			double recall = (double) TP / (TP + FN);
			double precision =(double) TP / (TP + FP);
			double fmeasure = (2*recall*precision)/(recall+precision);
			double accuracy = (double) (TP+TN)/(TP+TN+FP+FN);
			
			System.out.println(groupName+","+projectName+"," + precision+"," + recall+"," + fmeasure + "," +accuracy);
			System.out.println(TP + " " + TN + " " + FP + " " +FN);

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}     // set the options

	}

	private ArrayList<Double> getClusterLabelByMedianMetricValues(int i, Instances instances, int[] assignment, String posStrLabel) {
		ArrayList<Integer> instIdxInThisCLuster = new ArrayList<Integer>();
		ArrayList<Double> medians = new ArrayList<Double>();
		
		// find instances in the cluster i.
		for(int instIdx = 0; instIdx <assignment.length;instIdx++){
			if(assignment[instIdx]==i)
				instIdxInThisCLuster.add(instIdx);
		}
		
		// compute medians
		Instances instancesInCluster = WekaUtils.getInstancesFromIndice(instances, instIdxInThisCLuster);
		
		for(int attrIdx = 0; attrIdx < instancesInCluster.numAttributes(); attrIdx++){
			if(attrIdx==instancesInCluster.classIndex())
				continue;
			
			ArrayList<Double> values = new ArrayList<Double>();
			for(double value:instancesInCluster.attributeToDoubleArray(attrIdx)){
				values.add(value);
			}
			
			medians.add(ArrayListUtil.getMedian(values));
		}
		
		System.out.println("# of instances in cluster " + i + ": " + instIdxInThisCLuster.size() );
		
		return medians;
	}
}
