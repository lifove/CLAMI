package net.lifove.clami.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.commons.math3.stat.StatUtils;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveRange;

public class Utils {
	
	
	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @return instances labeled by CLA
	 */
	public static void getCLAResult(Instances instances,double percentileCutoff,String positiveLabel,boolean forCLAMI) {
		
		System.out.println("\nHigher value cutoff > P" + percentileCutoff );
		
		Instances instancesByCLA = getInstancesByCLA(instances, percentileCutoff, positiveLabel);
		
		// Print CLA results
		for(int instIdx = 0; instIdx < instancesByCLA.numInstances(); instIdx++){
			System.out.println("CLA: Instance " + (instIdx+1) + " predicted as, " + Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx) +
						", (Actual class: " + Utils.getStringValueOfInstanceLabel(instances,instIdx) + ") ");
		}
	}

	private static Instances getInstancesByCLA(Instances instances, double percentileCutoff, String positiveLabel) {
		Instances instancesByCLA = new Instances(instances);
		
		double[] cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(instances, percentileCutoff);
		
		// compute, K = the number of metrics whose values are greater than median, for each instance
		Double[] K = new Double[instances.numInstances()];
		
		for(int instIdx = 0; instIdx < instances.numInstances();instIdx++){
			K[instIdx]=0.0;
			for(int attrIdx = 0; attrIdx < instances.numAttributes();attrIdx++){
				if (attrIdx == instances.classIndex())
					continue;
				
				if(instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx]){
					K[instIdx]++;
				}
			}
		}
		
		// compute cutoff for the top half and bottom half clusters
		double cutoffOfKForTopClusters = Utils.getMedian(new ArrayList<Double>(new HashSet<Double>(Arrays.asList(K))));
		
		for(int instIdx = 0; instIdx < instances.numInstances(); instIdx++){
			if(K[instIdx]>=cutoffOfKForTopClusters)
				instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
			else
				instancesByCLA.instance(instIdx).setClassValue(getNegLabel(instancesByCLA,positiveLabel));
		}
		return instancesByCLA;
	}

	/**
	 * Get higher value cutoffs for each attribute
	 * @param instances
	 * @param percentileCutoff
	 * @return
	 */
	private static double[] getHigherValueCutoffs(Instances instances, double percentileCutoff) {
		// compute median values for attributes
		double[] cutoffForHigherValuesOfAttribute = new double[instances.numAttributes()];

		for(int attrIdx=0; attrIdx < instances.numAttributes();attrIdx++){
			if (attrIdx == instances.classIndex())
				continue;
			cutoffForHigherValuesOfAttribute[attrIdx] = StatUtils.percentile(instances.attributeToDoubleArray(attrIdx),percentileCutoff);
		}
		return cutoffForHigherValuesOfAttribute;
	}
	
	/**
	 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get instancesByCLA use getCLAResult.
	 * @param testInstances
	 * @param instancesByCLA
	 * @param positiveLabel
	 */
	public static void getCLAMIResult(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff) {
		
		String mlAlgorithm = "weka.classifiers.functions.Logistic";
		
		Instances instancesByCLA = getInstancesByCLA(instances, percentileCutoff, positiveLabel);
		
		// Compute medians
		double[] cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(instancesByCLA,percentileCutoff);
				
		// Metric selection
		
		// (1) get distinct violation scores ordered by ASC
		HashMap<Integer,String> metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(instancesByCLA,cutoffsForHigherValuesOfAttribute,positiveLabel);
		Object[] keys = metricIdxWithTheSameViolationScores.keySet().toArray();
		Arrays.sort(keys);
		
		Instances trainingInstancesByCLAMI = null;
		
		// (2) Generate instances for CLAMI. If there are no instances in the first round with the minimum violation scores,
		//     then use the next minimum violation score. (Keys are ordered violation scores)
		Instances newTestInstances = null;
		for(Object key: keys){
			
			String selectedMetricIndices = metricIdxWithTheSameViolationScores.get(key) + (instancesByCLA.classIndex() +1);
			trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(instancesByCLA,selectedMetricIndices,true);
			newTestInstances = getInstancesByRemovingSpecificAttributes(testInstances,selectedMetricIndices,true);
					
			// Instance selection
			String instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,cutoffsForHigherValuesOfAttribute,positiveLabel);
			trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,instIndicesNeedToRemove,false);
			
			if(trainingInstancesByCLAMI.numInstances() != 0)
				break;
		}
		
		// check if there are no instances in any one of two classes.
		if(trainingInstancesByCLAMI.attributeStats(trainingInstancesByCLAMI.classIndex()).nominalCounts[0]!=0 &&
				trainingInstancesByCLAMI.attributeStats(trainingInstancesByCLAMI.classIndex()).nominalCounts[1]!=0){
		
			try {
				Classifier classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
				classifier.buildClassifier(trainingInstancesByCLAMI);
				
				for(int instIdx = 0; instIdx < newTestInstances.numInstances(); instIdx++){
					double predictedLabelIdx = classifier.classifyInstance(newTestInstances.get(instIdx));
					System.out.println("CLAMI: Instance " + (instIdx+1) + " predicted as, " + 
							((newTestInstances.classAttribute().indexOfValue(positiveLabel))==predictedLabelIdx?"buggy":"clean") +
							", (Actual class: " + Utils.getStringValueOfInstanceLabel(newTestInstances,instIdx) + ") ");
				}
				
			} catch (Exception e) {
				e.printStackTrace();
			}
		}else{
			System.err.println("Dataset is not proper to build a CLAMI model! Dataset does not follow the assumption, i.e. the higher metric value, the more bug-prone.");
		}
	}

	private static HashMap<Integer, String> getMetricIndicesWithTheViolationScores(Instances instances,
			double[] cutoffsForHigherValuesOfAttribute, String positiveLabel) {

		int[] violations = new int[instances.numAttributes()];
		
		for(int attrIdx=0; attrIdx < instances.numAttributes(); attrIdx++){
			if(attrIdx == instances.classIndex()){
				violations[attrIdx] = instances.numInstances(); // make this as max to ignore since our concern is minimum violation.
				continue;
			}
			
			for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){
				if (instances.get(instIdx).value(attrIdx) <= cutoffsForHigherValuesOfAttribute[attrIdx]
						&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(positiveLabel)){
						violations[attrIdx]++;
				}else if(instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx]
						&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(getNegLabel(instances, positiveLabel))){
						violations[attrIdx]++;
				}
			}
		}
		
		HashMap<Integer,String> metricIndicesWithTheSameViolationScores = new HashMap<Integer,String>();
		
		for(int attrIdx=0; attrIdx < instances.numAttributes(); attrIdx++){
			if(attrIdx == instances.classIndex()){
				continue;
			}
			
			int key = violations[attrIdx];
			
			if(!metricIndicesWithTheSameViolationScores.containsKey(key)){
				metricIndicesWithTheSameViolationScores.put(key,(attrIdx+1) + ",");
			}else{
				String indices = metricIndicesWithTheSameViolationScores.get(key) + (attrIdx+1) + ",";
				metricIndicesWithTheSameViolationScores.put(key,indices);
			}
		}
		
		return metricIndicesWithTheSameViolationScores;
	}

	private static String getSelectedInstances(Instances instances, double[] cutoffsForHigherValuesOfAttribute,
			String positiveLabel) {
		
		int[] violations = new int[instances.numInstances()];
		
		for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){
			
			for(int attrIdx=0; attrIdx < instances.numAttributes(); attrIdx++){
				if(attrIdx == instances.classIndex())
					continue; // no need to compute violation score for the class attribute
				
				if (instances.get(instIdx).value(attrIdx) <= cutoffsForHigherValuesOfAttribute[attrIdx]
						&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(positiveLabel)){
						violations[instIdx]++;
				}else if(instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx]
						&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(getNegLabel(instances, positiveLabel))){
						violations[instIdx]++;
				}
			}
		}
		
		String selectedInstances = "";
		
		for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){
			if(violations[instIdx]>0)
				selectedInstances += (instIdx+1) + ","; // let the start attribute index be 1 
		}
		
		return selectedInstances;
	}
	
	/**
	 * Get the negative label string value from the positive label value
	 * @param instances
	 * @param positiveLabel
	 * @return
	 */
	static public String getNegLabel(Instances instances, String positiveLabel){
		if(instances.classAttribute().numValues()==2){
			int posIndex = instances.classAttribute().indexOfValue(positiveLabel);
			if(posIndex==0)
				return instances.classAttribute().value(1);
			else
				return instances.classAttribute().value(0);
		}
		else{
			System.err.println("Class labels must be binary");
			System.exit(0);
		}
		return null;
	}
	
	/**
	 * Load Instances from arff file. Last attribute will be set as class attribute
	 * @param path arff file path
	 * @return Instances
	 */
	public static Instances loadArff(String path,String classAttributeName){
		Instances instances=null;
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(path));
			instances = new Instances(reader);
			reader.close();
			instances.setClassIndex(instances.attribute(classAttributeName).index());
		} catch (NullPointerException e) {
			System.err.println("Class label name, " + classAttributeName + ", does not exist! Please, check if the label name is correct.");
			instances = null;
		} catch (FileNotFoundException e) {
			System.err.println("Data file does not exist. Please, check the path again!");
		} catch (IOException e) {
			System.err.println("I/O error! Please, try again!");
		}

		return instances;
	}
	
	/**
	 * Get label value of an instance
	 * @param instances
	 * @param instance index
	 * @return string label of an instance
	 */
	static public String getStringValueOfInstanceLabel(Instances instances,int intanceIndex){
		return instances.instance(intanceIndex).stringValue(instances.classIndex());
	}
	
	/**
	 * Get median from ArraList<Double>
	 * @param values
	 * @return
	 */
	static public double getMedian(ArrayList<Double> values){
		return getPercentile(values,50);
	}
	
	/**
	 * Get a value in a specific percentile from ArraList<Double>
	 * @param values
	 * @return
	 */
	static public double getPercentile(ArrayList<Double> values,double percentile){
		return StatUtils.percentile(getDoublePrimitive(values),percentile);
	}
	
	/**
	 * Get primitive double form ArrayList<Double>
	 * @param values
	 * @return
	 */
	public static double[] getDoublePrimitive(ArrayList<Double> values) {
		return Doubles.toArray(values);
	}
	
	/**
	 * Get instances by removing specific attributes
	 * @param instances
	 * @param attributeIndices attribute indices (e.g., 1,3,4) first index is 1
	 * @param invertSelection for invert selection, if true, select attributes with attributeIndices bug if false, remote attributes with attributeIndices
	 * @return new instances with specific attributes
	 */
	static public Instances getInstancesByRemovingSpecificAttributes(Instances instances,String attributeIndices,boolean invertSelection){
		Instances newInstances = new Instances(instances);

		Remove remove;

		remove = new Remove();
		remove.setAttributeIndices(attributeIndices);
		remove.setInvertSelection(invertSelection);
		try {
			remove.setInputFormat(newInstances);
			newInstances = Filter.useFilter(newInstances, remove);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}

		return newInstances;
	}
	
	/**
	 * Get instances by removing specific instances
	 * @param instances
	 * @param instance indices (e.g., 1,3,4) first index is 1
	 * @param option for invert selection
	 * @return selected instances
	 */
	static public Instances getInstancesByRemovingSpecificInstances(Instances instances,String instanceIndices,boolean invertSelection){
		Instances newInstances = null;

		RemoveRange instFilter = new RemoveRange();
		instFilter.setInstancesIndices(instanceIndices);
		instFilter.setInvertSelection(invertSelection);

		try {
			instFilter.setInputFormat(instances);
			newInstances = Filter.useFilter(instances, instFilter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newInstances;
	}
}
