package net.lifove.clami.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

import org.apache.commons.math3.stat.StatUtils;

import com.google.common.primitives.Doubles;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class Utils {
	
	
	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 */
	public static Instances getCLAResult(Instances instances,double percentileCutoff,String positiveLabel,boolean forCLAMI) {
		
		Instances instancesByCLA = new Instances(instances);
		
		// compute median values for attributes
		double[] mediansForAttributes = new double[instances.numAttributes()];

		for(int attrIdx=0; attrIdx < instances.numAttributes();attrIdx++){
			if (attrIdx == instances.classIndex())
				continue;
			mediansForAttributes[attrIdx] = StatUtils.percentile(instances.attributeToDoubleArray(attrIdx),50);
		}
		
		// compute, K = the number of metrics whose values are greater than median, for each instance
		Double[] K = new Double[instances.numInstances()];
		
		for(int instIdx = 0; instIdx < instances.numInstances();instIdx++){
			K[instIdx]=0.0;
			for(int attrIdx = 0; attrIdx < instances.numAttributes();attrIdx++){
				if (attrIdx == instances.classIndex())
					continue;
				if(instances.get(instIdx).value(attrIdx) > mediansForAttributes[attrIdx]){
					K[instIdx]++;
				}
			}
		}
		
		// compute cutoff for the top and bottom clusters, default = median (50)
		double cutoffOfKForTopClusters = Utils.getPercentile(new ArrayList<Double>(new HashSet<Double>(Arrays.asList(K))),percentileCutoff);
		
		// Predict
		for(int instIdx = 0; instIdx < instances.numInstances(); instIdx++){
			if(!forCLAMI)
				System.out.println("CLA: Instance " + (instIdx+1) + " predicted as, " + (K[instIdx]>=cutoffOfKForTopClusters?"buggy":"clean") +
						", (Actual class: " + Utils.getStringValueOfInstanceLabel(instances,instIdx) + ") ");
			
			if(K[instIdx]>=cutoffOfKForTopClusters)
				instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
			else
				instancesByCLA.instance(instIdx).setClassValue(getNegLabel(instancesByCLA,positiveLabel));
		}
		
		return instancesByCLA;
	}
	
	public static void getCLAMIResult(Instances testInstances, Instances instancesByCLA, String positiveLabel) {
		
		String mlAlgorithm = "weka.classifiers.functions.Logistic";
		
		Instances trainingInstances = getCLAMITrainingInstances(instancesByCLA,positiveLabel);
		
		try {
			Classifier classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
			classifier.buildClassifier(trainingInstances);
			
			for(int instIdx = 0; instIdx < testInstances.numInstances(); instIdx++){
				double predictedLabelIdx = classifier.classifyInstance(testInstances.get(instIdx));
				System.out.println("CLAMI: Instance " + (instIdx+1) + " predicted as, " + 
						((testInstances.classAttribute().indexOfValue(positiveLabel))==predictedLabelIdx?"buggy":"clean") +
						", (Actual class: " + Utils.getStringValueOfInstanceLabel(testInstances,instIdx) + ") ");
			}
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	private static Instances getCLAMITrainingInstances(Instances instancesByCLA, String positiveLabel) {
		return instancesByCLA;
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

}
