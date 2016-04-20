package hk.ust.cse.ipam.jc.selftrainer;

import hk.ust.cse.ipam.utils.Measure;
import hk.ust.cse.ipam.utils.WekaUtils;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

import weka.core.Instances;

public class ThresholdBased {

	public static void main(String[] args) {
		new ThresholdBased().run(args);

	}
	
	void run(String[] args) {
		String dataPath = args[0];
		String groupName = args[1];
		String projectName = args[2];
		String classAttributeName = args[3];
		String positiveLabel = args[4];
		
		// load arff file
		Instances instances = WekaUtils.loadArff(dataPath, classAttributeName);
		Measure measure = thdModel(positiveLabel, instances);
		
		double recall = measure.getRecall();
		double precision = measure.getPrecision();
		double fmeasure = measure.getFmeasure();
		double TP = measure.getTP();
		double FP = measure.getFP();
		double TN = measure.getTN();
		double FN = measure.getFN();
		
		/*System.out.println("Recall: " + recall);
		System.out.println("Precision: " + precision);
		System.out.println("Fmeasure: " + fmeasure);
		System.out.println("FPR: " + falsePositveRate);*/
		System.out.println(groupName+","+projectName+"," + precision+"," + recall+"," + fmeasure + "," + TP + "," + FP +"," + TN + "," + FN);
	}

	public Measure thdModel(String positiveLabel, Instances instances) {
		ArrayList<Threshold> attrThresholds = getAttrThresholds(instances,positiveLabel);
		
		Measure measure = predicionUsingAttrThresholds(instances,attrThresholds,positiveLabel);
		return measure;
	}

	private Measure predicionUsingAttrThresholds(Instances instances,
			ArrayList<Threshold> attrThresholds, String positiveLabel) {
		
		int TP=0,FP=0,FN=0,TN=0;
		
		for(int instIdx=0;instIdx<instances.numInstances();instIdx++){
			double buggyLabelIndex = WekaUtils.getClassValueIndex(instances, positiveLabel);
			double originalLabel = instances.instance(instIdx).value(instances.classIndex());
			for(int attrIdx=0;attrIdx<instances.numAttributes();attrIdx++){
				if(attrIdx==instances.classIndex())
					continue;
				
				// consider direction
				if(attrThresholds.get(attrIdx).isPositiveDirection()){
					// if there is any attribute greater than a threshold, the instances is predicted as buggy and stop this loop.
					if(instances.instance(instIdx).value(attrIdx)>attrThresholds.get(attrIdx).getThreshold()){
						// if coming in this block, the instance is predicted as buggy
						
						// judge TP or FP
						if(originalLabel==buggyLabelIndex) TP++; else FP++;
						
						break;
					}
				}
				else{	// negative direction
					// if there is any attribute less than a threshold, the instances is predicted as buggy and stop this loop.
					if(instances.instance(instIdx).value(attrIdx)<attrThresholds.get(attrIdx).getThreshold()){
						// if coming in this block, the instance is predicted as buggy
						
						// judge TP or FP
						if(originalLabel==buggyLabelIndex) TP++; else FP++;
						
						break;
					}		
				}
			
				// if reaching here, the instance is predicted as clean.
				// judge TN or FN
				if(originalLabel!=buggyLabelIndex) TN++; else FN++;
					break;
			}
		}
		
		Measure measure = new Measure();
		
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

	private ArrayList<Threshold> getAttrThresholds(Instances instances,String positiveLabel) {
		ArrayList<Threshold> attrThresholds = new ArrayList<Threshold>();
		
		for(int attrIdx=0; attrIdx < instances.numAttributes();attrIdx++){
			//skip class attribute
			if(attrIdx==instances.classIndex()){
				attrThresholds.add(new Threshold(-1,true)); // for the class attribute, just add default values to keep attribute index order on ArrayList<Threshold>
				continue;
			}
			
			// decide direction
			double correlation = new PearsonsCorrelation().correlation(instances.attributeToDoubleArray(attrIdx), instances.attributeToDoubleArray(instances.classIndex()));
			boolean positiveDirection = true;
			if(WekaUtils.getClassValueIndex(instances, positiveLabel)==0){
				if(correlation > 0)
					positiveDirection = false;
			}
			if(WekaUtils.getClassValueIndex(instances, positiveLabel)==1){
				if(correlation < 0)
					positiveDirection = false;
			}
			
			// compute threshold
			double value = getThreshold(instances,attrIdx,positiveDirection,positiveLabel);
			
			//System.out.println((attrIdx+1) + " " + value);
			
			// add in attrThresholds
			attrThresholds.add(new Threshold(value,positiveDirection));
		}
		
		return attrThresholds;
	}

	/**
	 * Compute threshold by Tuning Machine technique
	 * @param instances
	 * @param attrIdx
	 * @param positiveDirection
	 * @param positiveLabel
	 * @return
	 */
	private double getThreshold(Instances instances,int attrIdx, boolean positiveDirection, String positiveLabel) {
		
		HashMap<Double,Double> precisionForEachThreshold = new HashMap<Double,Double>(); // key: a certain attribute value (threshld), value: prcision=#correctly predicted as buggy/# all predicted as buggy
		
		for(int instIdx=0;instIdx<instances.numInstances();instIdx++){
			double threshold = instances.instance(instIdx).value(attrIdx);
			double precision = computePrecision(instances,threshold,instances.attributeToDoubleArray(attrIdx),positiveDirection,positiveLabel);
			precisionForEachThreshold.put(threshold, precision);
		}
		
		Double finalThreshold = Double.NaN;
		Double maxPrecision = Double.NaN;
		for(double key:precisionForEachThreshold.keySet()){
			if(maxPrecision.equals(Double.NaN) || maxPrecision<precisionForEachThreshold.get(key) ){
				maxPrecision = precisionForEachThreshold.get(key);
				finalThreshold = key;
			}
		}
		
		//System.out.println((attrIdx+1) + " precision: " + maxPrecision);
		return finalThreshold;
	}

	/**
	 * Compute maximum precision that Tuning Machine technique identifies
	 * @param instances
	 * @param threshold
	 * @param attributeValues
	 * @param positiveDirection
	 * @param positiveLabel
	 * @return
	 */
	private double computePrecision(Instances instances, double threshold,
			double[] attributeValues, boolean positiveDirection, String positiveLabel) {
		
		int numPredictedAsBuggy = 0;
		int numCorrectlyPredictedAsBuggy = 0;
		
		for(int instIdx=0;instIdx<instances.numInstances();instIdx++){
			if(positiveDirection){
				if(attributeValues[instIdx]>threshold){
					numPredictedAsBuggy++;
					// if both the predicted and original labels are same as buggy, it is correct prediction
					if(instances.instance(instIdx).value(instances.classIndex())==WekaUtils.getClassValueIndex(instances, positiveLabel))
						numCorrectlyPredictedAsBuggy++;
				}		
			}
			else{
				if(attributeValues[instIdx]<threshold){
					numPredictedAsBuggy++;
					// if both the predicted and original labels are same as buggy, it is correct prediction
					if(instances.instance(instIdx).value(instances.classIndex())==WekaUtils.getClassValueIndex(instances, positiveLabel))
						numCorrectlyPredictedAsBuggy++;
				}	
			}
		}
		
		return (double)numCorrectlyPredictedAsBuggy/numPredictedAsBuggy;
	}
}

class Threshold{
	public Threshold(double value, boolean direction) {
		threshold = value;
		positiveDirection = direction;
	}
	public double getThreshold() {
		return threshold;
	}
	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}
	public boolean isPositiveDirection() {
		return positiveDirection;
	}
	public void setPositiveDirection(boolean positiveDirection) {
		this.positiveDirection = positiveDirection;
	}
	double threshold;
	boolean positiveDirection; // if true, higher value than threshold is predicted as buggy.
}
