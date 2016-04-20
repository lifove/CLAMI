package hk.ust.cse.ipam.jc.selftrainer;

import java.util.ArrayList;

import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import org.apache.commons.math3.stat.inference.WilcoxonSignedRankTest;

import hk.ust.cse.ipam.utils.ArrayListUtil;
import hk.ust.cse.ipam.utils.WekaUtils;
import weka.core.Instances;

public class FeatureSelectionBased {

	public static void main(String[] args) {
		new FeatureSelectionBased().run(args);
	}

	private void run(String[] args) {
		String dataPath = args[0];
		String classAttributeName = args[1];
		
		Instances instances = WekaUtils.loadArff(dataPath, classAttributeName);
		
		int[] selectedFeatures = selectSimilarFeatures(instances);
		
		for(double value:selectedFeatures)
			System.out.println(value + "");
		
	}

	static public int[] selectSimilarFeatures(Instances instances) {
		
		//KolmogorovSmirnovTest KStest = new KolmogorovSmirnovTest();
		WilcoxonSignedRankTest statTest = new WilcoxonSignedRankTest();
		ArrayList<Integer> selectedAttrIndice = new ArrayList<Integer>();
		ArrayList<String> processedMatching = new ArrayList<String>();
		
		// find common features where the KS-test p-value is greater than 0.5
		for(int attrIndex=0;attrIndex<instances.numAttributes();attrIndex++){
			if(instances.attribute(attrIndex)==instances.classAttribute())
				continue;
		
			double[] sample1 = instances.attributeToDoubleArray(attrIndex);
			
			for(int attrIndex2=0;attrIndex2<instances.numAttributes();attrIndex2++){
				if(instances.attribute(attrIndex2)==instances.classAttribute() || attrIndex==attrIndex2)
					continue;
				
				double[] sample2 = instances.attributeToDoubleArray(attrIndex2);
				
				if(!(processedMatching.contains(attrIndex+"_" + attrIndex2) || processedMatching.contains(attrIndex2+"_" + attrIndex))){
					double pvalue =  statTest.wilcoxonSignedRankTest(sample1, sample2,false);//KStest.kolmogorovSmirnovTest(sample1, sample2);				
					if(pvalue>=0.9){
						processedMatching.add(attrIndex + "_" + attrIndex2);
					
						if(!selectedAttrIndice.contains(attrIndex))
							selectedAttrIndice.add(attrIndex);
						
						if(!selectedAttrIndice.contains(attrIndex2))
							selectedAttrIndice.add(attrIndex2);
					}
				}
			}
		}
		
		return ArrayListUtil.getIntPromitive(selectedAttrIndice);
	}
}
