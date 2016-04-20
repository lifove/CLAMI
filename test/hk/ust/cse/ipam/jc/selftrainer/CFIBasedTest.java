package hk.ust.cse.ipam.jc.selftrainer;

import static org.junit.Assert.*;

import org.junit.Test;

public class CFIBasedTest {

	@Test
	public void test() {
		String cutType = "P";
		
		String mlAlgorithm = "weka.classifiers.functions.Logistic";
		String verbose = "false";
		//String mlAlgorithm = "weka.classifiers.trees.LMT";
		//String mlAlgorithm = "weka.classifiers.trees.J48";
		//String mlAlgorithm = "weka.classifiers.trees.RandomForest";
		//String mlAlgorithm = "weka.classifiers.bayes.NaiveBayes";
		//String mlAlgorithm = "weka.classifiers.bayes.BayesNet";
		//String mlAlgorithm = "weka.classifiers.functions.SMO";
		
		//weka.classifiers.trees.LMT"
		//algorithms="weka.classifiers.functions.Logistic,weka.classifiers.trees.J48,,,,"
		
		//String[] args={"data/gene/rhino.arff", "gene", "rhino", "class", "buggy", "-1", "-1", "-1", "false", "false", cutType, "true", "false"};
		String[] args={"data/Relink/Apache.arff", "Relink", "Apache", "isDefective", "TRUE", "-1", "-1", "50", "false", "false", cutType, "false", "false",verbose,mlAlgorithm,"Single"};
		
		
		//for(int i=1; i*5 <100;i++){
			
			args[7] = "50";//"" + (i*5);
			
			args[0]="data/Relink/Apache.arff";
			args[1]="Relink";
			args[2]="Apache";
			args[3]="isDefective";
			args[4]="TRUE";
			//CFIBased.main(args);
			//args[0]="data/gene/jackrabbit.arff";
			//args[2]="jackrabbit";
			args[0]="data/Relink/Safe.arff";
			args[2]="Safe";
			//CFIBased.main(args);
			//args[0]="data/gene/httpclient.arff";
			//args[2]="httpclient";
			args[0]="data/Relink/Zxing.arff";
			args[2]="Zxing";
			//CFIBased.main(args);
			
			args[0]="data/gene/jackrabbit.arff";
			args[1]="gene";
			args[2]="jackrabbit";
			args[3]="class";
			args[4]="buggy";
			//args[0]="data/relink/Safe.arff";
			//args[2]="Safe";
			//CFIBased.main(args);
			args[0]="data/gene/httpclient.arff";
			args[2]="httpclient";
			//args[0]="data/relink/Zxing.arff";
			//args[2]="Zxing";
			//CFIBased.main(args);
			args[0]="data/gene/rhino.arff";
			args[2]="rhino";
			//args[0]="data/relink/Zxing.arff";
			//args[2]="Zxing";
			CFIBased.main(args);
			args[0]="data/gene/lucene.arff";
			args[2]="lucene";
			args[3]="class";
			args[4]="buggy";
			
			//CFIBased.main(args);
			
		//}
		//args[0]="data/relink/Zxing.arff";
		//args[2]="Zxing";
		//CFIBased.main(args);
		
		/*String[] args={"../JCTools/data/promise/velocity-1.4.arff", "promise", "velocity", "bug", "buggy", "-1", "-1", "-1", "false", "false", cutType, "true", "false"};
		
		
		String[] projects={"ant-1.3",
				"arc",
				"camel-1.0",
				"poi-1.5",
				"redaktor",
				"skarbonka",
				"tomcat",
				"velocity-1.4",
				"xalan-2.4",
				"xerces-1.2"};
		
		for(String project:projects){
			args[0] = "../JCTools/data/promise/" + project + ".arff";
			args[2] = project;
			CFIBased.main(args);
		}*/
	}

}
