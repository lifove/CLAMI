package hk.ust.cse.ipam.jc.selftrainer;

import static org.junit.Assert.*;

import org.junit.Test;

public class ClusteringAndExpertBasedTest {

	@Test
	public void test() {
		
		System.out.println("ClucsteringAndExpertBased");
		System.out.println("Group,Project,Precision,Recall,Fmeasure,Accuracy");
		
		String useBuggyRate = "true";
		String numBuggyClusters = "10";
		
		String[] args={"data/Relink/Apache.arff", "relink", "Apache","isDefective", "TRUE",useBuggyRate,numBuggyClusters};
		ClusteringAndExpertBased runner = new ClusteringAndExpertBased();
		for(int numClusters=1;numClusters<=20;numClusters++){
			
			args[0] = "data/Relink/Apache.arff";
			args[1] = "relink";
			args[2] = "Apache";
			args[3] = "isDefective";
			args[4] = "TRUE";
			
			System.out.print(numClusters +",");
			//runner.run(args);
			
			args[0] = "data/Relink/Safe.arff";
			args[2] = "Safe";
			args[6] = "" + numClusters;//"8";
			System.out.print(numClusters +",");
			runner.run(args);
			
			args[0] = "data/Relink/Zxing.arff";
			args[2] = "Zxing";
			args[6] = "" + numClusters;//"6";
			System.out.print(numClusters +",");
			runner.run(args);
			
			args[0] = "data/gene/httpclient.arff";
			args[1] = "gene";
			args[2] = "httpclient";
			args[3] = "class";
			args[4] = "buggy";
			args[6] = "" + numClusters;//"11";
			System.out.print(numClusters +",");
			runner.run(args);
			
			args[0] = "data/gene/rhino.arff";
			args[2] = "rhino";
			args[6] = "9";
			System.out.print(numClusters +",");
			runner.run(args);
			
			args[0] = "data/gene/jackrabbit.arff";
			args[2] = "jackrabbit";
			args[6] = "" + numClusters;//"8";
			System.out.print(numClusters +",");
			runner.run(args);
			
			args[0] = "data/gene/lucene.arff";
			args[2] = "lucene";
			args[6] = "" + numClusters;//"2";
		
			args[6] = ""+ numClusters;
			//System.out.print(numClusters +",");
			//runner.run(args);
		}
		
		/*
		String[] args={"data/promise/velocity-1.4.arff", "promise", "velocity", "bug", "buggy"};
		ClusteringAndExpertBased runner = new ClusteringAndExpertBased();
		//runner.run(args);
		
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
			args[0] = "data/promise/" + project + ".arff";
			args[2] = project;
			runner.main(args);
		}*/
	}

}
