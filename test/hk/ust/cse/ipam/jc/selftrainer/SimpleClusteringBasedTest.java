package hk.ust.cse.ipam.jc.selftrainer;

import static org.junit.Assert.*;

import org.junit.Test;

public class SimpleClusteringBasedTest {

	@Test
	public void test() {
		
		System.out.println("SimpleClucsteringBased");
		System.out.println("Group,Project,Precision,Recall,Fmeasure,Accuracy");
		
		String[] args={"data/Relink/Apache.arff", "relink", "Apache","isDefective", "TRUE"};
		SimpleClusteringBased runner = new SimpleClusteringBased();
		runner.run(args);
		
		args[0] = "data/Relink/Safe.arff";
		args[2] = "Safe";
		runner.run(args);
		
		args[0] = "data/Relink/Zxing.arff";
		args[2] = "Zxing";
		runner.run(args);
		
		/*args[0] = "data/gene/httpclient.arff";
		args[1] = "gene";
		args[2] = "httpclient";
		args[3] = "class";
		args[4] = "buggy";
		runner.run(args);
		
		args[0] = "data/gene/rhino.arff";
		args[2] = "rhino";
		runner.run(args);
		
		args[0] = "data/gene/jackrabbit.arff";
		args[2] = "jackrabbit";
		runner.run(args);
		
		args[0] = "data/gene/lucene.arff";
		args[2] = "lucene";
		runner.run(args); */
		
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
