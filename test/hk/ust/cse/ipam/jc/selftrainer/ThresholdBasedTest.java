package hk.ust.cse.ipam.jc.selftrainer;

import static org.junit.Assert.*;

import org.junit.Test;

public class ThresholdBasedTest {

	@Test
	public void test() {
		
		System.out.println("ThresholdBased");
		System.out.println("Group,Project,Precision,Recall,Fmeasure");
		
		String[] args={"data/Relink/Apache.arff", "relink", "Apache","isDefective", "TRUE"};
		ThresholdBased runner = new ThresholdBased();
		//runner.run(args);
		
		/*args[0] = "data/Relink/Safe.arff";
		args[2] = "Safe";
		runner.run(args);
		
		args[0] = "data/Relink/Zxing.arff";
		args[2] = "Zxing";
		runner.run(args);*/
		
		args[0] = "data/gene/httpclient.arff";
		args[1] = "gene";
		args[2] = "httpclient";
		args[3] = "class";
		args[4] = "buggy";
		//runner.run(args);
		
		args[0] = "data/gene/rhino.arff";
		args[2] = "rhino";
		runner.run(args);
		
		args[0] = "data/gene/jackrabbit.arff";
		args[2] = "jackrabbit";
		//runner.run(args);
		
		args[0] = "data/gene/lucene.arff";
		args[2] = "lucene";
		//runner.run(args);*/
		
	}

}
