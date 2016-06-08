package net.lifove.clami;


import static org.junit.Assert.*;

import org.junit.Test;

public class CLAMITest2 {

	@Test
	public void testRunner() {
		
		String[] args = {"-f","../../Documents/CDDP/CDDP/data/gene/httpclient.arff","-l","class","-p","buggy","","--suppress"};
		CLAMI runner = new CLAMI();
		runner.runner(args);
		
		args[1] = "../../Documents/CDDP/CDDP/data/gene/jackrabbit.arff";
		runner.runner(args);
		
		args[1] = "../../Documents/CDDP/CDDP/data/gene/lucene.arff";
		runner.runner(args);
		
		args[1] = "../../Documents/CDDP/CDDP/data/gene/rhino.arff";
		runner.runner(args);
		
		args[1] = "../../Documents/CDDP/CDDP/data/Relink/Apache.arff";
		args[3] = "isDefective";
		args[5] = "TRUE";
		runner.runner(args);
		
		args[1] = "../../Documents/CDDP/CDDP/data/Relink/Apache.arff";
		runner.runner(args);
		
		args[1] = "../../Documents/CDDP/CDDP/data/Relink/Safe.arff";
		runner.runner(args);
		
		args[1] = "../../Documents/CDDP/CDDP/data/Relink/Zxing.arff";
		runner.runner(args);
		
		assertEquals(runner.posLabelValue,"TRUE");
	}

}
