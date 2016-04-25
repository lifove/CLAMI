package net.lifove.clami;


import static org.junit.Assert.*;

import org.junit.Test;

public class CLAMITest {

	@Test
	public void testRunner() {
		String[] args = {"-f","test.arff","-l","label","-p","buggy","-m"};
		CLAMI runner = new CLAMI();
		runner.runner(args);
		assert(runner.forCLAMI);
		assertEquals(runner.dataFilePath,args[1]);
		assertEquals(runner.labelName,args[3]);
		assertEquals(runner.posLabelValue,args[5]);
		assertEquals(runner.forCLAMI,true);
		
		args[1] = "";
		runner = new CLAMI();
		assertEquals(runner.dataFilePath,null);
		assertEquals(runner.labelName,null);
		assertEquals(runner.posLabelValue,null);
	}

}
