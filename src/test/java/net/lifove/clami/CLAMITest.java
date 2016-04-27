package net.lifove.clami;


import static org.junit.Assert.*;

import org.junit.Test;

public class CLAMITest {

	@Test
	public void testRunner() {
		
		CLAMI runner = new CLAMI();
		String [] arg={"-h"};
		runner.runner(arg);
		
		String[] args = {"-f","data/sample.arff","-l","class","-p","buggy","-m","-c","50"};
		runner = new CLAMI();
		runner.runner(args);
		assert(runner.forCLAMI);
		assertEquals(runner.dataFilePath,args[1]);
		assertEquals(runner.labelName,args[3]);
		assertEquals(runner.posLabelValue,args[5]);
		assertEquals(runner.forCLAMI,true);
		
		args[8] = "60";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "40";
		runner = new CLAMI();
		runner.runner(args);
		
		args[3] = "label";
		runner = new CLAMI();
		runner.runner(args);
		
		args[1] = "";
		runner = new CLAMI();
		runner.runner(args);
		assertEquals(runner.dataFilePath,"");
		assertEquals(runner.labelName,"label");
		assertEquals(runner.posLabelValue,"buggy");
	}

}
