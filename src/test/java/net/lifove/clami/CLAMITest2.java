package net.lifove.clami;


import static org.junit.Assert.*;

import org.junit.Test;

public class CLAMITest2 {

	@Test
	public void testRunner() {
		
		String[] args = {"-f","../../Documents/CDDP/CDDP/data/Relink/Zxing.arff","-l","isDefective","-p","TRUE","","--suppress"};
		CLAMI runner = new CLAMI();
		runner.runner(args);
		
		args[6] = "-m";
		runner.runner(args);
		
		assertEquals(runner.posLabelValue,"TRUE");
	}

}
