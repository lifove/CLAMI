package net.lifove.clami;


import static org.junit.Assert.*;

import org.junit.Test;

public class CLAMITest {

	@Test
	public void testRunner() {
		
		CLAMI runner = new CLAMI();
		String [] arg={"-k"};
		runner.runner(arg);
		
		String [] arg2={"-h"};
		runner.runner(arg2);
		
		String[] args = {"-f","data/sample.arff","-l","class","-p","buggy","-m","-c","50"};
		runner = new CLAMI();
		runner.runner(args);
		
		assert(runner.forCLAMI);
		assertEquals(runner.dataFilePath,args[1]);
		assertEquals(runner.labelName,args[3]);
		assertEquals(runner.posLabelValue,args[5]);
		assertEquals(runner.forCLAMI,true);
		
		args[6] = "";
		
		args[8] = "90";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "88";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "87";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "86";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "85";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "84";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "83";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "80";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "70";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "60";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "57";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "55";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "52";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "50";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "40";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "30";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "20";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "10";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "-1";
		runner = new CLAMI();
		runner.runner(args);
		
		args[8] = "50";
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
