package net.lifove.clami;

import java.io.File;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

import net.lifove.clami.util.Utils;
import weka.core.Instances;

/**
 * CLAMI implementation:
 * CLAMI: Defect Prediction on Unlabeled Datasets, in Proceedings of the 30th IEEE/ACM International Conference on Automated Software Engineering (ASE 2015), Lincoln, Nebraska, USA, November 9 - 13, 2015
 * 
 * @author JC
 *
 */
public class CLAMI {
	
	String dataFilePath;
	String labelName;
	String posLabelValue;
	double percentileCutoff = 50;
	boolean forCLAMI = false;
	boolean help = false;
	boolean suppress = false;
	String experimental;

	public static void main(String[] args) {
		
		new CLAMI().runner(args);
		
	}

	void runner(String[] args) {
		
		Options options = createOptions();
		
		if(parseOptions(options, args)){
			if (help){
				printHelp(options);
				return;
			}
			
			// exit when percentile range is not correct (it should be 0 < range <= 100)
			if (percentileCutoff <=0 || 100 < percentileCutoff){
				System.err.println("Cutoff percentile must be 0 < and <=100");
				return;
			}
			
			// exit experimental option format is not correct
			if(experimental!=null && !checkExperimentalOption(experimental)){
				System.err.println("Experimental option format is incorrect. Option format: [# of folds]:[# of repetition]. "
						+ "E.g, -e 2:500 (Two-fold cross validation 500 repetition");
				return;
			}
			
			// load an arff file
			Instances instances = Utils.loadArff(dataFilePath, labelName);
			
			if (instances !=null){
				double unit = (double) 100/(instances.numInstances());
				//double unitFloor = Math.floor(unit);
				double unitCeil = Math.ceil(unit);
				
				// TODO need to check how median is computed
				if (unit >= 1 && 100-unitCeil < percentileCutoff){
					System.err.println("Cutoff percentile must be 0 < and <=" + (100-unitCeil));
					return;
				}
				
				if (experimental==null || experimental.equals("")){
					// do prediction
					prediction(instances,posLabelValue,false);
				}else{
					experiment(instances,posLabelValue);
				}
			}
		}
	}
	
	private boolean checkExperimentalOption(String expOpt) {	
		Pattern pattern=Pattern.compile("^[0-9]+:[0-9]");
		Matcher m = pattern.matcher(expOpt);
		return m.find();
	}

	private void experiment(Instances instances, String posLabelValue) {
		
		String[] splitOptions = experimental.split(":");
		int folds = Integer.parseInt(splitOptions[0]);
		int numRuns = Integer.parseInt(splitOptions[1]);
		
		String source = dataFilePath.substring(dataFilePath.lastIndexOf(File.separator)+1).replace(".arff", "");
		
		for(int repeat=0;repeat < numRuns;repeat++){
			
			// randomize with different seed for each iteration
			instances.randomize(new Random(repeat)); 
			instances.stratify(folds);
			
			for(int fold = 0; fold < folds; fold++){
				System.out.print(repeat + "," +fold + "," + source + ",");
				Instances targetInstances = instances.testCV(folds, fold);
				prediction(targetInstances,posLabelValue,true);
				System.out.println();
			}
		}
	}

	void prediction(Instances instances,String positiveLabel,boolean isExperimental){
		
		if(!forCLAMI)
			Utils.getCLAResult(instances, percentileCutoff,positiveLabel,suppress,isExperimental);
		else
			Utils.getCLAMIResult(instances,instances,positiveLabel,percentileCutoff,suppress,isExperimental);
			
			
	}

	private void printHelp(Options options) {
		// automatically generate the help statement
		HelpFormatter formatter = new HelpFormatter();
		String header = "Execute CLA/CLAMI unsuprvised defect predicition. On Windows, use CLAMI.bat instead of ./CLAMI";
		String footer ="\nPlease report issues at https://github.com/lifove/CLAMI/issues";
		formatter.printHelp( "./CLAMI", header, options, footer, true);
	}
	
	Options createOptions(){
		
		// create Options object
		Options options = new Options();
		
		// add options
		options.addOption(Option.builder("f").longOpt("file")
		        .desc("Arff file path to predict defects")
		        .hasArg()
		        .argName("file")
		        .required()
		        .build());
		
		options.addOption(Option.builder("h").longOpt("help")
		        .desc("Help")
		        .build());
		
		options.addOption(Option.builder("c").longOpt("cutoff")
		        .desc("Cutoff percentile for higher values. Default is median (50).")
		        .hasArg()
		        .argName("cutoff percentile")
		        .build());
		
		options.addOption(Option.builder("s").longOpt("suppress")
		        .desc("Suppress detailed prediction results. Only works when the arff data is labeled.")
		        .build());
		
		options.addOption(Option.builder("l").longOpt("lable")
		        .desc("Label (Class attrubite) name")
		        .hasArg()
		        .argName("attribute name")
		        .required()
		        .build());
		
		options.addOption(Option.builder("p").longOpt("poslabel")
		        .desc("String value of buggy label. Since CLA/CLAMI works for unlabeld data (in case of weka arff files, labeled as '?',"
		        		+ " it is not necessary to use this option. "
		        		+ "However, if the data file is labeled, "
		        		+ "it will show prediction results in terms of precision, recall, and f-measure for evaluation puerpose.")
		        .hasArg()
		        .required()
		        .argName("postive label value")
		        .build());
		
		options.addOption(Option.builder("m").longOpt("clami")
		        .desc("Run CLAMI instead of CLA")
		        .build());
		
		options.addOption(Option.builder("e").longOpt("experimental")
		        .desc("Options for experimenets to compare CLA/CLAMI with other cross-project defect prediction approaches by k-fold cross validation. "
		        		+ "Support k-fold cross validation n times. "
		        		+ "Option format: [# of folds]:[# of repetition]. E.g, -e 2:500 (Two-fold cross validation 500 repetition")
		        .hasArg()
		        .argName("#folds:#repeat")
		        .build());

		return options;

	}
	
	boolean parseOptions(Options options,String[] args){

		CommandLineParser parser = new DefaultParser();

		try {

			CommandLine cmd = parser.parse(options, args);

			dataFilePath = cmd.getOptionValue("f");
			labelName = cmd.getOptionValue("l");
			posLabelValue = cmd.getOptionValue("p");
			if(cmd.getOptionValue("c") != null)
				percentileCutoff = Double.parseDouble(cmd.getOptionValue("c"));
			forCLAMI = cmd.hasOption("m");
			help = cmd.hasOption("h");
			suppress = cmd.hasOption("s");
			experimental = cmd.getOptionValue("e");

		} catch (Exception e) {
			printHelp(options);
			return false;
		}

		return true;
	}
}
