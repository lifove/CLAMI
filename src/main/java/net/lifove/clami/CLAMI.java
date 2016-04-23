package net.lifove.clami;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

/**
 * CLAMI implementation:
 * CLAMI: Defect Prediction on Unlabeled Datasets, in Proceedings of the 30th IEEE/ACM International Conference on Automated Software Engineering (ASE 2015), Lincoln, Nebraska, USA, November 9 - 13, 2015
 * 
 * @author JC
 *
 */
public class CLAMI {
	
	String sourcePath;
	String sourceLabelName;
	String sourceLabelPos;

	public static void main(String[] args) {
		
		new CLAMI().runner(args);
		
	}

	private void runner(String[] args) {
		
		Options options = createOptions();
		
		if(args.length < 3){

			// automatically generate the help statement
			HelpFormatter formatter = new HelpFormatter();
			String header = "Execute CLA/CLAMI unsuprvised defect predicition. On Windows, use CLAMI.bat instead of ./CLAMI";
			String footer ="\nPlease report issues at https://github.com/lifove/CLAMI/issues";
			formatter.printHelp( "./CLAMI", header, options, footer, true);
		}
		else{
			parseOptions(options, args);
		}
		
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
		options.addOption(Option.builder("l").longOpt("lable")
		        .desc("Label (Class attrubite) name")
		        .hasArg()
		        .argName("attribute name")
		        .required()
		        .build());
		options.addOption(Option.builder("p").longOpt("poslabel")
		        .desc("String value of buggy label")
		        .hasArg()
		        .argName("attribute value")
		        .required()
		        .build());
		options.addOption(Option.builder("m").longOpt("clami")
		        .desc("Run CLAMI instead of CLA")
		        .build());

		return options;

	}
	
	void parseOptions(Options options,String[] args){

		CommandLineParser parser = new DefaultParser();

		try {

			CommandLine cmd = parser.parse(options, args);

			sourcePath = cmd.getOptionValue("sourcefile");
			sourceLabelName = cmd.getOptionValue("srclabelname");
			sourceLabelPos = cmd.getOptionValue("possrclabel");

		} catch (ParseException e) {
			e.printStackTrace();
		}
	}
}
