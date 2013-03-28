package org.apache.mahout.clustering.streaming.vectorizer;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.List;

/**
 * Creates a sequence file of Text, VectorWritable for a set of documents trying different
 * approaches to TF/IDF scoring.
 * @see org.apache.mahout.clustering.streaming.vectorizer.TFIDFScorer
 */
public class TFIDFVectorizerMain {
  private String inputFile;
  private String outputFileBase;
  private String scorer;

  public static void main(String[] args) throws IOException, ClassNotFoundException, NoSuchMethodException, InvocationTargetException, IllegalAccessException, InstantiationException {
    TFIDFVectorizerMain runner = new TFIDFVectorizerMain();
    if (runner.parseArgs(args)) {
      runner.run();
    }
  }

  public void run() {
    // Create the list of documents to be vectorized.
    List<String> paths = Lists.newArrayList();
    try {
      FileContentsToSequenceFiles.getRecursiveFilePaths(inputFile, paths);

      // Parse arguments and see what scorer to use.
      Function<TFIDFScorer.Tuple, Double> tfIdfScorer;
      if (scorer.equals("const")) {
        tfIdfScorer = new TFIDFScorer.Const();
      } else if (scorer.equals("linear")) {
        tfIdfScorer = new TFIDFScorer.Linear();
      } else if (scorer.equals("log")) {
        tfIdfScorer = new TFIDFScorer.Log();
      } else if (scorer.equals("sqrt")) {
        tfIdfScorer = new TFIDFScorer.Sqrt();
      } else {
        tfIdfScorer = (Function<TFIDFScorer.Tuple, Double>) Class.forName(scorer).getConstructor().newInstance();
      }

      // Vectorize the documents and write them out.
      Configuration conf = new Configuration();
      FileSystem fs = FileSystem.get(conf);
      TFIDFVectorizer vectorizer = new TFIDFVectorizer(tfIdfScorer, true);
      vectorizer.vectorizePaths(paths, new Path(outputFileBase), fs, conf);
    } catch (Exception e) {
      System.out.printf("Failed to run vectorizer\n");
      e.printStackTrace();
    }
  }

  private boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help").withDescription("print this list").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFileOption = builder.withLongName("input")
        .withShortName("i")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
        .withDescription("where to get test data (encoded with tf-idf)")
        .create();

    Option outputFileOption = builder.withLongName("output")
        .withShortName("o")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("the base name of the folder containing the training and test sets; the training set will " +
            "be in a folder called training/ and the test set will be in a folder called test/")
        .create();

    Option scorerOption = builder.withLongName("scorer")
        .withShortName("s")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("scorer").withMaximum(1).create())
        .withDescription("what type of scorer to use; use one of const, linear, log, sqrt or a fully qualified " +
            "class name (see TFIDFScorer)")
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(outputFileOption)
        .withOption(scorerOption)
        .create();

    Parser parser = new Parser();
    parser.setHelpOption(help);
    parser.setHelpTrigger("--help");
    parser.setGroup(normalArgs);
    parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 130));
    CommandLine cmdLine = parser.parseAndHelp(args);

    if (cmdLine == null) {
      return false;
    }

    inputFile = (String) cmdLine.getValue(inputFileOption);
    outputFileBase = (String) cmdLine.getValue(outputFileOption);
    scorer = (String) cmdLine.getValue(scorerOption);
    return true;
  }
}
