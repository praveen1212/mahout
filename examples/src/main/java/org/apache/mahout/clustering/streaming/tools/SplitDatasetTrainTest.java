package org.apache.mahout.clustering.streaming.tools;

import com.google.common.base.Preconditions;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.commons.io.Charsets;
import org.apache.commons.lang.math.RandomUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

public class SplitDatasetTrainTest {
  private String inputFile;
  private String outputFileBase;
  private double testPercent;
  private boolean useKeys;

  private void run(PrintWriter printWriter) throws IOException {
    Configuration conf = new Configuration();
    SequenceFileDirIterable<Text, VectorWritable> inputIterable = new
        SequenceFileDirIterable<Text, VectorWritable>(new Path(inputFile), PathType.LIST, conf);
    FileSystem fs = FileSystem.get(conf);

    if (useKeys) {
      printWriter.printf("Using keys for train/test split\n");
    } else {
      printWriter.printf("Using random testPercent [%f] for train/test split\n", testPercent);
    }

    SequenceFile.Writer trainWriter = SequenceFile.createWriter(fs, conf,
        new Path(outputFileBase + "-train"), Text.class, VectorWritable.class);
    SequenceFile.Writer testWriter = SequenceFile.createWriter(fs, conf,
        new Path(outputFileBase + "-test"), Text.class, VectorWritable.class);

    int numTest = 0;
    int numTrain = 0;
    Text writerText = new Text();
    VectorWritable writerVector = new VectorWritable();
    for (Pair<Text, VectorWritable> item : inputIterable) {
      boolean addToTest;
      if (useKeys) {
        addToTest = item.getFirst().toString().contains("test");
      } else {
        addToTest = (RandomUtils.nextDouble() < testPercent);
      }
      writerText.set(item.getFirst());
      writerVector.set(item.getSecond().get());
      if (addToTest) {
        ++numTest;
        testWriter.append(writerText, writerVector);
      } else {
        trainWriter.append(writerText, writerVector);
        ++numTrain;
      }
    }
    trainWriter.close();
    testWriter.close();

    printWriter.printf("Finished with %d training vector and %d test vectors\n", numTrain, numTest);
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

    Option testPercentOption = builder.withLongName("testPercent")
        .withShortName("te")
        .withArgument(argumentBuilder.withName("testPercent").withMaximum(1).create())
        .withDescription("if set, will do a random split of the data into a training and test set and te% of the " +
            "data will be part of the test set")
        .create();

    Option useKeysOption = builder.withLongName("useKeys")
        .withShortName("k")
        .withDescription("whether to use the keys from the sequence file to decide which vectors are used for " +
            "training and which for testing; if a key contains 'train' the corresponding vector will be used for " +
            "training")
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(outputFileOption)
        .withOption(testPercentOption)
        .withOption(useKeysOption)
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
    if (!cmdLine.hasOption(useKeysOption)) {
      Preconditions.checkNotNull(cmdLine.getValue(testPercentOption),
          "testPercent not set but useKeys also not set");
      testPercent = Double.parseDouble((String) cmdLine.getValue(testPercentOption));
      Preconditions.checkArgument(testPercent > 0 && testPercent < 1, "invalid percent for test set");
      useKeys = false;
    } else {
      useKeys = true;
    }
    return true;
  }

  public static void main(String[] args) throws IOException {
    SplitDatasetTrainTest runner = new SplitDatasetTrainTest();
    if (runner.parseArgs(args)) {
      runner.run(new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true));
    }
  }
}
