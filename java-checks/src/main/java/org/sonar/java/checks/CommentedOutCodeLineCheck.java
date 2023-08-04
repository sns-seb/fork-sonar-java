/*
 * SonarQube Java
 * Copyright (C) 2012-2023 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.java.checks;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.apache.commons.lang3.StringUtils;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.check.Rule;
import org.sonar.java.checks.s125model.CommentPreparation;
import org.sonar.java.checks.s125model.FeatureExtractor;
import org.sonar.java.checks.s125model.Model;
import org.sonar.java.checks.s125model.RoBERTaBPEEncoder;
import org.sonar.java.checks.s125model.RoBERTaTokenizer;
import org.sonar.java.model.DefaultJavaFileScannerContext;
import org.sonar.java.model.LineUtils;
import org.sonar.java.reporting.AnalyzerMessage;
import org.sonar.java.reporting.JavaComment;
import org.sonar.plugins.java.api.IssuableSubscriptionVisitor;
import org.sonar.plugins.java.api.location.Position;
import org.sonar.plugins.java.api.tree.SyntaxToken;
import org.sonar.plugins.java.api.tree.SyntaxTrivia;
import org.sonar.plugins.java.api.tree.Tree;
import org.sonarsource.analyzer.commons.annotations.DeprecatedRuleKey;
import org.sonarsource.analyzer.commons.recognizers.CodeRecognizer;

@DeprecatedRuleKey(ruleKey = "CommentedOutCodeLine", repositoryKey = "squid")
@Rule(key = "S125")
public class CommentedOutCodeLineCheck extends IssuableSubscriptionVisitor {

  private static final double THRESHOLD = 0.9;
  private static final String START_JSNI = "/*-{";
  private static final String END_JSNI = "}-*/";
  private static final String MESSAGE = "This block of commented-out lines of code should be removed.";

  private final CodeRecognizer codeRecognizer;

  public CommentedOutCodeLineCheck() {
    codeRecognizer = new CodeRecognizer(THRESHOLD, new JavaFootprint());
  }

  @Override
  public List<Tree.Kind> nodesToVisit() {
    return Collections.singletonList(Tree.Kind.TOKEN);
  }

  @Override
  public void visitToken(SyntaxToken syntaxToken) {
    if (context.useS125Model()) {
      try {
        modelVisitToken(syntaxToken);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    } else {
      originalVisitToken(syntaxToken);
    }
  }

  private void modelVisitToken(SyntaxToken syntaxToken) throws IOException {
    List<SyntaxTrivia> lineCommentSeries = new ArrayList<>(1);
    int previousLineCommentLine = -1;
    for (SyntaxTrivia syntaxTrivia : syntaxToken.trivias()) {
      int currentLine = LineUtils.startLine(syntaxTrivia);

      if (isJavadoc(syntaxTrivia.comment())) {
        continue;
      }

      if (syntaxTrivia.isBlock()) {
        // run model on any ongoing line comment series and reset state
        runModel(lineCommentSeries);
        previousLineCommentLine = LineUtils.endLine(syntaxTrivia);
        lineCommentSeries.clear();

        // run the model on block comment
        runModel(List.of(syntaxTrivia));
      } else if (previousLineCommentLine == -1 || currentLine <= previousLineCommentLine + 1) {
        // add line comment to ongoing line comment series (or start it) and move on
        lineCommentSeries.add(syntaxTrivia);
        previousLineCommentLine = currentLine;
      } else {
        // run model on ongoing line comment series and reset state
        runModel(lineCommentSeries);
        previousLineCommentLine = currentLine;
        lineCommentSeries.clear();

        // start a new series of lines comments
        lineCommentSeries.add(syntaxTrivia);
      }
    }

    // run model on any ongoing series
    runModel(lineCommentSeries);
  }

  private static final int MAX_TOKENS_PER_STRING = 500;
  private static final double DECISION_THRESHOLD = 0.83d;

  private CommentPreparation commentPreparation;
  private RoBERTaTokenizer tokenizer;
  private FeatureExtractor featureExtractor;
  private Model model;

  private static final Logger LOGGER = Loggers.get(CommentedOutCodeLineCheck.class);

  private void runModel(List<SyntaxTrivia> triviaSeries) throws IOException {
    if (triviaSeries.isEmpty()) {
      return;
    }

    DefaultJavaFileScannerContext scannerContext = (DefaultJavaFileScannerContext) this.context;
    scannerContext.captureComment(new JavaComment(context.getInputFile(), triviaSeries));

    ensureModelIsLoaded();

    String rawComment = triviaSeries.stream().flatMap(t -> LineUtils.splitLines(t.comment()).stream()).collect(Collectors.joining("\n"));
    String comment = commentPreparation.stripCommentSigns(rawComment);
    String[] tokens = tokenizer.tokenize(comment);
    double[] features = featureExtractor.extractFrom(tokens);
    Model.Prediction prediction = model.predict(features);
    LOGGER.info("LR: " + prediction.getLinearRegression() + ", NR: " + prediction.getNormalizedRegression() + ", Decision:" + prediction.getDecision() + "\n" + rawComment);
    if (prediction.getDecision() == 1) {
      SyntaxTrivia firstTrivia = triviaSeries.get(0);
      Position start = firstTrivia.range().start();

      SyntaxTrivia lastTrivia = triviaSeries.get(triviaSeries.size() - 1);
      Position end = lastTrivia.range().end();

      AnalyzerMessage.TextSpan textSpan = new AnalyzerMessage.TextSpan(
        start.line(), start.column() - 1,
        end.line(), end.column() - 1);

      scannerContext.reportIssue(new AnalyzerMessage(this, context.getInputFile(), textSpan, MESSAGE, 0));
    }
  }

  private void ensureModelIsLoaded() throws IOException {
    if (model == null) {
      this.commentPreparation = new CommentPreparation();
      this.tokenizer = new RoBERTaTokenizer(new RoBERTaBPEEncoder(getClass().getResourceAsStream("s125model/merges.txt")));
      this.featureExtractor = FeatureExtractor.create(getClass().getResourceAsStream("s125model/vocab-100.json"), MAX_TOKENS_PER_STRING);
      this.model = Model.create(getClass().getResourceAsStream("s125model/model-lr-100.json"), DECISION_THRESHOLD);
    }
  }

  private void originalVisitToken(SyntaxToken syntaxToken) {
    DefaultJavaFileScannerContext scannerContext = (DefaultJavaFileScannerContext) this.context;
    List<AnalyzerMessage> issues = new ArrayList<>();
    AnalyzerMessage previousRelatedIssue = null;
    int previousCommentLine = -1;
    for (SyntaxTrivia syntaxTrivia : syntaxToken.trivias()) {
      scannerContext.captureComment(new JavaComment(context.getInputFile(), List.of(syntaxTrivia)));
      int currentCommentLine = LineUtils.startLine(syntaxTrivia);
      if (currentCommentLine != previousCommentLine + 1 &&
        currentCommentLine != previousCommentLine) {
        previousRelatedIssue = null;
      }
      if (!isHeader(syntaxTrivia) && !isJavadoc(syntaxTrivia.comment()) && !isJSNI(syntaxTrivia.comment())) {
        previousRelatedIssue = collectIssues(issues, syntaxTrivia, previousRelatedIssue);
        previousCommentLine = currentCommentLine;
      }
    }
    issues.forEach(scannerContext::reportIssue);
  }

  public AnalyzerMessage collectIssues(List<AnalyzerMessage> issues, SyntaxTrivia syntaxTrivia, @Nullable AnalyzerMessage previousRelatedIssue) {
    String[] lines = syntaxTrivia.comment().split("\r\n?|\n");
    AnalyzerMessage issue = previousRelatedIssue;
    for (int lineOffset = 0; lineOffset < lines.length; lineOffset++) {
      String line = lines[lineOffset];
      if (!isJavadocLink(line) && codeRecognizer.isLineOfCode(line)) {
        int startLine = LineUtils.startLine(syntaxTrivia) + lineOffset;
        int startColumnOffset = (lineOffset == 0 ? Position.startOf(syntaxTrivia).columnOffset() : 0);
        if (issue != null) {
          issue.flows.add(Collections.singletonList(createAnalyzerMessage(startLine, startColumnOffset, line, "Code")));
        } else {
          issue = createAnalyzerMessage(startLine, startColumnOffset, line, MESSAGE);
          issues.add(issue);
        }
      }
    }
    return issue;
  }

  private AnalyzerMessage createAnalyzerMessage(int startLine, int startColumn, String line, String message) {
    String lineWithoutCommentPrefix = line.replaceFirst("^(//|/\\*\\*?|[ \t]*\\*)?[ \t]*+", "");
    int prefixSize = line.length() - lineWithoutCommentPrefix.length();
    String lineWithoutCommentPrefixAndSuffix = removeCommentSuffix(lineWithoutCommentPrefix);

    AnalyzerMessage.TextSpan textSpan = new AnalyzerMessage.TextSpan(
      startLine,
      startColumn + prefixSize,
      startLine,
      startColumn + prefixSize + lineWithoutCommentPrefixAndSuffix.length());

    return new AnalyzerMessage(this, context.getInputFile(), textSpan, message, 0);
  }

  private static String removeCommentSuffix(String line) {
    // We do not use a regex for this task, to avoid ReDoS.
    if (line.endsWith("*/")) {
      line = line.substring(0, line.length() - 2);
    }
    return line.stripTrailing();
  }

  /**
   * We assume that comment on a first line - is a header with license.
   * However possible to imagine corner case: file may contain commented-out code starting from first line.
   * But we assume that probability of this is really low.
   */
  private static boolean isHeader(SyntaxTrivia syntaxTrivia) {
    return LineUtils.startLine(syntaxTrivia) == 1;
  }

  private static boolean isJavadocLink(String line) {
    return line.contains("{@link");
  }

  /**
   * From documentation for Javadoc-tool:
   * Documentation comments should be recognized only when placed
   * immediately before class, interface, constructor, method, or field declarations.
   */
  private static boolean isJavadoc(String comment) {
    return StringUtils.startsWith(comment, "/**");
  }

  /**
   * From GWT documentation:
   * JSNI methods are declared native and contain JavaScript code in a specially formatted comment block
   * between the end of the parameter list and the trailing semicolon.
   * A JSNI comment block begins with the exact token {@link #START_JSNI} and ends with the exact token {@link #END_JSNI}.
   */
  private static boolean isJSNI(String comment) {
    return StringUtils.startsWith(comment, START_JSNI) && StringUtils.endsWith(comment, END_JSNI);
  }

}
